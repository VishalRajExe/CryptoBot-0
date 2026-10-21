"""
Strategy: SmoothScalp StochCCI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Based on berlinguyinca's famous 'SmoothScalp' strategy from freqtrade-strategies.
Tailored for fast-moving, high-frequency scalping (SOL, DOGE) on 1-minute and 3-minute bars.

Core Logic:
1. EMA High/Low Envelopes (5-period on High and Low).
2. Fast Stochastic (%K, %D) Oversold Crossover:
   - %K and %D both < 25 (deep oversold).
   - %K crosses above %D (momentum reversal upward).
3. Commodity Channel Index (CCI):
   - CCI(20) < -100 (extreme statistical price compression).
4. Money Flow Index (MFI) & ADX:
   - ADX > 22 (sufficient market volatility to ensure quick fills).
5. Quick Scalp Exits:
   - Takes profit quickly (0.8% - 1.5%) when Stoch cross above 70 or price hits EMA High.
   - Dynamic stop loss (1.0% - 1.2%).
"""

from typing import Optional
import numpy as np
import pandas as pd

from .base import BaseStrategy, SignalResult


class SmoothScalp(BaseStrategy):
    NAME = "SmoothScalp"
    TIMEFRAME_MINUTES = 1
    MIN_CANDLES = 50

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # 5-EMA on High, Close, Low
        df["ema5_high"] = high.ewm(span=5, adjust=False).mean()
        df["ema5_close"] = close.ewm(span=5, adjust=False).mean()
        df["ema5_low"] = low.ewm(span=5, adjust=False).mean()

        # Fast Stochastic (%K, %D)
        low14 = low.rolling(window=14).min()
        high14 = high.rolling(window=14).max()
        fastk = 100.0 * (close - low14) / ((high14 - low14) + 1e-9)
        df["fastk"] = fastk.rolling(window=3).mean()
        df["fastd"] = df["fastk"].rolling(window=3).mean()

        # CCI(20)
        tp = (high + low + close) / 3.0
        sma_tp = tp.rolling(window=20).mean()
        mad = (tp - sma_tp).abs().rolling(window=20).mean()
        df["cci"] = (tp - sma_tp) / (0.015 * mad + 1e-9)

        # ADX(14)
        hl = high - low
        hpc = (high - close.shift(1)).abs()
        lpc = (low - close.shift(1)).abs()
        tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)

        up = high.diff()
        down = -low.diff()
        pdm = np.where((up > down) & (up > 0), up, 0.0)
        ndm = np.where((down > up) & (down > 0), down, 0.0)

        tr_smooth = tr.ewm(span=14, adjust=False).mean()
        pdi = 100.0 * pd.Series(pdm, index=df.index).ewm(span=14, adjust=False).mean() / (tr_smooth + 1e-9)
        ndi = 100.0 * pd.Series(ndm, index=df.index).ewm(span=14, adjust=False).mean() / (tr_smooth + 1e-9)
        dx = 100.0 * (pdi - ndi).abs() / (pdi + ndi + 1e-9)
        df["adx"] = dx.ewm(span=14, adjust=False).mean()

        # ATR(14)
        df["atr"] = tr.ewm(span=14, adjust=False).mean()

        return df

    def get_signal(self, df: pd.DataFrame, current_price: float) -> SignalResult:
        if len(df) < self.MIN_CANDLES:
            return SignalResult("HOLD", 0.0, 0.0, 0.0, "Warming up indicators")

        row = df.iloc[-1]
        prev = df.iloc[-2]

        price = current_price or float(row["close"])
        fastk = float(row["fastk"])
        fastd = float(row["fastd"])
        prev_k = float(prev["fastk"])
        prev_d = float(prev["fastd"])
        cci = float(row["cci"])
        adx = float(row["adx"])
        ema5_low = float(row["ema5_low"])
        ema5_high = float(row["ema5_high"])
        atr = float(row["atr"]) if pd.notna(row["atr"]) else price * 0.01

        # Stoch Cross Above in Oversold (< 28)
        stoch_cross = (prev_k <= prev_d) and (fastk > fastd) and (fastk < 28.0) and (fastd < 28.0)
        deep_dip = (price <= ema5_low) and (cci < -90.0)
        active_market = adx > 20.0

        # Exit signal: Stoch reaches overbought > 70 or price breaks above 5-EMA High
        if (fastk > 70.0 and prev_k <= 70.0) or (price >= ema5_high and cci > 100.0):
            return SignalResult(
                signal="SELL",
                stop_loss=0.0,
                take_profit=0.0,
                confidence=0.85,
                reason=f"Scalp Target Hit 🎯 FastK={fastk:.1f} CCI={cci:.1f}"
            )

        # Entry signal
        if stoch_cross and (deep_dip or cci < -80.0) and active_market:
            sl = round(price - max(1.2 * atr, price * 0.009), 4)
            tp = round(price + max(1.8 * atr, price * 0.014), 4)
            return SignalResult(
                signal="BUY",
                stop_loss=sl,
                take_profit=tp,
                confidence=0.88,
                reason=f"SmoothScalp Entry 🟢 StochCross={fastk:.1f}/{fastd:.1f} CCI={cci:.1f} ADX={adx:.1f}"
            )

        return SignalResult(
            signal="HOLD",
            stop_loss=0.0,
            take_profit=0.0,
            confidence=0.0,
            reason=f"Stoch %K={fastk:.1f} %D={fastd:.1f} CCI={cci:.1f}"
        )
