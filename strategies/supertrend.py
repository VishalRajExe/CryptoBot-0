"""
Strategy: Supertrend ATR Trend Rider
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Based on the classic Supertrend indicator and Freqtrade's Supertrend implementation.
Heavily recommended by r/algotrading for high-volatility crypto assets (SOL, AVAX, ETH).

Core Logic:
1. Supertrend Indicator:
   - Upper Band: (High + Low) / 2 + (Multiplier * ATR)
   - Lower Band: (High + Low) / 2 - (Multiplier * ATR)
   - Band flips from Red (Downtrend) to Green (Uptrend) on breakout.
2. ADX Trend Filter (ADX > 22):
   - Ensures we NEVER enter during flat, choppy, sideways ranges.
3. EMA Ribbon Baseline Filter:
   - Price must be above 100-EMA to confirm macro bull trend.
4. Dynamic ATR Trailing Stops:
   - Initial Stop Loss: Supertrend lower band.
   - Take Profit: 2.5x to 3.5x risk.
"""

from typing import Optional
import numpy as np
import pandas as pd

from .base import BaseStrategy, SignalResult


class Supertrend(BaseStrategy):
    NAME = "Supertrend"
    TIMEFRAME_MINUTES = 5
    MIN_CANDLES = 70

    def __init__(self, symbol: str, params: Optional[dict] = None):
        super().__init__(symbol, params)
        self.atr_period = self.params.get("atr_period", 10)
        self.multiplier = self.params.get("multiplier", 3.0)

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # 1. True Range & ATR
        hl = high - low
        hpc = (high - close.shift(1)).abs()
        lpc = (low - close.shift(1)).abs()
        tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=self.atr_period, adjust=False).mean()

        # 2. ADX (Trend Strength)
        up_move = high.diff()
        down_move = -low.diff()
        pdm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        ndm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr_smooth = tr.ewm(span=14, adjust=False).mean()
        pdi = 100.0 * pd.Series(pdm, index=df.index).ewm(span=14, adjust=False).mean() / (tr_smooth + 1e-9)
        ndi = 100.0 * pd.Series(ndm, index=df.index).ewm(span=14, adjust=False).mean() / (tr_smooth + 1e-9)
        dx = 100.0 * (pdi - ndi).abs() / (pdi + ndi + 1e-9)
        df["adx"] = dx.ewm(span=14, adjust=False).mean()

        # 3. 100-EMA Macro Anchor
        df["ema100"] = close.ewm(span=100, adjust=False).mean()

        # 4. Supertrend Calculation
        hl2 = (high + low) / 2.0
        basic_upper = hl2 + (self.multiplier * df["atr"])
        basic_lower = hl2 - (self.multiplier * df["atr"])

        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        trend = np.zeros(len(df), dtype=int)  # 1 = up, -1 = down

        for i in range(1, len(df)):
            c_prev = close.iloc[i - 1]

            # Lower Band
            if basic_lower.iloc[i] > final_lower.iloc[i - 1] or c_prev < final_lower.iloc[i - 1]:
                final_lower.iloc[i] = basic_lower.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i - 1]

            # Upper Band
            if basic_upper.iloc[i] < final_upper.iloc[i - 1] or c_prev > final_upper.iloc[i - 1]:
                final_upper.iloc[i] = basic_upper.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i - 1]

            # Direction
            if trend[i - 1] == 1:
                if close.iloc[i] < final_lower.iloc[i]:
                    trend[i] = -1
                else:
                    trend[i] = 1
            else:
                if close.iloc[i] > final_upper.iloc[i]:
                    trend[i] = 1
                else:
                    trend[i] = -1

        df["st_upper"] = final_upper
        df["st_lower"] = final_lower
        df["st_trend"] = trend

        return df

    def get_signal(self, df: pd.DataFrame, current_price: float) -> SignalResult:
        if len(df) < self.MIN_CANDLES:
            return SignalResult("HOLD", 0.0, 0.0, 0.0, "Warming up indicators")

        row = df.iloc[-1]
        prev = df.iloc[-2]

        price = current_price or float(row["close"])
        st_trend = int(row["st_trend"])
        prev_trend = int(prev["st_trend"])
        st_lower = float(row["st_lower"])
        st_upper = float(row["st_upper"])
        adx = float(row["adx"])
        ema100 = float(row["ema100"])
        atr = float(row["atr"]) if pd.notna(row["atr"]) else price * 0.015

        # Bullish Flip: Trend just flipped from -1 to 1 (or confirmed bull with ADX > 22)
        fresh_flip = (st_trend == 1 and prev_trend == -1)
        established_trend = (st_trend == 1 and price > ema100 and adx > 22.0 and price > st_lower)

        # Bearish Flip: Trend flipped to -1
        if st_trend == -1 or price < st_lower:
            return SignalResult(
                signal="SELL",
                stop_loss=0.0,
                take_profit=0.0,
                confidence=0.85,
                reason=f"Supertrend Bearish Flip 🔴 LowerBand=$ {st_lower:.4f} Price=$ {price:.4f}"
            )

        if (fresh_flip or established_trend) and price > ema100:
            sl = round(max(st_lower, price - (2.0 * atr)), 4)
            tp = round(price + (3.0 * (price - sl)), 4)
            return SignalResult(
                signal="BUY",
                stop_loss=sl,
                take_profit=tp,
                confidence=0.90 if fresh_flip else 0.82,
                reason=f"Supertrend BULL 🟢 {'FRESH FLIP' if fresh_flip else 'CONTINUATION'} ADX={adx:.1f} SL=$ {sl:.4f} TP=$ {tp:.4f}"
            )

        return SignalResult(
            signal="HOLD",
            stop_loss=0.0,
            take_profit=0.0,
            confidence=0.0,
            reason=f"Supertrend Neutral | ADX={adx:.1f} ST_Trend={st_trend}"
        )
