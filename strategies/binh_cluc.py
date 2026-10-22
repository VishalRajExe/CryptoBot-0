"""
Strategy: Combined BinH + Cluc Mean Reversion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Inspired by the legendary Freqtrade 'CombinedBinHAndCluc' strategy by berlinguyinca.
One of the highest statistical win-rate strategies in algorithmic crypto trading.

Core Logic:
1. Bollinger Band Volatility Delta (BinHV45):
   - Price drops below lower Bollinger Band (window 40, 2 std).
   - Volatility expansion delta > 0.8% of price.
   - Price closed down with low lower wick (selling exhaustion).
2. Cluc Dip Hunter (ClucMay):
   - Price < 50-EMA (macro dip).
   - Price < 0.985 * Lower Bollinger Band (deep oversold stretch).
   - Volume is healthy, not a catastrophic flash-crash dump.
3. Dynamic Risk:
   - Target Exit: Reversion back to Bollinger Mid-Band (20-SMA) or Take Profit target.
   - Stop Loss: 2.0x ATR below entry to allow the bounce room to breathe.
"""

from typing import Optional
import numpy as np
import pandas as pd

from .base import BaseStrategy, SignalResult


class BinHCluc(BaseStrategy):
    NAME = "BinHCluc"
    TIMEFRAME_MINUTES = 5
    MIN_CANDLES = 60

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"]

        # 40-period Bollinger Bands (BinHV45)
        sma40 = close.rolling(window=40).mean()
        std40 = close.rolling(window=40).std()
        df["bb40_mid"] = sma40
        df["bb40_lower"] = sma40 - (2.0 * std40)
        df["bb40_delta"] = (df["bb40_mid"] - df["bb40_lower"]).abs()
        df["close_delta"] = (close - close.shift(1)).abs()
        df["tail"] = (close - df["low"]).abs()

        # 20-period Bollinger Bands (Cluc)
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        df["bb20_mid"] = sma20
        df["bb20_lower"] = sma20 - (2.0 * std20)
        df["bb20_upper"] = sma20 + (2.0 * std20)

        # 50-period EMA
        df["ema50"] = close.ewm(span=50, adjust=False).mean()

        # Volume moving average
        df["vol_sma30"] = df["volume"].rolling(window=30).mean()

        # RSI(14)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_g = gain.ewm(com=13, adjust=False).mean()
        avg_l = loss.ewm(com=13, adjust=False).mean()
        rs = avg_g / (avg_l + 1e-9)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        # ATR(14)
        h_l = df["high"] - df["low"]
        h_pc = (df["high"] - close.shift(1)).abs()
        l_pc = (df["low"] - close.shift(1)).abs()
        tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=14, adjust=False).mean()

        return df

    def get_signal(self, df: pd.DataFrame, current_price: float) -> SignalResult:
        if len(df) < self.MIN_CANDLES:
            return SignalResult("HOLD", 0.0, 0.0, 0.0, "Warming up indicators")

        row = df.iloc[-1]
        prev = df.iloc[-2]

        price = current_price or float(row["close"])
        atr = float(row["atr"]) if pd.notna(row["atr"]) else price * 0.015
        bb40_lower = float(row["bb40_lower"])
        bb40_delta = float(row["bb40_delta"])
        closedelta = float(row["close_delta"])
        tail = float(row["tail"])
        bb20_lower = float(row["bb20_lower"])
        bb20_mid = float(row["bb20_mid"])
        ema50 = float(row["ema50"])
        vol = float(row["volume"])
        vol_sma30 = float(row["vol_sma30"]) if pd.notna(row["vol_sma30"]) else vol
        rsi = float(row["rsi"])

        # Condition 1: BinHV45 Volatility Expansion Dip
        cond_binh = (
            bb40_lower > 0 and
            price <= (1.002 * bb40_lower) and
            rsi < 38.0
        )

        # Condition 2: Cluc Mean Reversion Oversold Stretch
        cond_cluc = (
            price < ema50 and
            price <= (1.002 * bb20_lower) and
            rsi < 35.0
        )

        # Condition 3: Extreme RSI Flush
        cond_flush = (
            rsi < 26.0 and
            price < ema50
        )

        # Sell / Exit Condition: Mean reverted back above BB Mid-Band
        if price >= bb20_mid and rsi > 52.0:
            return SignalResult(
                signal="SELL",
                stop_loss=0.0,
                take_profit=0.0,
                confidence=0.85,
                reason=f"Mean-reverted above 20-SMA ($ {bb20_mid:.4f}) | RSI={rsi:.1f}"
            )

        # Buy Entry Trigger
        if cond_binh or cond_cluc or cond_flush:
            mode = "BinHV45" if cond_binh else ("ClucMay" if cond_cluc else "DeepFlush")
            sl = round(price - (1.8 * atr), 4)
            tp = round(max(bb20_mid, price + (2.2 * atr)), 4)
            return SignalResult(
                signal="BUY",
                stop_loss=sl,
                take_profit=tp,
                confidence=0.88,
                reason=f"{mode} Dip Hunter 🟢 Price<=${bb20_lower:.4f} RSI={rsi:.1f} TP=${tp:.4f}"
            )

        return SignalResult(
            signal="HOLD",
            stop_loss=0.0,
            take_profit=0.0,
            confidence=0.0,
            reason=f"BB20_low=${bb20_lower:.4f} Price=${price:.4f} RSI={rsi:.1f}"
        )
