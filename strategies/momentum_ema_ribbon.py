"""
Strategy: Momentum – EMA Ribbon Breakout on 5-minute bars
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uses an EMA Ribbon (8, 13, 21, 34, 55) to identify ribbon
alignment. When all EMAs stack in order, trend is strong.

BUY  when:
  1. EMA8 > EMA13 > EMA21 > EMA34 > EMA55   (full bull ribbon)
  2. Price > EMA8                              (price above ribbon)
  3. RSI > 50                                 (bullish momentum)
  4. ADX > 20                                 (trending, not ranging)

Best for: ETH/USD, LINK/USD  (liquid, trend-following assets)
"""
import pandas as pd
from .base import BaseStrategy, SignalResult


class MomentumEmaRibbon(BaseStrategy):
    NAME = "MomentumEmaRibbon"
    TIMEFRAME_MINUTES = 5
    MIN_CANDLES = 100

    def __init__(self, symbol: str, params: dict = None):
        defaults = {
            "ribbon": [8, 13, 21, 34, 55],
            "rsi_period": 14, "rsi_threshold": 50,
            "adx_period": 14, "adx_threshold": 20,
            "atr_period": 14, "atr_sl_mult": 1.5, "rr_ratio": 2.0,
        }
        super().__init__(symbol, {**defaults, **(params or {})})

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        df = df.copy()
        close = df["close"]

        # EMA Ribbon
        for span in p["ribbon"]:
            df[f"ema{span}"] = close.ewm(span=span, adjust=False).mean()

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        ag    = gain.ewm(com=p["rsi_period"] - 1, min_periods=p["rsi_period"]).mean()
        al    = loss.ewm(com=p["rsi_period"] - 1, min_periods=p["rsi_period"]).mean()
        rs    = ag / al.replace(0, float("nan"))
        df["rsi"] = 100 - (100 / (1 + rs))

        # ATR
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - close.shift()).abs()
        lc  = (df["low"]  - close.shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=p["atr_period"], adjust=False).mean()

        # ADX (manual calculation)
        up_move   = df["high"].diff()
        down_move = -df["low"].diff()
        plus_dm   = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm  = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        atr_s     = tr.ewm(span=p["adx_period"], adjust=False).mean()
        plus_di   = 100 * (plus_dm.ewm(span=p["adx_period"], adjust=False).mean() / atr_s)
        minus_di  = 100 * (minus_dm.ewm(span=p["adx_period"], adjust=False).mean() / atr_s)
        dx        = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1))
        df["adx"] = dx.ewm(span=p["adx_period"], adjust=False).mean()

        return df

    def get_signal(self, df: pd.DataFrame, current_price: float) -> SignalResult:
        p = self.params
        if len(df) < self.MIN_CANDLES:
            return SignalResult("HOLD", 0, 0, 0.0, "Not enough bars")

        last   = df.iloc[-1]
        ribbon = p["ribbon"]
        atr    = last["atr"]
        risk   = atr * p["atr_sl_mult"]

        # Bull ribbon: each EMA > next larger EMA
        bull_ribbon = all(last[f"ema{ribbon[i]}"] > last[f"ema{ribbon[i+1]}"]
                          for i in range(len(ribbon) - 1))
        # Bear ribbon
        bear_ribbon = all(last[f"ema{ribbon[i]}"] < last[f"ema{ribbon[i+1]}"]
                          for i in range(len(ribbon) - 1))

        price_above = current_price > last[f"ema{ribbon[0]}"]
        price_below = current_price < last[f"ema{ribbon[0]}"]
        rsi_bull    = last["rsi"] > p["rsi_threshold"]
        rsi_bear    = last["rsi"] < (100 - p["rsi_threshold"])
        trending    = last["adx"] > p["adx_threshold"]

        if bull_ribbon and price_above and rsi_bull and trending:
            sl   = round(current_price - risk, 4)
            tp   = round(current_price + risk * p["rr_ratio"], 4)
            conf = min(1.0, last["adx"] / 50)
            return SignalResult("BUY", sl, tp, conf,
                                f"EMA Ribbon🟢 RSI={last['rsi']:.1f} ADX={last['adx']:.1f}")

        if bear_ribbon and price_below and rsi_bear and trending:
            sl = round(current_price + risk, 4)
            tp = round(current_price - risk * p["rr_ratio"], 4)
            return SignalResult("SELL", sl, tp, 0.6,
                                f"EMA Ribbon🔴 RSI={last['rsi']:.1f} ADX={last['adx']:.1f}")

        return SignalResult("HOLD", 0, 0, 0.0,
                            f"Ribbon={'BULL' if bull_ribbon else 'BEAR' if bear_ribbon else 'MIXED'} ADX={last['adx']:.1f}")
