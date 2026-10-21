"""
Strategy: Scalping – MACD + RSI on 1-minute bars
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUY  when:
  1. Price > EMA(200)           (trend filter)
  2. MACD line crosses above signal line   (momentum entry)
  3. RSI 40–65                  (not exhausted)

SELL / EXIT:
  - Dynamic ATR Stop-Loss (1.0x ATR)
  - ATR Take-Profit (1.5x ATR for 1:1.5 R/R)

Best for: SOL/USD, DOGE/USD  (high-volatility fast movers)
"""
import pandas as pd
from .base import BaseStrategy, SignalResult


class ScalpMacdRsi(BaseStrategy):
    NAME = "ScalpMacdRsi"
    TIMEFRAME_MINUTES = 1
    MIN_CANDLES = 250  # need 200-EMA to stabilise

    def __init__(self, symbol: str, params: dict = None):
        defaults = {
            "ema_slow": 200,
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "rsi_period": 14, "rsi_min": 40, "rsi_max": 65,
            "atr_period": 14, "atr_sl_mult": 1.0, "rr_ratio": 1.5,
        }
        super().__init__(symbol, {**defaults, **(params or {})})

    # ── Indicators ────────────────────────────────────────────────────────────
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        df = df.copy()
        close = df["close"]

        # EMA trend filter
        df["ema200"] = close.ewm(span=p["ema_slow"], adjust=False).mean()

        # MACD
        ema_fast = close.ewm(span=p["macd_fast"], adjust=False).mean()
        ema_slow = close.ewm(span=p["macd_slow"], adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_sig"] = df["macd"].ewm(span=p["macd_signal"], adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_sig"]
        df["macd_cross_up"]   = (df["macd"] > df["macd_sig"]) & (df["macd"].shift(1) <= df["macd_sig"].shift(1))
        df["macd_cross_down"] = (df["macd"] < df["macd_sig"]) & (df["macd"].shift(1) >= df["macd_sig"].shift(1))

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        ag    = gain.ewm(com=p["rsi_period"] - 1, min_periods=p["rsi_period"]).mean()
        al    = loss.ewm(com=p["rsi_period"] - 1, min_periods=p["rsi_period"]).mean()
        rs    = ag / al.replace(0, float("nan"))
        df["rsi"] = 100 - (100 / (1 + rs))

        # ATR
        hl   = df["high"] - df["low"]
        hc   = (df["high"] - close.shift()).abs()
        lc   = (df["low"]  - close.shift()).abs()
        tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=p["atr_period"], adjust=False).mean()

        return df

    # ── Signal ────────────────────────────────────────────────────────────────
    def get_signal(self, df: pd.DataFrame, current_price: float) -> SignalResult:
        p = self.params
        if len(df) < self.MIN_CANDLES:
            return SignalResult("HOLD", 0, 0, 0.0, "Not enough bars")

        last = df.iloc[-1]
        atr  = last["atr"] if pd.notna(last["atr"]) else current_price * 0.005
        risk = atr * p["atr_sl_mult"]

        trend_up   = current_price > last["ema200"]
        macd_bull  = bool(last["macd_cross_up"])
        rsi_ok     = p["rsi_min"] <= last["rsi"] <= p["rsi_max"]

        trend_down  = current_price < last["ema200"]
        macd_bear   = bool(last["macd_cross_down"])
        rsi_sell_ok = 35 <= last["rsi"] <= 60

        if trend_up and macd_bull and rsi_ok:
            sl = round(current_price - risk, 4)
            tp = round(current_price + risk * p["rr_ratio"], 4)
            conf = min(1.0, (last["rsi"] - 40) / 25 + 0.4)
            return SignalResult("BUY", sl, tp, conf,
                                f"EMA200✅ MACD↑✅ RSI={last['rsi']:.1f}✅")

        if trend_down and macd_bear and rsi_sell_ok:
            sl = round(current_price + risk, 4)
            tp = round(current_price - risk * p["rr_ratio"], 4)
            return SignalResult("SELL", sl, tp, 0.6,
                                f"EMA200↓✅ MACD↓✅ RSI={last['rsi']:.1f}✅")

        return SignalResult("HOLD", 0, 0, 0.0,
                            f"RSI={last['rsi']:.1f} MACD_hist={last['macd_hist']:.4f}")
