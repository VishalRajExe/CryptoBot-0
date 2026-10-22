"""
Strategy: VWAP Breakout on 15-minute bars
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VWAP = session-anchored Volume Weighted Average Price.
Resets at 00:00 UTC each day to capture intraday momentum.

BUY  when:
  1. Price crosses ABOVE VWAP (bullish VWAP break)
  2. Volume > 1.5x 20-bar volume MA  (strong conviction)
  3. MACD histogram turning positive  (momentum confirming)

Best for: ETH/USD, SOL/USD (high-liquidity VWAP response)
"""
import pandas as pd
from .base import BaseStrategy, SignalResult


class VwapBreakout(BaseStrategy):
    NAME = "VwapBreakout"
    TIMEFRAME_MINUTES = 15
    MIN_CANDLES = 50

    def __init__(self, symbol: str, params: dict = None):
        defaults = {
            "vol_ma_period": 20,
            "vol_threshold": 1.5,       # volume must be 1.5x the MA
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "atr_period": 14, "atr_sl_mult": 1.5, "rr_ratio": 2.0,
        }
        super().__init__(symbol, {**defaults, **(params or {})})

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        df = df.copy()
        close = df["close"]

        # VWAP — reset per UTC calendar day
        df["tp"]     = (df["high"] + df["low"] + close) / 3
        df["tp_vol"] = df["tp"] * df["volume"]
        df["date"]   = df.index.normalize()          # UTC day bucket
        df["cum_tp_vol"] = df.groupby("date")["tp_vol"].cumsum()
        df["cum_vol"]    = df.groupby("date")["volume"].cumsum()
        df["vwap"]   = df["cum_tp_vol"] / df["cum_vol"].replace(0, float("nan"))

        # VWAP cross flags
        df["above_vwap"] = close > df["vwap"]
        df["vwap_cross_up"]   = df["above_vwap"] & ~df["above_vwap"].shift(1).fillna(False)
        df["vwap_cross_down"] = ~df["above_vwap"] & df["above_vwap"].shift(1).fillna(True)

        # Volume filter
        df["vol_ma"]     = df["volume"].rolling(p["vol_ma_period"]).mean()
        df["vol_surge"]  = df["volume"] > (p["vol_threshold"] * df["vol_ma"])

        # MACD histogram direction
        ema_f    = close.ewm(span=p["macd_fast"],   adjust=False).mean()
        ema_s    = close.ewm(span=p["macd_slow"],   adjust=False).mean()
        macd     = ema_f - ema_s
        sig      = macd.ewm(span=p["macd_signal"],  adjust=False).mean()
        df["macd_hist"] = macd - sig
        df["macd_positive"] = df["macd_hist"] > 0

        # RSI(14) for AI feature vectors
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
        df["macd_negative"] = df["macd_hist"] < 0

        # ATR
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - close.shift()).abs()
        lc  = (df["low"]  - close.shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=p["atr_period"], adjust=False).mean()

        return df

    def get_signal(self, df: pd.DataFrame, current_price: float) -> SignalResult:
        p = self.params
        if len(df) < self.MIN_CANDLES:
            return SignalResult("HOLD", 0, 0, 0.0, "Not enough bars")

        last = df.iloc[-1]
        prev = df.iloc[-2]
        atr  = float(last["atr"]) if pd.notna(last["atr"]) else current_price * 0.015
        risk = atr * p["atr_sl_mult"]
        vwap = float(last["vwap"]) if pd.notna(last["vwap"]) else current_price
        rsi  = float(last["rsi"]) if "rsi" in last and pd.notna(last["rsi"]) else 50.0
        macd_hist = float(last.get("macd_hist", 0.0))
        prev_hist = float(prev.get("macd_hist", 0.0))
        macd_rising = macd_hist > prev_hist

        cross_up   = bool(last.get("vwap_cross_up", False))
        above_vwap = current_price >= (0.998 * vwap)

        # ── BUY 1: Oversold Dip Reversal under VWAP ──────────────────────────
        if rsi < 35.0 and current_price < vwap:
            sl = round(current_price - risk, 4)
            tp = round(max(vwap, current_price + risk * p["rr_ratio"]), 4)
            return SignalResult("BUY", sl, tp, 0.82,
                                f"VWAP Oversold Dip 🟢 RSI={rsi:.1f} VWAP=${vwap:.2f} TP=${tp:.4f}")

        # ── BUY 2: Bullish VWAP Breakout or Continuation ─────────────────────
        if (cross_up or (above_vwap and macd_rising)) and (42.0 <= rsi <= 68.0):
            sl = round(current_price - risk, 4)
            tp = round(current_price + risk * p["rr_ratio"], 4)
            mode = "VWAP Cross↑" if cross_up else "VWAP Momentum↑"
            return SignalResult("BUY", sl, tp, 0.78,
                                f"{mode} 🟢 RSI={rsi:.1f} VWAP=${vwap:.2f} TP=${tp:.4f}")

        # ── SELL / EXIT: Severe breakdown or overbought ───────────────────────
        if rsi > 70.0 and current_price > (1.02 * vwap):
            sl = round(current_price + risk, 4)
            tp = round(current_price - risk * p["rr_ratio"], 4)
            return SignalResult("SELL", sl, tp, 0.75,
                                f"VWAP Overbought Extension 🔴 RSI={rsi:.1f}")

        return SignalResult("HOLD", 0, 0, 0.0,
                            f"VWAP=${vwap:.2f} Price=${current_price:.2f} RSI={rsi:.1f}")
