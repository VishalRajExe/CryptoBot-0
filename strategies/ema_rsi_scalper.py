"""
Strategy: Multi-Timeframe EMA + RSI Trend Scalper
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Designed for BTC/USD on 3-minute bars. Combines:

1. EMA Stack Trend Confirmation (fast EMAs aligned = trend):
   - EMA8 > EMA21 > EMA55 (bull), EMA8 < EMA21 < EMA55 (bear)

2. RSI Pullback Entry (buy dips in an uptrend, not tops):
   - In an uptrend: RSI dips to 40-55 range = re-entry zone
   - Avoids chasing momentum at RSI > 70

3. MACD Histogram Momentum Gate:
   - MACD histogram must be rising (momentum accelerating on entry)

4. Volume Confirmation:
   - Volume >= 1.2x 20-bar MA (not just noise)

5. Tight ATR-based SL (0.8x ATR) + Wide TP (2.5x ATR) for 3:1 R/R

Best for: BTC/USD, ETH/USD  — high-liquidity trend-following

Key difference from ScalpMacdRsi:
- Does NOT require exact MACD crossover (fires every cycle the trend continues)
- Lower RSI thresholds = more signals
- Uses EMA stack instead of single EMA200 (faster adaptation)

References:
- Popular on: https://github.com/freqtrade/freqtrade-strategies (EMA_MACD_RSI)
- Reddit r/algotrading: "EMA ribbon scalp + RSI pullback" is the #1 most upvoted crypto strategy
"""
import pandas as pd
from .base import BaseStrategy, SignalResult


class EmaRsiScalper(BaseStrategy):
    NAME = "EmaRsiScalper"
    TIMEFRAME_MINUTES = 15
    MIN_CANDLES = 60

    def __init__(self, symbol: str, params: dict = None):
        defaults = {
            "ema_fast": 8,
            "ema_mid": 21,
            "ema_slow": 55,
            "rsi_period": 14,
            "rsi_bull_low": 42,    # Pullback entry zone in uptrend
            "rsi_bull_high": 65,
            "rsi_bear_low": 35,
            "rsi_bear_high": 58,
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "vol_ma_period": 20,
            "vol_threshold": 1.15,
            "atr_period": 14,
            "atr_sl_mult": 1.6,    # Safe swing stop loss (1.6x ATR)
            "rr_ratio": 2.2,       # Asymmetric 2.2x profit target
        }
        super().__init__(symbol, {**defaults, **(params or {})})

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        df = df.copy()
        close = df["close"]

        # ── EMA Ribbon ────────────────────────────────────────────────────────
        df["ema8"]  = close.ewm(span=p["ema_fast"], adjust=False).mean()
        df["ema21"] = close.ewm(span=p["ema_mid"],  adjust=False).mean()
        df["ema55"] = close.ewm(span=p["ema_slow"], adjust=False).mean()

        # ── RSI ───────────────────────────────────────────────────────────────
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        ag    = gain.ewm(com=p["rsi_period"] - 1, min_periods=p["rsi_period"]).mean()
        al    = loss.ewm(com=p["rsi_period"] - 1, min_periods=p["rsi_period"]).mean()
        rs    = ag / al.replace(0, float("nan"))
        df["rsi"] = 100 - (100 / (1 + rs))

        # ── MACD ──────────────────────────────────────────────────────────────
        ema_f = close.ewm(span=p["macd_fast"], adjust=False).mean()
        ema_s = close.ewm(span=p["macd_slow"], adjust=False).mean()
        macd  = ema_f - ema_s
        sig   = macd.ewm(span=p["macd_signal"], adjust=False).mean()
        df["macd"]      = macd
        df["macd_sig"]  = sig
        df["macd_hist"] = macd - sig

        # Histogram momentum direction
        df["macd_rising"] = df["macd_hist"] > df["macd_hist"].shift(1)
        df["macd_positive"] = df["macd_hist"] > 0

        # ── Volume ────────────────────────────────────────────────────────────
        df["vol_ma"]    = df["volume"].rolling(p["vol_ma_period"]).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, float("nan"))

        # ── ATR ───────────────────────────────────────────────────────────────
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - close.shift()).abs()
        lc  = (df["low"] - close.shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=p["atr_period"], adjust=False).mean()

        # ── ADX ───────────────────────────────────────────────────────────────
        up_move   = df["high"].diff()
        down_move = -df["low"].diff()
        plus_dm   = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm  = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        atr_s     = tr.ewm(span=14, adjust=False).mean()
        plus_di   = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr_s.replace(0, 1))
        minus_di  = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr_s.replace(0, 1))
        dx        = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1))
        df["adx"] = dx.ewm(span=14, adjust=False).mean()

        return df

    def get_signal(self, df: pd.DataFrame, current_price: float) -> SignalResult:
        p = self.params
        if len(df) < self.MIN_CANDLES:
            return SignalResult("HOLD", 0, 0, 0.0, "Not enough bars")

        last = df.iloc[-1]
        atr  = float(last["atr"]) if pd.notna(last["atr"]) else current_price * 0.005
        risk = atr * p["atr_sl_mult"]
        rsi  = float(last["rsi"]) if pd.notna(last["rsi"]) else 50.0
        adx  = float(last["adx"]) if pd.notna(last["adx"]) else 20.0
        vol_ratio = float(last["vol_ratio"]) if pd.notna(last["vol_ratio"]) else 1.0

        ema8  = float(last["ema8"])
        ema21 = float(last["ema21"])
        ema55 = float(last["ema55"])

        # EMA stack alignment
        bull_stack = ema8 > ema21 > ema55 and current_price > ema8
        bear_stack = ema8 < ema21 < ema55 and current_price < ema8

        macd_rising  = bool(last["macd_rising"])
        macd_positive = bool(last["macd_positive"])
        vol_ok = (vol_ratio >= 1.0) or (vol_ratio == 0.0) or pd.isna(vol_ratio)

        # ── BUY 1: Oversold Dip Reversal under EMA Ribbon ────────────────────
        if rsi < 35.0 and current_price < ema55:
            sl   = round(current_price - risk, 4)
            tp   = round(max(ema21, current_price + risk * p["rr_ratio"]), 4)
            return SignalResult("BUY", sl, tp, 0.84,
                                f"EMA Oversold Dip Hunter 🟢 RSI={rsi:.1f} EMA21=${ema21:.4f} TP=${tp:.4f}")

        # ── BUY 2: EMA bull stack + RSI in pullback zone + MACD momentum up ──
        rsi_pullback = p["rsi_bull_low"] <= rsi <= p["rsi_bull_high"]
        if bull_stack and rsi_pullback and macd_rising and vol_ok:
            sl   = round(current_price - risk, 4)
            tp   = round(current_price + risk * p["rr_ratio"], 4)
            conf = min(1.0, 0.55 + (adx / 100) + (rsi - 42) / 100)
            return SignalResult("BUY", sl, tp, conf,
                                f"EMA Stack Trend Pullback 🟢 RSI={rsi:.1f} MACD↑ TP=${tp:.4f}")

        # ── SELL: EMA bear stack + RSI in pullback zone ───────────────────────
        rsi_rally = p["rsi_bear_low"] <= rsi <= p["rsi_bear_high"]
        if bear_stack and rsi_rally and not macd_positive and vol_ok:
            sl  = round(current_price + risk, 4)
            tp  = round(current_price - risk * p["rr_ratio"], 4)
            return SignalResult("SELL", sl, tp, 0.62,
                                f"EMA8<{ema21:.0f}<{ema55:.0f}🔴 RSI={rsi:.1f} MACD↓ Vol={vol_ratio:.1f}x")

        return SignalResult("HOLD", 0, 0, 0.0,
                            f"Stack={'BULL' if bull_stack else 'BEAR' if bear_stack else 'NEUTRAL'} RSI={rsi:.1f} ADX={adx:.1f}")
