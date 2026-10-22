"""
Strategy: RSI + Bollinger Band Squeeze (Mean Reversion + Momentum Breakout)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Combines two of the most popular crypto bot signals used across GitHub/Reddit:

1. BB Squeeze Entry (Larry Connors style):
   - Bollinger Band Width contracts below 20-period low (energy coiling).
   - When price breaks the upper BB → momentum explosion UP.
   - When price breaks the lower BB → momentum explosion DOWN.

2. RSI Confirmation:
   - RSI(14) > 55 on breakout confirms bullish momentum.
   - RSI(14) < 45 on breakdown confirms bearish momentum.

3. Volume Surge Gating:
   - Volume must be ≥ 1.3x 20-bar MA to validate real breakout vs. fake.

Best for: DOGE/USD, AVAX/USD, XRP/USD (high-volatility mean-reverting coins with
          regular squeeze-and-explode price action patterns)

References:
- Freqtrade community: BBandsStrategy, DogeSqueezeBot
- GitHub: quickfingers123/freqtrade-strategies, NachE/freqtrade-strategies
"""
import pandas as pd
from .base import BaseStrategy, SignalResult


class RsiBbSqueeze(BaseStrategy):
    NAME = "RsiBbSqueeze"
    TIMEFRAME_MINUTES = 5
    MIN_CANDLES = 50

    def __init__(self, symbol: str, params: dict = None):
        defaults = {
            "bb_period": 20,
            "bb_std": 2.0,
            "squeeze_lookback": 20,
            "rsi_period": 14,
            "rsi_bull": 52,       # Relaxed from 55 to capture more signals
            "rsi_bear": 48,
            "vol_ma_period": 20,
            "vol_threshold": 1.2,  # Relaxed from 1.3x
            "atr_period": 14,
            "atr_sl_mult": 1.5,
            "rr_ratio": 2.0,
        }
        super().__init__(symbol, {**defaults, **(params or {})})

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        df = df.copy()
        close = df["close"]

        # ── Bollinger Bands ──────────────────────────────────────────────────
        sma = close.rolling(p["bb_period"]).mean()
        std = close.rolling(p["bb_period"]).std()
        df["bb_mid"] = sma
        df["bb_upper"] = sma + p["bb_std"] * std
        df["bb_lower"] = sma - p["bb_std"] * std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma

        # Squeeze: current width < rolling min of past N candles
        df["bb_squeeze"] = df["bb_width"] < df["bb_width"].rolling(p["squeeze_lookback"]).min().shift(1)
        # Post-squeeze expansion: width is now growing
        df["bb_expanding"] = df["bb_width"] > df["bb_width"].shift(1)

        # ── RSI ──────────────────────────────────────────────────────────────
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        ag = gain.ewm(com=p["rsi_period"] - 1, min_periods=p["rsi_period"]).mean()
        al = loss.ewm(com=p["rsi_period"] - 1, min_periods=p["rsi_period"]).mean()
        rs = ag / al.replace(0, float("nan"))
        df["rsi"] = 100 - (100 / (1 + rs))

        # ── Volume Surge ─────────────────────────────────────────────────────
        df["vol_ma"] = df["volume"].rolling(p["vol_ma_period"]).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, float("nan"))

        # ── ATR ───────────────────────────────────────────────────────────────
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - close.shift()).abs()
        lc  = (df["low"] - close.shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=p["atr_period"], adjust=False).mean()

        # ── EMA Trend Filter ──────────────────────────────────────────────────
        df["ema21"] = close.ewm(span=21, adjust=False).mean()
        df["ema55"] = close.ewm(span=55, adjust=False).mean()

        return df

    def get_signal(self, df: pd.DataFrame, current_price: float) -> SignalResult:
        p = self.params
        if len(df) < self.MIN_CANDLES:
            return SignalResult("HOLD", 0, 0, 0.0, "Not enough bars")

        last = df.iloc[-1]
        prev = df.iloc[-2]

        atr  = float(last["atr"]) if pd.notna(last["atr"]) else current_price * 0.008
        risk = atr * p["atr_sl_mult"]
        rsi  = float(last["rsi"]) if pd.notna(last["rsi"]) else 50.0
        vol_ratio = float(last["vol_ratio"]) if pd.notna(last["vol_ratio"]) else 1.0

        bb_upper = float(last["bb_upper"])
        bb_lower = float(last["bb_lower"])
        bb_mid   = float(last["bb_mid"])
        ema21    = float(last["ema21"])
        ema55    = float(last["ema55"])
        expanding = bool(last["bb_expanding"])

        # Previous candle was in or just exiting a squeeze
        was_squeezing = bool(prev.get("bb_squeeze", False))
        vol_ok = vol_ratio >= p["vol_threshold"]

        # ── BUY: Price breaks above upper BB during/after squeeze ────────────
        # Also allow: price > BB_mid + upward expanding, RSI bullish, trend up
        upper_break = current_price > bb_upper and (was_squeezing or expanding)
        trend_up = ema21 > ema55  # Basic trend alignment
        rsi_bull = rsi > p["rsi_bull"]

        # Relaxed buy: price near upper BB with bullish RSI (no strict squeeze needed)
        near_upper = current_price > (bb_mid + 0.4 * (bb_upper - bb_mid)) and rsi > 58 and trend_up

        if (upper_break or near_upper) and rsi_bull and vol_ok:
            conf = min(1.0, 0.6 + (rsi - 50) / 100 + (vol_ratio - 1.0) * 0.1)
            sl = round(current_price - risk, 4)
            tp = round(current_price + risk * p["rr_ratio"], 4)
            mode = "BB Squeeze Break↑" if upper_break else "BB Upper Momentum↑"
            return SignalResult("BUY", sl, tp, conf,
                                f"{mode} RSI={rsi:.1f} Vol={vol_ratio:.1f}x ATR={atr:.4f}")

        # ── SELL / EXIT: Price breaks below lower BB ─────────────────────────
        lower_break = current_price < bb_lower and (was_squeezing or expanding)
        rsi_bear = rsi < p["rsi_bear"]

        near_lower = current_price < (bb_mid - 0.4 * (bb_mid - bb_lower)) and rsi < 42

        if (lower_break or near_lower) and rsi_bear and vol_ok:
            sl = round(current_price + risk, 4)
            tp = round(current_price - risk * p["rr_ratio"], 4)
            mode = "BB Squeeze Break↓" if lower_break else "BB Lower Momentum↓"
            return SignalResult("SELL", sl, tp, 0.65,
                                f"{mode} RSI={rsi:.1f} Vol={vol_ratio:.1f}x")

        return SignalResult("HOLD", 0, 0, 0.0,
                            f"BB_W={last['bb_width']:.4f} RSI={rsi:.1f} Vol={vol_ratio:.1f}x Squeeze={'Y' if last.get('bb_squeeze') else 'N'}")
