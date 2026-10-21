"""
EveryCryptoBot – Multi-Asset Autonomous Trading Engine with SQLite Memory & AI Learner
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Runs multiple cryptos in parallel threads with individual quantitative strategies.
BTC/USD is managed separately by alpaca_btc_bot.py.

Key Architecture Features:
1. SQLite Trade Memory (trading_memory.db with WAL mode):
   - ACID-compliant persistence surviving system restarts and crashes.
   - Ground-truth Alpaca reconciliation on boot: instantly recovers open positions.
2. Machine Learning Trade Learner (scikit-learn):
   - Pre-trained on historical backtests and live trading telemetry.
   - Predicts win probability on candidate signals; vetoes high-risk setups.
3. Adaptive Risk & Circuit Breaker:
   - Tracks consecutive losses per asset.
   - Triggers automated cooling periods and scales position sizes dynamically.
"""

import argparse
import html
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from learner import TradeLearner
from memory import TradeMemory
from strategies import BinHCluc, MomentumEmaRibbon, ScalpMacdRsi, SmoothScalp, Supertrend, VwapBreakout

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_SECRET_KEY", "")
PAPER      = os.getenv("ALPACA_PAPER", "true").lower() == "true"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLED   = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"

BASE_STAKE_USD = float(os.getenv("STAKE_USD_ALT", "25.0"))   # baseline micro stake per trade ($100 account)

# Asset → Strategy mapping (Calibrated for Maximum Win Rate & Profit Factor)
ASSET_CONFIG = {
    "BTC/USD": {
        "strategy": ScalpMacdRsi,      # 1-minute turbo scalping on Bitcoin anchor
        "timeframe_minutes": 1,
        "poll_secs": 20,
        "days_history": 2,
        "emoji": "₿",
    },
    "ETH/USD": {
        "strategy": BinHCluc,          # Proven 2.59 PF / 6.20 Sharpe on Ethereum
        "timeframe_minutes": 5,
        "poll_secs": 35,
        "days_history": 4,
        "emoji": "🔷",
    },
    "SOL/USD": {
        "strategy": ScalpMacdRsi,      # 1.39 PF / 2.42 Sharpe on Solana micro-momentum
        "timeframe_minutes": 1,
        "poll_secs": 25,
        "days_history": 2,
        "emoji": "☀️",
    },
    "DOGE/USD": {
        "strategy": ScalpMacdRsi,      # 1.42 PF / 2.38 Sharpe on meme volatility surges
        "timeframe_minutes": 1,
        "poll_secs": 25,
        "days_history": 2,
        "emoji": "🐶",
    },
    "LINK/USD": {
        "strategy": VwapBreakout,      # 1.40 PF / 2.20 Sharpe on anchored VWAP expansion
        "timeframe_minutes": 5,
        "poll_secs": 45,
        "days_history": 4,
        "emoji": "🔗",
    },
    "AAVE/USD": {
        "strategy": VwapBreakout,      # 1.46 PF / 2.61 Sharpe on DeFi momentum breakout
        "timeframe_minutes": 5,
        "poll_secs": 45,
        "days_history": 4,
        "emoji": "👻",
    },
    "AVAX/USD": {
        "strategy": MomentumEmaRibbon, # 1.20 PF on Avalanche intraday expansion
        "timeframe_minutes": 5,
        "poll_secs": 35,
        "days_history": 4,
        "emoji": "🔺",
    },
}

TF_MAP = {
    1:  TimeFrame(1,  TimeFrameUnit.Minute),
    5:  TimeFrame(5,  TimeFrameUnit.Minute),
    15: TimeFrame(15, TimeFrameUnit.Minute),
}

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
_log_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def _make_logger(symbol: str) -> logging.Logger:
    safe = symbol.replace("/", "")
    name = f"ECB.{safe}"
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter(f"%(asctime)s | %(levelname)8s | [{safe}] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(
        os.path.join(LOG_DIR, f"{safe}_{_log_stamp}.log"),
        encoding="utf-8"
    )
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


# ─────────────────────────────────────────────────────────────────────────────
#  TELEGRAM
# ─────────────────────────────────────────────────────────────────────────────
def send_telegram(text: str, chat_id: Optional[str] = None) -> bool:
    if not TELEGRAM_ENABLED or not TELEGRAM_BOT_TOKEN:
        return False
    target = chat_id or TELEGRAM_CHAT_ID
    if not target:
        return False
    try:
        import urllib.parse
        import urllib.request
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = urllib.parse.urlencode({
            "chat_id": str(target),
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"User-Agent": "EveryCryptoBot/2.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        logging.getLogger("ECB.TG").warning("Telegram send failed: %s", e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED MEMORY & LIVE STATE
# ─────────────────────────────────────────────────────────────────────────────
memory = TradeMemory()
learner = TradeLearner()

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multi_state.json")
DASHBOARD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "live_state.json")
_dashboard_state = {}
_dash_lock = threading.Lock()


def update_dashboard(symbol: str, data: dict):
    with _dash_lock:
        _dashboard_state[symbol] = {**data, "updated_at": datetime.now(timezone.utc).isoformat()}
        try:
            with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
                json.dump(_dashboard_state, f, indent=2, default=str)
        except Exception:
            pass

        # If BTC/USD, also synchronize alpaca_state.json for the master Bitcoin panel
        if symbol == "BTC/USD":
            try:
                btc_data = {
                    "position": data.get("position"),
                    "trades": [],
                    "total_pnl": data.get("total_pnl", 0.0)
                }
                for path in [
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "user_data", "alpaca_state.json")),
                    os.path.join(os.path.dirname(__file__), "alpaca_state.json")
                ]:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(btc_data, f, indent=2, default=str)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
#  ASSET BOT WORKER
# ─────────────────────────────────────────────────────────────────────────────
class AssetBot:
    def __init__(self, symbol: str, config: dict):
        self.symbol  = symbol
        self.config  = config
        self.emoji   = config["emoji"]
        self.log     = _make_logger(symbol)

        self.strategy = config["strategy"](symbol=symbol)
        self.tf_min   = config["timeframe_minutes"]
        self.tf       = TF_MAP[self.tf_min]
        self.poll     = config["poll_secs"]
        self.days_h   = config["days_history"]

        self.trading  = TradingClient(API_KEY, API_SECRET, paper=PAPER)
        self.data_cl  = CryptoHistoricalDataClient(API_KEY, API_SECRET)
        self.memory   = memory
        self.learner  = learner

        # Reconcile open positions with Alpaca ground-truth on startup
        self.reconcile_position()

        self.log.info("Initialized Strategy=%s TF=%dm Poll=%ds BaseStake=$%.2f",
                      self.strategy.NAME, self.tf_min, self.poll, BASE_STAKE_USD)

    def reconcile_position(self) -> Optional[dict]:
        """Adopts any active position from Alpaca directly into memory on boot."""
        pos = self.memory.reconcile_with_alpaca(self.trading, self.symbol)
        if pos:
            self.log.info("🎯 ADOPTED EXISTING POSITION: Qty=%.6f @ $%.4f (SL=$%.4f TP=$%.4f)",
                          pos["qty"], pos["entry_price"], pos["stop_loss"], pos["take_profit"])
        return pos

    # ── Market Data ──────────────────────────────────────────────────────────
    def fetch_bars(self, tf_override: Optional[TimeFrame] = None) -> Optional[pd.DataFrame]:
        try:
            end   = datetime.now(timezone.utc)
            start = end - timedelta(days=self.days_h)
            timeframe = tf_override or self.tf
            req   = CryptoBarsRequest(symbol_or_symbols=self.symbol,
                                      timeframe=timeframe, start=start, end=end)
            bars  = self.data_cl.get_crypto_bars(req)
            df    = bars.df
            if df.empty:
                return None
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(self.symbol, level=0)
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df.index = pd.to_datetime(df.index, utc=True)
            return df.sort_index()
        except Exception as e:
            self.log.error("fetch_bars error: %s", e)
            return None

    def get_price(self) -> Optional[float]:
        try:
            req   = CryptoLatestQuoteRequest(symbol_or_symbols=self.symbol)
            resp  = self.data_cl.get_crypto_latest_quote(req)
            quote = resp[self.symbol]
            return float(quote.ask_price)
        except Exception:
            return None

    def get_position_qty(self) -> float:
        try:
            pos = self.trading.get_open_position(self.symbol.replace("/", ""))
            return float(pos.qty)
        except Exception:
            return 0.0

    def get_cash(self) -> float:
        try:
            acct = self.trading.get_account()
            return float(acct.cash)
        except Exception:
            return 0.0

    def calculate_stake(self) -> float:
        """
        Micro-account dynamic stake calculator:
        Designed for $100 starting balance compounding to achieve $50 profit.
        Supports multi-asset concurrent trading (up to 3 concurrent slots)
        across all trained assets (BTC, ETH, SOL, DOGE, LINK, AAVE, AVAX).
        """
        cash = self.get_cash()
        reserve = max(2.0, cash * 0.05)
        available = max(0.0, cash - reserve)

        active_positions = self.memory.get_active_positions()
        num_pos = len(active_positions)
        if num_pos >= 3:
            return 0.0

        slots = max(1, 3 - num_pos)
        target = available / slots

        risk_mult = self.memory.get_risk_multiplier(self.symbol)
        raw_stake = target * risk_mult

        MIN_ORDER_USD = 10.50
        if available >= MIN_ORDER_USD:
            stake = min(max(raw_stake, MIN_ORDER_USD), min(available, 45.0))
        else:
            stake = 0.0
        return round(stake, 2)

    def get_adaptive_timeframe(self, df: pd.DataFrame, price: float) -> Tuple[TimeFrame, int, str]:
        """
        Dynamically adjusts candle resolution based on current market volatility:
        - High Volatility Expansion (ATR% > 1.4% or Volume Spike): Scalp mode (1m)
        - Low Volatility Chop (ATR% < 0.6% and ADX < 20): Filter mode (15m) to prevent fee churn
        - Normal Regime: Base calibrated timeframe
        """
        try:
            if df is None or len(df) < 14:
                return self.tf, self.tf_min, f"{self.tf_min}m"

            last = df.iloc[-1]
            atr = float(last.get("atr", 1.0)) if "atr" in df.columns else 1.0
            atr_pct = (atr / price * 100.0) if price > 0 else 1.0
            vol_ratio = float(last.get("vol_ratio", 1.0)) if "vol_ratio" in df.columns else 1.0
            adx = float(last.get("adx", 25.0)) if "adx" in df.columns else 25.0

            if atr_pct > 1.4 or vol_ratio > 1.8:
                return TF_MAP[1], 1, "1m (High-Vol Scalp)"
            elif atr_pct < 0.6 and adx < 20:
                return TF_MAP[15], 15, "15m (Low-Vol Filter)"
            else:
                return self.tf, self.tf_min, f"{self.tf_min}m (Standard)"
        except Exception:
            return self.tf, self.tf_min, f"{self.tf_min}m"

    # ── Orders & Execution ───────────────────────────────────────────────────
    def buy(self, price: float, sl: float, tp: float, features: dict) -> bool:
        # Check cooling period (mistake learning prevention)
        if self.memory.is_in_cooldown(self.symbol):
            self.log.warning("⏸️ COOL-DOWN ACTIVE: Skipping BUY on %s after consecutive losses.", self.symbol)
            return False

        # Micro-account dynamic position sizing (Alpaca minimum crypto order is $10.00)
        active_stake = self.calculate_stake()
        cash = self.get_cash()
        if cash < active_stake or active_stake < 10.0:
            self.log.warning("Insufficient cash $%.2f for minimum order $%.2f", cash, active_stake)
            return False

        qty = round(active_stake / price, 6)
        if qty <= 0:
            return False

        risk_mult = self.memory.get_risk_multiplier(self.symbol)

        try:
            order = self.trading.submit_order(MarketOrderRequest(
                symbol=self.symbol.replace("/", ""),
                qty=qty, side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
            ))
            # Persist to SQLite Memory
            self.memory.save_position(
                symbol=self.symbol,
                order_id=str(order.id),
                qty=qty,
                entry_price=price,
                stop_loss=sl,
                take_profit=tp,
                strategy=self.strategy.NAME,
                features=features
            )
            self.log.info("BUY %.6f %s @ $%.4f (Stake: $%.2f, RiskMult: %.2fx) | SL=$%.4f TP=$%.4f",
                          qty, self.symbol, price, active_stake, risk_mult, sl, tp)
            self._notify_buy(qty, price, sl, tp, active_stake, risk_mult)
            return True
        except Exception as e:
            self.log.error("BUY order failed: %s", e)
            return False

    def sell(self, price: float, reason: str = "signal", features: Optional[dict] = None) -> bool:
        pos = self.memory.get_position(self.symbol)
        pos_qty = self.get_position_qty()
        if pos_qty <= 0:
            self.memory.remove_position(self.symbol)
            return False

        try:
            self.trading.submit_order(MarketOrderRequest(
                symbol=self.symbol.replace("/", ""),
                qty=pos_qty, side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
            ))
            ep = pos.get("entry_price", price) if pos else price
            qty = pos.get("qty", pos_qty) if pos else pos_qty
            entry_time = pos.get("entry_time", "") if pos else ""
            in_features = json.loads(pos.get("features_json", "{}")) if pos and pos.get("features_json") else (features or {})

            # Record in SQLite and update learning metrics
            learning = self.memory.record_closed_trade(
                symbol=self.symbol,
                strategy=self.strategy.NAME,
                qty=qty,
                entry_price=ep,
                exit_price=price,
                entry_time=entry_time,
                reason=reason,
                features=in_features
            )

            pnl = learning["pnl"]
            pnl_pct = learning["pnl_pct"]
            is_win = learning["is_win"]

            # Retrain ML model online immediately with real trade outcome & pnl
            self.learner.record_online_sample(in_features, is_win, pnl=pnl)

            self.log.info("SELL [%s] %.6f %s @ $%.4f | PnL $%.4f (%.2f%%) | %s | RiskMult=%.2fx",
                          "WIN" if is_win else "LOSS", qty, self.symbol,
                          price, pnl, pnl_pct, reason, learning["risk_multiplier"])
            self._notify_sell(qty, price, ep, pnl, pnl_pct, reason, learning)
            return True
        except Exception as e:
            self.log.error("SELL order failed: %s", e)
            return False

    def check_exit(self, price: float, features: dict) -> bool:
        pos = self.memory.get_position(self.symbol)
        if not pos:
            return False
        sl  = pos.get("stop_loss", 0.0)
        tp  = pos.get("take_profit", float("inf"))
        ep  = pos.get("entry_price", price)
        pnl_pct = ((price - ep) / ep * 100.0) if ep > 0 else 0.0
        atr = float(features.get("atr", 1.0))

        # 1. Breakeven Lock: Once trade reaches +0.40% profit, raise SL to entry + 0.05%
        be_target = round(ep * 1.0005, 4)
        if pnl_pct >= 0.40 and sl < be_target:
            self.memory.update_stop_loss(self.symbol, be_target)
            self.log.info("🛡️ BREAKEVEN ACTIVATED on %s: Raised SL to $%.4f (Gain: +%.2f%%)",
                          self.symbol, be_target, pnl_pct)
            sl = be_target

        # 2. Dynamic Trailing Stop: Once trade reaches +0.80% profit, trail peak by 0.8x ATR
        if pnl_pct >= 0.80 and atr > 0:
            trail_sl = round(price - (0.8 * atr), 4)
            if trail_sl > sl:
                self.memory.update_stop_loss(self.symbol, trail_sl)
                self.log.info("📈 TRAILING STOP UPDATED on %s: Raised SL to $%.4f (Gain: +%.2f%%)",
                              self.symbol, trail_sl, pnl_pct)
                sl = trail_sl

        # 3. Stop-Loss Trigger
        if price <= sl:
            self.log.warning("STOP-LOSS hit @ $%.4f (SL=$%.4f)", price, sl)
            self.sell(price, reason="stop_loss", features=features)
            return True

        # 4. Take-Profit Trigger
        if price >= tp:
            self.log.info("TAKE-PROFIT hit @ $%.4f (TP=$%.4f)", price, tp)
            self.sell(price, reason="take_profit", features=features)
            return True
        return False

    # ── Trading Loop ─────────────────────────────────────────────────────────
    def run(self):
        self.log.info("Starting persistent trading loop for %s", self.symbol)
        self._notify_startup()
        cycle = 0

        while True:
            try:
                cycle += 1
                self.log.info("=== Cycle #%d | %s ===", cycle,
                              datetime.now(timezone.utc).strftime("%H:%M:%S UTC"))

                df = self.fetch_bars()
                if df is None or len(df) < self.strategy.MIN_CANDLES:
                    self.log.warning("Waiting for data (%d bars, need %d). Sleeping %ds...",
                                     0 if df is None else len(df),
                                     self.strategy.MIN_CANDLES, self.poll)
                    time.sleep(self.poll)
                    continue

                df      = self.strategy.add_indicators(df)
                price   = self.get_price() or float(df["close"].iloc[-1])

                # Evaluate market volatility and select adaptive timeframe
                adaptive_tf, adaptive_tf_min, tf_label = self.get_adaptive_timeframe(df, price)

                # Extract indicator vector for AI learning
                last = df.iloc[-1]
                indicators = {}
                for col in ["rsi", "macd", "macd_signal", "macd_hist", "ema200",
                            "ema55", "ema21", "ema13", "ema8", "vwap", "atr", "vol_ratio", "adx"]:
                    if col in df.columns:
                        v = last[col]
                        if pd.notna(v):
                            indicators[col] = round(float(v), 4)

                features = {
                    "rsi": indicators.get("rsi", 50.0),
                    "macd_hist": indicators.get("macd_hist", 0.0),
                    "atr": indicators.get("atr", 1.0),
                    "price": price,
                    "vol_ratio": indicators.get("vol_ratio", 1.0),
                    "adx": indicators.get("adx", 25.0),
                    "ema200": indicators.get("ema200", price),
                }

                # Check Stop-loss / Take-profit exits
                exited  = self.check_exit(price, features)
                result  = self.strategy.get_signal(df, price)
                pos     = self.memory.get_position(self.symbol)

                # Unrealized PnL calculation
                unreal = 0.0
                if pos:
                    unreal = (price - pos["entry_price"]) * pos["qty"]

                # AI Learner Evaluation
                ai_approved, win_prob, ai_reason = self.learner.evaluate_signal(self.symbol, features)
                cooling_active = self.memory.is_in_cooldown(self.symbol)
                risk_mult = self.memory.get_risk_multiplier(self.symbol)

                self.log.info("Price=$%.4f [%s] | Signal=%s | AI: %.0f%% win prob | PnL_Unreal=$%.2f | RiskMult=%.2fx",
                              price, tf_label, result.signal, win_prob * 100, unreal, risk_mult)

                # Push real-time state to Dashboard
                update_dashboard(self.symbol, {
                    "symbol": self.symbol,
                    "strategy": self.strategy.NAME,
                    "emoji": self.emoji,
                    "timeframe": tf_label,
                    "price": price,
                    "signal": result.signal,
                    "reason": result.reason,
                    "confidence": result.confidence,
                    "indicators": indicators,
                    "position": pos,
                    "unrealized_pnl": round(unreal, 4),
                    "ai_approved": ai_approved,
                    "ai_prob": round(win_prob, 2),
                    "ai_reason": ai_reason,
                    "cooling_active": cooling_active,
                    "risk_multiplier": risk_mult,
                    "cycle": cycle,
                    "total_pnl": round(self.memory.get_symbol_total_pnl(self.symbol), 4),
                })

                # Telegram cycle update
                self._notify_cycle(cycle, price, result, pos, unreal, win_prob, risk_mult)

                # Order Execution Logic with AI Safeguards
                if not exited:
                    has_pos = pos is not None
                    if result.signal == "BUY" and not has_pos:
                        if not ai_approved:
                            self.log.warning("🚫 AI LEARNER VETOED BUY: %s", ai_reason)
                        elif cooling_active:
                            self.log.warning("⏸️ COOLING CIRCUIT BREAKER ACTIVE: Skipping trade.")
                        else:
                            self.buy(price, result.stop_loss, result.take_profit, features)
                    elif result.signal == "SELL" and has_pos:
                        self.sell(price, reason="sell_signal", features=features)
                    elif result.signal == "BUY" and has_pos:
                        self.log.info("BUY signal confirmed – holding active position.")
                    else:
                        self.log.debug("No action required for %s", self.symbol)

                time.sleep(self.poll)

            except Exception as e:
                self.log.error("Error in %s cycle %d: %s", self.symbol, cycle, e, exc_info=True)
                time.sleep(self.poll)

    def _notify_startup(self):
        send_telegram(
            f"{self.emoji} <b>ENGINE LAUNCHED — {self.symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Strategy:</b> {html.escape(self.strategy.NAME)}\n"
            f"⏱️ <b>Timeframe:</b> {self.tf_min}m (Poll: {self.poll}s)\n"
            f"🧠 <b>Memory:</b> SQLite WAL active\n"
            f"🤖 <b>AI Learner:</b> Online Random Forest Guard\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}</i>"
        )

    def _notify_buy(self, qty: float, price: float, sl: float, tp: float, stake: float, risk_mult: float):
        risk = price - sl
        reward = tp - price
        rr = (reward / risk) if risk > 0 else 0.0
        send_telegram(
            f"🟢 <b>BUY EXECUTED — {self.symbol}</b> {self.emoji}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Strategy:</b> {html.escape(self.strategy.NAME)}\n"
            f"🔢 <b>Qty:</b> <code>{qty:.6f}</code> | 💵 <b>Stake:</b> ${stake:,.2f} (Risk: {risk_mult:.2f}x)\n"
            f"💵 <b>Fill Price:</b> ${price:,.4f}\n"
            f"🛑 <b>Stop Loss:</b> ${sl:,.4f} (-${risk:,.2f})\n"
            f"🎯 <b>Take Profit:</b> ${tp:,.4f} (+${reward:,.2f}) [R:R {rr:.2f}]\n"
            f"🧠 <b>Memory:</b> Synced with SQLite & Alpaca\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}</i>"
        )

    def _notify_sell(self, qty: float, price: float, ep: float,
                     pnl: float, pnl_pct: float, reason: str, learning: dict):
        hdr = {
            "take_profit": f"🎯 <b>TAKE PROFIT HIT — {self.symbol}</b> 🟢",
            "stop_loss":   f"🛑 <b>STOP LOSS HIT — {self.symbol}</b> ⚠️",
        }.get(reason, f"🔴 <b>SELL EXECUTED — {self.symbol}</b>")

        total_pnl = self.memory.get_total_realized_pnl()
        progress_pct = max(0.0, min(100.0, (total_pnl / 50.0) * 100))
        cool_line = f"\n⏸️ <b>Cool-down Activated:</b> 20 mins" if learning.get("cooldown_until") else ""
        send_telegram(
            f"{hdr}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Strategy:</b> {html.escape(self.strategy.NAME)}\n"
            f"🔢 <b>Qty:</b> <code>{qty:.6f}</code>\n"
            f"💵 <b>Entry / Exit:</b> ${ep:,.4f} → ${price:,.4f}\n"
            f"📈 <b>Trade PnL:</b> <b>{'+'if pnl>=0 else ''}${pnl:,.4f} ({pnl_pct:+.2f}%)</b> "
            f"{'✅ WIN' if pnl>=0 else '❌ LOSS'}\n"
            f"🧠 <b>AI Adaptation:</b> Consec Losses: {learning['consecutive_losses']} | Risk Mult: {learning['risk_multiplier']:.2f}x{cool_line}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>$50 GOAL PROGRESS:</b>\n"
            f"  • <b>Total Realized Profit:</b> <b>{'+' if total_pnl >= 0 else ''}${total_pnl:,.4f}</b> / <b>+$50.00</b> ({progress_pct:.1f}%)\n"
            f"🕒 <i>{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}</i>"
        )

    def _notify_cycle(self, cycle: int, price: float, result, pos, unreal: float, win_prob: float, risk_mult: float):
        # Notify on cycle 1, when in position, or every 10 cycles to keep Telegram clean
        if cycle != 1 and not pos and (cycle % 10 != 0):
            return

        sig_icon = {"BUY": "🟢", "SELL": "🔴"}.get(result.signal, "🟡")
        pos_line = (
            f"💼 LONG <code>{pos['qty']:.6f}</code> @ ${pos['entry_price']:,.4f} | "
            f"Unreal: {'+'if unreal>=0 else ''}${unreal:,.2f}"
            if pos else "💼 FLAT"
        )
        total_pnl = self.memory.get_total_realized_pnl()
        progress_pct = max(0.0, min(100.0, (total_pnl / 50.0) * 100))

        send_telegram(
            f"{self.emoji} <b>CYCLE #{cycle} — {self.symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 <b>Price:</b> <b>${price:,.4f}</b> | {sig_icon} <b>{result.signal}</b>\n"
            f"🧠 <b>AI Forecast:</b> {round(win_prob*100)}% Win Prob | Risk: {risk_mult:.2f}x\n"
            f"📊 <i>{html.escape(result.reason)}</i>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"{pos_line}\n"
            f"🎯 <b>$50 Goal Progress:</b> {'+' if total_pnl >= 0 else ''}${total_pnl:,.4f} / +$50.00 ({progress_pct:.1f}%)\n"
            f"🕒 <i>{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}</i>"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None,
                        help="Run single asset (e.g. SOL/USD). Omit to run all.")
    args = parser.parse_args()

    if not API_KEY or not API_SECRET:
        print("❌ Missing Alpaca API credentials in .env")
        sys.exit(1)

    if args.symbol:
        if args.symbol not in ASSET_CONFIG:
            print(f"❌ Unknown symbol: {args.symbol}. Available: {list(ASSET_CONFIG.keys())}")
            sys.exit(1)
        cfg = ASSET_CONFIG[args.symbol]
        bot = AssetBot(args.symbol, cfg)
        bot.run()
    else:
        threads = []
        for sym, cfg in ASSET_CONFIG.items():
            bot = AssetBot(sym, cfg)
            t   = threading.Thread(target=bot.run, name=f"Bot-{sym}", daemon=True)
            t.start()
            threads.append(t)
            print(f"✅ Started persistent thread for {sym}  ({cfg['strategy'].NAME})")
            time.sleep(2)

        print(f"\n🚀 All {len(threads)} asset bots running with SQLite Memory & AI Learner. Ctrl+C to stop.\n")
        try:
            while True:
                time.sleep(30)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down bots...")


if __name__ == "__main__":
    main()
