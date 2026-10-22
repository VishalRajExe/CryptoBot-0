"""
EveryCryptoBot - Multi-Asset Autonomous Trading Engine v3.0
===========================================================
11 crypto assets running in parallel, each with the best-in-class
strategy calibrated for its volatility profile.

Strategy assignment research:
- GitHub: freqtrade-strategies (NachE, quickfingers123, iterativv, paulcpk)
- Reddit r/algotrading top posts 2023-2024
- 22 backtest JSON files in backtests/ directory

Key changes vs v2:
- No profit-goal pressure: bots trade freely to maximise PnL
- 11 assets (added LTC, XRP, UNI, MATIC) for more signal diversity
- Best-in-class strategy per coin (not one-size-fits-all)
- Up to 5 concurrent positions (was 3)
- EmaRsiScalper fires on pullbacks (far more active than ScalpMacdRsi)
- RsiBbSqueeze captures squeeze-explode patterns on volatile coins
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, Any, List

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
from strategies import (BinHCluc, EmaRsiScalper, MomentumEmaRibbon, RsiBbSqueeze,
                        ScalpMacdRsi, SmoothScalp, Supertrend, VwapBreakout)

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_SECRET_KEY", "")
PAPER      = os.getenv("ALPACA_PAPER", "true").lower() == "true"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLED   = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"

BASE_STAKE_USD           = float(os.getenv("STAKE_USD_ALT", "15.0"))
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_POSITIONS", "5"))

# ---------------------------------------------------------------------------
# ASSET -> STRATEGY MAPPING  (best-in-class per coin)
#
# Rationale:
#  BTC/SOL/MATIC -> EmaRsiScalper  : trend-following, fires on RSI pullbacks
#  ETH/UNI      -> BinHCluc        : mean reversion BB dip-hunter (2.59 PF)
#  DOGE/AVAX/XRP-> RsiBbSqueeze    : meme/squeeze-explode patterns
#  LINK         -> VwapBreakout    : VWAP-anchored momentum
#  AAVE         -> MomentumEmaRibbon: DeFi full-ribbon trend
#  LTC          -> SmoothScalp     : BTC-correlated smooth entry
# ---------------------------------------------------------------------------
ASSET_CONFIG = {
    "BTC/USD":  {"strategy": EmaRsiScalper,     "timeframe_minutes": 3,  "poll_secs": 20, "days_history": 3, "emoji": "BTC"},
    "ETH/USD":  {"strategy": BinHCluc,          "timeframe_minutes": 5,  "poll_secs": 30, "days_history": 4, "emoji": "ETH"},
    "SOL/USD":  {"strategy": EmaRsiScalper,     "timeframe_minutes": 5,  "poll_secs": 30, "days_history": 3, "emoji": "SOL"},
    "DOGE/USD": {"strategy": RsiBbSqueeze,      "timeframe_minutes": 5,  "poll_secs": 30, "days_history": 3, "emoji": "DOGE"},
    "LINK/USD": {"strategy": VwapBreakout,      "timeframe_minutes": 5,  "poll_secs": 40, "days_history": 4, "emoji": "LINK"},
    "AAVE/USD": {"strategy": MomentumEmaRibbon, "timeframe_minutes": 5,  "poll_secs": 45, "days_history": 4, "emoji": "AAVE"},
    "AVAX/USD": {"strategy": RsiBbSqueeze,      "timeframe_minutes": 5,  "poll_secs": 35, "days_history": 3, "emoji": "AVAX"},
    "LTC/USD":  {"strategy": SmoothScalp,       "timeframe_minutes": 5,  "poll_secs": 40, "days_history": 3, "emoji": "LTC"},
    "XRP/USD":  {"strategy": RsiBbSqueeze,      "timeframe_minutes": 5,  "poll_secs": 30, "days_history": 3, "emoji": "XRP"},
    "UNI/USD":  {"strategy": BinHCluc,          "timeframe_minutes": 5,  "poll_secs": 45, "days_history": 4, "emoji": "UNI"},
    "MATIC/USD":{"strategy": EmaRsiScalper,     "timeframe_minutes": 5,  "poll_secs": 35, "days_history": 3, "emoji": "MATIC"},
}

TF_MAP = {
    1:  TimeFrame(1,  TimeFrameUnit.Minute),
    3:  TimeFrame(3,  TimeFrameUnit.Minute),
    5:  TimeFrame(5,  TimeFrameUnit.Minute),
    15: TimeFrame(15, TimeFrameUnit.Minute),
}

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOG_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
_log_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def _make_logger(symbol: str) -> logging.Logger:
    safe = symbol.replace("/", "")
    name = f"ECB.{safe}"
    log  = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter(f"%(asctime)s | %(levelname)8s | [{safe}] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(LOG_DIR, f"{safe}_{_log_stamp}.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------
def send_telegram_detailed(text: str, chat_id: Optional[str] = None) -> tuple[bool, dict]:
    enabled = os.getenv("TELEGRAM_ENABLED", "true").lower() in ("true", "1", "yes")
    token   = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip().strip('\'"') or "8766601647:AAGu0Q1MzS6fMTvW3OW7_0WZGiIxgYq748c"
    primary = (chat_id or os.getenv("TELEGRAM_CHAT_ID") or "").strip().strip('\'"') or "8842786135"

    details = {"enabled": enabled, "token_length": len(token),
               "primary_target": primary, "attempts": []}

    if not enabled:
        details["error"] = "TELEGRAM_ENABLED is false"
        return False, details

    VTOK = "8766601647:AAGu0Q1MzS6fMTvW3OW7_0WZGiIxgYq748c"
    VCHAT = "8842786135"
    tokens  = [token] + ([VTOK] if token != VTOK else [])
    targets = [primary] + ([VCHAT] if VCHAT not in [primary] else [])

    for tok in tokens:
        for tgt in targets:
            tc = str(tgt).strip().strip('\'"')
            info = {"token": tok[:6] + "...", "target": tc}
            try:
                import urllib.parse, urllib.request
                url = f"https://api.telegram.org/bot{tok}/sendMessage"
                data = urllib.parse.urlencode({
                    "chat_id": tc, "text": text,
                    "parse_mode": "HTML", "disable_web_page_preview": "true",
                }).encode("utf-8")
                req = urllib.request.Request(url, data=data,
                                             headers={"User-Agent": "EveryCryptoBot/3.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    info["status"] = resp.status
                    details["attempts"].append(info)
                    if resp.status == 200:
                        return True, details
            except urllib.error.HTTPError as e:
                info["error"] = f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
                details["attempts"].append(info)
            except Exception as e:
                info["error"] = str(e)
                details["attempts"].append(info)
    return False, details


def send_telegram(text: str, chat_id: Optional[str] = None) -> bool:
    success, _ = send_telegram_detailed(text, chat_id)
    return success


# ---------------------------------------------------------------------------
# SHARED MEMORY & LIVE STATE
# ---------------------------------------------------------------------------
memory  = TradeMemory()
learner = TradeLearner()

STATE_FILE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multi_state.json")
DASHBOARD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "live_state.json")
_dashboard_state: dict = {}
_dash_lock = threading.Lock()


def update_dashboard(symbol: str, data: dict) -> None:
    with _dash_lock:
        _dashboard_state[symbol] = {**data, "updated_at": datetime.now(timezone.utc).isoformat()}
        try:
            with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
                json.dump(_dashboard_state, f, indent=2, default=str)
        except Exception:
            pass
        if symbol == "BTC/USD":
            btc = {"position": data.get("position"), "trades": [], "total_pnl": data.get("total_pnl", 0.0)}
            for path in [
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "user_data", "alpaca_state.json")),
                os.path.join(os.path.dirname(__file__), "alpaca_state.json"),
            ]:
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(btc, f, indent=2, default=str)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# ASSET BOT WORKER
# ---------------------------------------------------------------------------
class AssetBot:
    def __init__(self, symbol: str, config: dict):
        self.symbol   = symbol
        self.config   = config
        self.emoji    = config["emoji"]
        self.log      = _make_logger(symbol)
        self.strategy = config["strategy"](symbol=symbol)
        self.tf_min   = config["timeframe_minutes"]
        self.tf       = TF_MAP[self.tf_min]
        self.poll     = config["poll_secs"]
        self.days_h   = config["days_history"]
        self.trading  = TradingClient(API_KEY, API_SECRET, paper=PAPER)
        self.data_cl  = CryptoHistoricalDataClient(API_KEY, API_SECRET)
        self.memory   = memory
        self.learner  = learner
        self.reconcile_position()
        self.log.info("Initialized Strategy=%s TF=%dm Poll=%ds MaxPos=%d",
                      self.strategy.NAME, self.tf_min, self.poll, MAX_CONCURRENT_POSITIONS)

    def reconcile_position(self) -> Optional[dict]:
        pos = self.memory.reconcile_with_alpaca(self.trading, self.symbol)
        if pos:
            self.log.info("ADOPTED POSITION: Qty=%.6f @ $%.4f", pos["qty"], pos["entry_price"])
        return pos

    # -- Market data ----------------------------------------------------------
    def fetch_bars(self, tf_override: Optional[TimeFrame] = None) -> Optional[pd.DataFrame]:
        try:
            end   = datetime.now(timezone.utc)
            start = end - timedelta(days=self.days_h)
            tf    = tf_override or self.tf
            req   = CryptoBarsRequest(symbol_or_symbols=self.symbol, timeframe=tf, start=start, end=end)
            df    = self.data_cl.get_crypto_bars(req).df
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
            resp  = self.data_cl.get_crypto_latest_quote(CryptoLatestQuoteRequest(symbol_or_symbols=self.symbol))
            return float(resp[self.symbol].ask_price)
        except Exception:
            return None

    def get_position_qty(self) -> float:
        try:
            return float(self.trading.get_open_position(self.symbol.replace("/", "")).qty)
        except Exception:
            return 0.0

    def get_cash(self) -> float:
        try:
            return float(self.trading.get_account().cash)
        except Exception:
            return 0.0

    # -- Position sizing ------------------------------------------------------
    def calculate_stake(self) -> float:
        """
        Dynamic stake per trade.
        - Up to MAX_CONCURRENT_POSITIONS slots across all assets.
        - Each slot gets an equal share of available cash scaled by risk multiplier.
        - Min $10.50 (Alpaca crypto minimum), Max $30 per trade.
        """
        cash      = self.get_cash()
        available = max(0.0, cash - max(2.0, cash * 0.04))
        num_pos   = len(self.memory.get_active_positions())
        if num_pos >= MAX_CONCURRENT_POSITIONS:
            return 0.0
        slots     = max(1, MAX_CONCURRENT_POSITIONS - num_pos)
        target    = available / slots
        risk_mult = self.memory.get_risk_multiplier(self.symbol)
        raw       = target * risk_mult
        if available >= 10.50:
            return round(min(max(raw, 10.50), min(available, 30.0)), 2)
        return 0.0

    # -- Adaptive timeframe ---------------------------------------------------
    def get_adaptive_timeframe(self, df: pd.DataFrame, price: float) -> Tuple[TimeFrame, int, str]:
        try:
            if df is None or len(df) < 14:
                return self.tf, self.tf_min, f"{self.tf_min}m"
            last      = df.iloc[-1]
            atr       = float(last.get("atr", 1.0)) if "atr" in df.columns else 1.0
            atr_pct   = (atr / price * 100.0) if price > 0 else 1.0
            vol_ratio = float(last.get("vol_ratio", 1.0)) if "vol_ratio" in df.columns else 1.0
            adx       = float(last.get("adx", 25.0)) if "adx" in df.columns else 25.0
            if atr_pct > 1.4 or vol_ratio > 1.8:
                ftf = min(self.tf_min, 3)
                return TF_MAP.get(ftf, self.tf), ftf, f"{ftf}m (High-Vol Scalp)"
            if atr_pct < 0.6 and adx < 20:
                return TF_MAP[15], 15, "15m (Low-Vol Filter)"
            return self.tf, self.tf_min, f"{self.tf_min}m (Standard)"
        except Exception:
            return self.tf, self.tf_min, f"{self.tf_min}m"

    # -- Orders ---------------------------------------------------------------
    def buy(self, price: float, sl: float, tp: float, features: dict) -> bool:
        if self.memory.is_in_cooldown(self.symbol):
            self.log.warning("COOL-DOWN: Skipping BUY on %s", self.symbol)
            return False
        stake = self.calculate_stake()
        cash  = self.get_cash()
        if cash < stake or stake < 10.0:
            self.log.warning("Insufficient cash $%.2f for order $%.2f", cash, stake)
            return False
        qty      = round(stake / price, 6)
        risk_mul = self.memory.get_risk_multiplier(self.symbol)
        try:
            order = self.trading.submit_order(MarketOrderRequest(
                symbol=self.symbol.replace("/", ""), qty=qty,
                side=OrderSide.BUY, time_in_force=TimeInForce.GTC))
            self.memory.save_position(symbol=self.symbol, order_id=str(order.id), qty=qty,
                entry_price=price, stop_loss=sl, take_profit=tp,
                strategy=self.strategy.NAME, features=features)
            self.log.info("BUY %.6f %s @ $%.4f  stake=$%.2f  SL=$%.4f  TP=$%.4f",
                          qty, self.symbol, price, stake, sl, tp)
            self._notify_buy(qty, price, sl, tp, stake, risk_mul)
            return True
        except Exception as e:
            self.log.error("BUY failed: %s", e)
            return False

    def sell(self, price: float, reason: str = "signal", features: Optional[dict] = None) -> bool:
        pos     = self.memory.get_position(self.symbol)
        pos_qty = self.get_position_qty()
        if pos_qty <= 0:
            self.memory.remove_position(self.symbol)
            return False
        try:
            self.trading.submit_order(MarketOrderRequest(
                symbol=self.symbol.replace("/", ""), qty=pos_qty,
                side=OrderSide.SELL, time_in_force=TimeInForce.GTC))
            ep         = pos.get("entry_price", price) if pos else price
            qty        = pos.get("qty", pos_qty) if pos else pos_qty
            entry_time = pos.get("entry_time", "") if pos else ""
            feats      = json.loads(pos.get("features_json", "{}")) if pos and pos.get("features_json") else (features or {})
            learning   = self.memory.record_closed_trade(
                symbol=self.symbol, strategy=self.strategy.NAME, qty=qty,
                entry_price=ep, exit_price=price, entry_time=entry_time,
                reason=reason, features=feats)
            pnl, pnl_pct, is_win = learning["pnl"], learning["pnl_pct"], learning["is_win"]
            self.learner.record_online_sample(feats, is_win, pnl=pnl)
            self.log.info("SELL [%s] %.6f %s @ $%.4f  PnL=$%.4f (%.2f%%)  %s",
                          "WIN" if is_win else "LOSS", qty, self.symbol, price, pnl, pnl_pct, reason)
            self._notify_sell(qty, price, ep, pnl, pnl_pct, reason, learning)
            return True
        except Exception as e:
            self.log.error("SELL failed: %s", e)
            return False

    def check_exit(self, price: float, features: dict) -> bool:
        pos = self.memory.get_position(self.symbol)
        if not pos:
            return False
        sl  = pos.get("stop_loss", 0.0)
        tp  = pos.get("take_profit", float("inf"))
        ep  = pos.get("entry_price", price)
        pct = ((price - ep) / ep * 100.0) if ep > 0 else 0.0
        atr = float(features.get("atr", 1.0))

        # Breakeven at +0.4%
        be = round(ep * 1.0005, 4)
        if pct >= 0.40 and sl < be:
            self.memory.update_stop_loss(self.symbol, be)
            self.log.info("BREAKEVEN @ $%.4f  gain=+%.2f%%", be, pct)
            sl = be
        # Trailing at +0.8%
        if pct >= 0.80 and atr > 0:
            trail = round(price - 0.8 * atr, 4)
            if trail > sl:
                self.memory.update_stop_loss(self.symbol, trail)
                sl = trail
        if price <= sl:
            self.log.warning("SL HIT @ $%.4f", price)
            self.sell(price, reason="stop_loss", features=features)
            return True
        if price >= tp:
            self.log.info("TP HIT @ $%.4f", price)
            self.sell(price, reason="take_profit", features=features)
            return True
        return False

    # -- Main loop ------------------------------------------------------------
    def run(self) -> None:
        self.log.info("Starting trading loop for %s", self.symbol)
        self._notify_startup()
        cycle = 0
        while True:
            try:
                cycle += 1
                self.log.info("=== Cycle #%d | %s ===", cycle,
                              datetime.now(timezone.utc).strftime("%H:%M:%S UTC"))
                df = self.fetch_bars()
                if df is None or len(df) < self.strategy.MIN_CANDLES:
                    self.log.warning("Need %d bars, have %d. Sleeping...",
                                     self.strategy.MIN_CANDLES, 0 if df is None else len(df))
                    time.sleep(self.poll)
                    continue

                df    = self.strategy.add_indicators(df)
                price = self.get_price() or float(df["close"].iloc[-1])
                _, _, tf_label = self.get_adaptive_timeframe(df, price)

                last = df.iloc[-1]
                indicators: dict = {}
                for col in ["rsi", "macd", "macd_signal", "macd_hist", "ema200",
                            "ema55", "ema21", "ema13", "ema8", "vwap", "atr", "vol_ratio", "adx"]:
                    if col in df.columns and pd.notna(last[col]):
                        indicators[col] = round(float(last[col]), 4)

                features = {
                    "rsi":       indicators.get("rsi",       50.0),
                    "macd_hist": indicators.get("macd_hist",  0.0),
                    "atr":       indicators.get("atr",        1.0),
                    "price":     price,
                    "vol_ratio": indicators.get("vol_ratio",  1.0),
                    "adx":       indicators.get("adx",       25.0),
                    "ema200":    indicators.get("ema200",    price),
                }

                exited  = self.check_exit(price, features)
                result  = self.strategy.get_signal(df, price)
                pos     = self.memory.get_position(self.symbol)
                unreal  = ((price - pos["entry_price"]) * pos["qty"]) if pos else 0.0

                ai_ok, win_prob, ai_reason = self.learner.evaluate_signal(self.symbol, features)
                cooling   = self.memory.is_in_cooldown(self.symbol)
                risk_mult = self.memory.get_risk_multiplier(self.symbol)

                self.log.info("$%.4f [%s] | %s | AI=%.0f%% | Unreal=$%.2f | Risk=%.2fx",
                              price, tf_label, result.signal, win_prob * 100, unreal, risk_mult)

                update_dashboard(self.symbol, {
                    "symbol": self.symbol, "strategy": self.strategy.NAME,
                    "emoji": self.emoji, "timeframe": tf_label, "price": price,
                    "signal": result.signal, "reason": result.reason,
                    "confidence": result.confidence, "indicators": indicators,
                    "position": pos, "unrealized_pnl": round(unreal, 4),
                    "ai_approved": ai_ok, "ai_prob": round(win_prob, 2),
                    "ai_reason": ai_reason, "cooling_active": cooling,
                    "risk_multiplier": risk_mult, "cycle": cycle,
                    "total_pnl": round(self.memory.get_symbol_total_pnl(self.symbol), 4),
                })

                self._notify_cycle(cycle, price, result, pos, unreal, win_prob, risk_mult)

                if not exited:
                    has_pos = pos is not None
                    if result.signal == "BUY" and not has_pos:
                        if not ai_ok:
                            self.log.warning("AI VETOED BUY: %s", ai_reason)
                        elif cooling:
                            self.log.warning("COOLING: Skipping.")
                        else:
                            self.buy(price, result.stop_loss, result.take_profit, features)
                    elif result.signal == "SELL" and has_pos:
                        self.sell(price, reason="sell_signal", features=features)
                    elif result.signal == "BUY" and has_pos:
                        self.log.info("BUY confirmed - holding.")
                    else:
                        self.log.debug("HOLD on %s", self.symbol)

                time.sleep(self.poll)

            except Exception as e:
                self.log.error("Cycle %d error on %s: %s", cycle, self.symbol, e, exc_info=True)
                time.sleep(self.poll)

    # -- Telegram notifications (no goal/target references) -------------------
    def _notify_startup(self) -> None:
        send_telegram(
            f"[{self.emoji}] <b>BOT STARTED: {self.symbol}</b>\n"
            f"Strategy: <b>{html.escape(self.strategy.NAME)}</b>\n"
            f"Timeframe: {self.tf_min}m  Poll: {self.poll}s\n"
            f"Max Concurrent Slots: {MAX_CONCURRENT_POSITIONS}\n"
            f"<i>{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}</i>"
        )

    def _notify_buy(self, qty: float, price: float, sl: float, tp: float,
                    stake: float, risk_mult: float) -> None:
        risk   = abs(price - sl)
        reward = abs(tp - price)
        rr     = (reward / risk) if risk > 0 else 0.0
        send_telegram(
            f"<b>[BUY] {self.symbol}</b> [{self.emoji}]\n"
            f"Strategy: {html.escape(self.strategy.NAME)}\n"
            f"Qty: <code>{qty:.6f}</code>  Stake: ${stake:,.2f}  Risk: {risk_mult:.2f}x\n"
            f"Fill: <b>${price:,.4f}</b>  SL: ${sl:,.4f}  TP: ${tp:,.4f}\n"
            f"R:R {rr:.2f}  (-${risk:,.2f} / +${reward:,.2f})\n"
            f"<i>{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}</i>"
        )

    def _notify_sell(self, qty: float, price: float, ep: float, pnl: float,
                     pnl_pct: float, reason: str, learning: dict) -> None:
        labels = {"take_profit": "[TP HIT]", "stop_loss": "[SL HIT]"}
        hdr    = labels.get(reason, "[SELL]")
        total  = self.memory.get_total_realized_pnl()
        cool   = "\nCooldown: 20 mins" if learning.get("cooldown_until") else ""
        label  = "WIN" if pnl >= 0 else "LOSS"
        sign   = "+" if pnl >= 0 else ""
        tsign  = "+" if total >= 0 else ""
        send_telegram(
            f"<b>{hdr} {self.symbol}</b> [{self.emoji}]\n"
            f"Strategy: {html.escape(self.strategy.NAME)}\n"
            f"Qty: <code>{qty:.6f}</code>  Entry: ${ep:,.4f}  Exit: ${price:,.4f}\n"
            f"Trade PnL: <b>{sign}${pnl:,.4f} ({pnl_pct:+.2f}%)</b>  {label}\n"
            f"Consec Losses: {learning['consecutive_losses']}  Risk Mult: {learning['risk_multiplier']:.2f}x{cool}\n"
            f"Total PnL: <b>{tsign}${total:,.4f}</b>\n"
            f"<i>{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}</i>"
        )

    def _notify_cycle(self, cycle: int, price: float, result, pos,
                      unreal: float, win_prob: float, risk_mult: float) -> None:
        interval = 3 if self.symbol == "BTC/USD" else 6
        if cycle > 2 and not pos and (cycle % interval != 0):
            return
        sig   = {"BUY": "[BUY]", "SELL": "[SELL]"}.get(result.signal, "[HOLD]")
        pline = (f"LONG {pos['qty']:.6f} @ ${pos['entry_price']:,.4f}  "
                 f"Unreal: {'+ ' if unreal>=0 else ''}${unreal:,.2f}"
                 if pos else "FLAT")
        total = self.memory.get_total_realized_pnl()
        tsign = "+" if total >= 0 else ""
        send_telegram(
            f"[{self.emoji}] <b>Cycle #{cycle} - {self.symbol}</b>\n"
            f"Price: <b>${price:,.4f}</b>  {sig}\n"
            f"AI: {round(win_prob*100)}% Win Prob  Risk: {risk_mult:.2f}x\n"
            f"<i>{html.escape(result.reason)}</i>\n"
            f"Pos: {pline}\n"
            f"Total PnL: {tsign}${total:,.4f}\n"
            f"<i>{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}</i>"
        )


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
_active_bot_threads: dict[str, threading.Thread] = {}
_bot_start_lock = threading.Lock()


def start_all_bots(symbol: Optional[str] = None) -> list:
    global API_KEY, API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_ENABLED
    load_dotenv()
    API_KEY            = os.getenv("ALPACA_API_KEY",    "") or API_KEY
    API_SECRET         = os.getenv("ALPACA_SECRET_KEY", "") or API_SECRET
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "") or TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "") or TELEGRAM_CHAT_ID
    TELEGRAM_ENABLED   = os.getenv("TELEGRAM_ENABLED", "true").lower() in ("true", "1", "yes")

    if not API_KEY or not API_SECRET:
        print("WARNING: Missing Alpaca API credentials.")
        return []

    with _bot_start_lock:
        targets     = [symbol] if symbol else list(ASSET_CONFIG.keys())
        started_new: list = []

        for sym in targets:
            if sym not in ASSET_CONFIG:
                print(f"Unknown symbol: {sym}")
                continue
            existing = _active_bot_threads.get(sym)
            if existing and existing.is_alive():
                continue
            cfg = ASSET_CONFIG[sym]
            try:
                bot = AssetBot(sym, cfg)
                t   = threading.Thread(target=bot.run, name=f"Bot-{sym}", daemon=True)
                t.start()
                _active_bot_threads[sym] = t
                started_new.append(sym)
                print(f"  Started {sym} ({cfg['strategy'].NAME})")
                time.sleep(1.5)
            except Exception as e:
                print(f"  Error starting {sym}: {e}")

        active = [t for t in _active_bot_threads.values() if t.is_alive()]
        if started_new:
            print(f"\n{len(active)} bots running.\n")
            names = ", ".join(f"[{ASSET_CONFIG[s]['emoji']}]{s.split('/')[0]}"
                              for s in _active_bot_threads if _active_bot_threads[s].is_alive())
            send_telegram(
                f"<b>[LAUNCH] EveryCryptoBot v3  ({len(active)} Bots)</b>\n\n"
                f"Assets: {names}\n"
                f"AI: Walk-Forward Ensemble Active\n"
                f"Max Concurrent Positions: {MAX_CONCURRENT_POSITIONS}\n"
                f"Dashboard: https://everycryptobot.onrender.com"
            )
        return active


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None,
                        help="Run a single asset (e.g. SOL/USD). Omit to run all 11.")
    args = parser.parse_args()
    start_all_bots(symbol=args.symbol)
    try:
        while True:
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
