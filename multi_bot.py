"""
EveryCryptoBot – Multi-Asset Paper Trading Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Runs multiple cryptos, each with their own strategy, in parallel threads.
BTC/USD is handled by the separate alpaca_btc_bot.py – DO NOT TOUCH IT.

Assets & Strategies:
  ETH/USD   → MomentumEmaRibbon (5-min)  – Trend following & EMA ribbon
  SOL/USD   → ScalpMacdRsi      (1-min)  – Fast momentum scalping
  DOGE/USD  → ScalpMacdRsi      (1-min)  – Volatility breakout scalping
  LINK/USD  → VwapBreakout      (15-min) – Anchored VWAP institutional breakout
  AVAX/USD  → MomentumEmaRibbon (5-min)  – Intraday trend expansion

Usage:
  python multi_bot.py                    # run all 5 assets in parallel
  python multi_bot.py --symbol SOL/USD   # run single asset
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

from strategies import ScalpMacdRsi, MomentumEmaRibbon, VwapBreakout

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

STAKE_USD = float(os.getenv("STAKE_USD_ALT", "300.0"))   # per alt-coin trade

# Asset → Strategy mapping
ASSET_CONFIG = {
    "ETH/USD": {
        "strategy": MomentumEmaRibbon,
        "timeframe_minutes": 5,
        "poll_secs": 60,
        "days_history": 4,
        "emoji": "🔷",
    },
    "SOL/USD": {
        "strategy": ScalpMacdRsi,
        "timeframe_minutes": 1,
        "poll_secs": 60,
        "days_history": 2,
        "emoji": "☀️",
    },
    "DOGE/USD": {
        "strategy": ScalpMacdRsi,
        "timeframe_minutes": 1,
        "poll_secs": 60,
        "days_history": 2,
        "emoji": "🐶",
    },
    "LINK/USD": {
        "strategy": VwapBreakout,
        "timeframe_minutes": 15,
        "poll_secs": 120,
        "days_history": 4,
        "emoji": "🔗",
    },
    "AVAX/USD": {
        "strategy": MomentumEmaRibbon,
        "timeframe_minutes": 5,
        "poll_secs": 60,
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
#  PERSISTENT STATE
# ─────────────────────────────────────────────────────────────────────────────
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multi_state.json")
_state_lock = threading.Lock()


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_asset_state(state: dict, symbol: str) -> dict:
    if symbol not in state:
        state[symbol] = {"position": None, "trades": [], "total_pnl": 0.0}
    return state[symbol]


def save_state(state: dict) -> None:
    with _state_lock:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
#  DASHBOARD LIVE STATE
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
#  ASSET BOT WORKER
# ─────────────────────────────────────────────────────────────────────────────
class AssetBot:
    def __init__(self, symbol: str, config: dict, global_state: dict):
        self.symbol  = symbol
        self.config  = config
        self.emoji   = config["emoji"]
        self.log     = _make_logger(symbol)
        self.state   = global_state
        self.a_state = get_asset_state(global_state, symbol)

        self.strategy = config["strategy"](symbol=symbol)
        self.tf_min   = config["timeframe_minutes"]
        self.tf       = TF_MAP[self.tf_min]
        self.poll     = config["poll_secs"]
        self.days_h   = config["days_history"]

        self.trading  = TradingClient(API_KEY, API_SECRET, paper=PAPER)
        self.data_cl  = CryptoHistoricalDataClient(API_KEY, API_SECRET)

        self.log.info("Initialized Strategy=%s TF=%dm Poll=%ds Stake=$%.2f",
                      self.strategy.NAME, self.tf_min, self.poll, STAKE_USD)

    # ── Market Data ──────────────────────────────────────────────────────────
    def fetch_bars(self) -> Optional[pd.DataFrame]:
        try:
            end   = datetime.now(timezone.utc)
            start = end - timedelta(days=self.days_h)
            req   = CryptoBarsRequest(symbol_or_symbols=self.symbol,
                                      timeframe=self.tf, start=start, end=end)
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

    # ── Orders & Execution ───────────────────────────────────────────────────
    def buy(self, price: float, sl: float, tp: float) -> bool:
        qty = round(STAKE_USD / price, 6)
        if qty <= 0:
            return False
        cash = self.get_cash()
        if cash < STAKE_USD:
            self.log.warning("Insufficient cash $%.2f for stake $%.2f", cash, STAKE_USD)
            return False
        try:
            order = self.trading.submit_order(MarketOrderRequest(
                symbol=self.symbol.replace("/", ""),
                qty=qty, side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
            ))
            entry = {
                "order_id": str(order.id),
                "entry_price": price,
                "qty": qty,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "stop_loss": sl,
                "take_profit": tp,
            }
            self.a_state["position"] = entry
            save_state(self.state)
            self.log.info("BUY %.6f %s @ $%.4f | SL=$%.4f TP=$%.4f",
                          qty, self.symbol, price, sl, tp)
            self._notify_buy(qty, price, sl, tp)
            return True
        except Exception as e:
            self.log.error("BUY order failed: %s", e)
            return False

    def sell(self, price: float, reason: str = "signal") -> bool:
        pos_qty = self.get_position_qty()
        if pos_qty <= 0:
            self.a_state["position"] = None
            save_state(self.state)
            return False
        try:
            self.trading.submit_order(MarketOrderRequest(
                symbol=self.symbol.replace("/", ""),
                qty=pos_qty, side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
            ))
            pos     = self.a_state.get("position", {})
            ep      = pos.get("entry_price", price)
            qty     = pos.get("qty", pos_qty)
            pnl     = (price - ep) * qty
            pnl_pct = ((price - ep) / ep) * 100 if ep else 0.0

            self.a_state["trades"].append({
                "entry_price": ep,
                "exit_price": price,
                "qty": qty,
                "pnl": round(pnl, 6),
                "pnl_pct": round(pnl_pct, 3),
                "reason": reason,
                "exit_time": datetime.now(timezone.utc).isoformat(),
            })
            self.a_state["total_pnl"] = round(
                self.a_state.get("total_pnl", 0.0) + pnl, 6)
            self.a_state["position"] = None
            save_state(self.state)

            self.log.info("SELL [%s] %.6f %s @ $%.4f | PnL $%.4f (%.2f%%) | %s",
                          "WIN" if pnl >= 0 else "LOSS", qty, self.symbol,
                          price, pnl, pnl_pct, reason)
            self._notify_sell(qty, price, ep, pnl, pnl_pct, reason)
            return True
        except Exception as e:
            self.log.error("SELL order failed: %s", e)
            return False

    def check_exit(self, price: float) -> bool:
        pos = self.a_state.get("position")
        if not pos:
            return False
        sl  = pos.get("stop_loss", 0.0)
        tp  = pos.get("take_profit", float("inf"))
        if price <= sl:
            self.log.warning("STOP-LOSS hit @ $%.4f (SL=$%.4f)", price, sl)
            self.sell(price, reason="stop_loss")
            return True
        if price >= tp:
            self.log.info("TAKE-PROFIT hit @ $%.4f (TP=$%.4f)", price, tp)
            self.sell(price, reason="take_profit")
            return True
        return False

    # ── Trading Loop ─────────────────────────────────────────────────────────
    def run(self):
        self.log.info("Starting paper trade loop for %s", self.symbol)
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
                exited  = self.check_exit(price)
                result  = self.strategy.get_signal(df, price)
                pos     = self.a_state.get("position")

                # Unrealized PnL
                unreal = 0.0
                if pos:
                    unreal = (price - pos["entry_price"]) * pos["qty"]

                # Extract indicator snapshot
                indicators = {}
                last = df.iloc[-1]
                for col in ["rsi", "macd", "macd_signal", "macd_hist", "ema200",
                            "ema55", "ema21", "ema13", "ema8", "vwap", "atr", "vol_ratio", "adx"]:
                    if col in df.columns:
                        v = last[col]
                        if pd.notna(v):
                            indicators[col] = round(float(v), 4)

                self.log.info("Price=$%.4f | Signal=%s | %s | PnL_Session=$%.4f",
                              price, result.signal, result.reason,
                              self.a_state.get("total_pnl", 0.0))

                # Update live dashboard state
                update_dashboard(self.symbol, {
                    "symbol": self.symbol,
                    "strategy": self.strategy.NAME,
                    "emoji": self.emoji,
                    "timeframe": f"{self.tf_min}m",
                    "price": price,
                    "signal": result.signal,
                    "reason": result.reason,
                    "confidence": result.confidence,
                    "indicators": indicators,
                    "position": pos,
                    "unrealized_pnl": round(unreal, 4),
                    "total_pnl": self.a_state.get("total_pnl", 0.0),
                    "trades_count": len(self.a_state.get("trades", [])),
                    "cycle": cycle,
                })

                # Telegram cycle notification
                self._notify_cycle(cycle, price, result, pos, unreal)

                # Order Execution
                if not exited:
                    has_pos = pos is not None
                    if result.signal == "BUY" and not has_pos:
                        self.buy(price, result.stop_loss, result.take_profit)
                    elif result.signal == "SELL" and has_pos:
                        self.sell(price, reason="sell_signal")
                    elif result.signal == "BUY" and has_pos:
                        self.log.info("BUY signal confirmed – holding active position.")
                    elif result.signal == "SELL" and not has_pos:
                        self.log.info("SELL signal – no open position to close.")
                    else:
                        self.log.info("HOLD – market scanning in progress.")

            except Exception as e:
                self.log.error("Loop error: %s", e, exc_info=True)

            time.sleep(self.poll)

    # ── Telegram Notifications ───────────────────────────────────────────────
    def _notify_startup(self):
        total = self.a_state.get("total_pnl", 0.0)
        tc    = len(self.a_state.get("trades", []))
        send_telegram(
            f"{self.emoji} <b>{self.symbol} Bot ONLINE</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 <b>Strategy:</b> {html.escape(self.strategy.NAME)}\n"
            f"⏱️ <b>Timeframe:</b> {self.tf_min}-min | Poll: {self.poll}s\n"
            f"💵 <b>Stake:</b> ${STAKE_USD:,.2f} per trade\n"
            f"💼 <b>Session PnL:</b> ${total:,.4f} ({tc} trades)\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</i>"
        )

    def _notify_buy(self, qty: float, price: float, sl: float, tp: float):
        risk   = price - sl
        reward = tp - price
        send_telegram(
            f"{self.emoji} 🟢 <b>BUY ORDER FILLED — {self.symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Strategy:</b> {html.escape(self.strategy.NAME)}\n"
            f"🔢 <b>Qty:</b> <code>{qty:.6f}</code> (~${qty*price:,.2f})\n"
            f"💵 <b>Entry Price:</b> <b>${price:,.4f}</b>\n"
            f"🛑 <b>Stop Loss:</b> ${sl:,.4f} (-${risk:,.2f})\n"
            f"🎯 <b>Take Profit:</b> ${tp:,.4f} (+${reward:,.2f})\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}</i>"
        )

    def _notify_sell(self, qty: float, price: float, ep: float,
                     pnl: float, pnl_pct: float, reason: str):
        total = self.a_state.get("total_pnl", 0.0)
        tc    = len(self.a_state.get("trades", []))
        hdr   = {
            "take_profit": f"🎯 <b>TAKE PROFIT HIT — {self.symbol}</b> 🟢",
            "stop_loss":   f"🛑 <b>STOP LOSS HIT — {self.symbol}</b> ⚠️",
        }.get(reason, f"🔴 <b>SELL EXECUTED — {self.symbol}</b>")
        send_telegram(
            f"{hdr}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Strategy:</b> {html.escape(self.strategy.NAME)}\n"
            f"🔢 <b>Qty:</b> <code>{qty:.6f}</code>\n"
            f"💵 <b>Entry / Exit:</b> ${ep:,.4f} → ${price:,.4f}\n"
            f"📈 <b>Trade PnL:</b> <b>{'+'if pnl>=0 else ''}${pnl:,.4f} ({pnl_pct:+.2f}%)</b> "
            f"{'✅ WIN' if pnl>=0 else '❌ LOSS'}\n"
            f"💼 <b>Cumulative PnL:</b> ${total:,.4f} ({tc} trades)\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}</i>"
        )

    def _notify_cycle(self, cycle: int, price: float, result, pos, unreal: float):
        sig_icon = {"BUY": "🟢", "SELL": "🔴"}.get(result.signal, "🟡")
        pos_line = (
            f"💼 LONG <code>{pos['qty']:.6f}</code> @ ${pos['entry_price']:,.4f} | "
            f"Unreal: {'+'if unreal>=0 else ''}${unreal:,.2f}"
            if pos else "💼 FLAT"
        )
        total = self.a_state.get("total_pnl", 0.0)
        tc    = len(self.a_state.get("trades", []))
        send_telegram(
            f"{self.emoji} <b>CYCLE #{cycle} — {self.symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 <b>Price:</b> <b>${price:,.4f}</b> | {sig_icon} <b>{result.signal}</b>\n"
            f"📊 <i>{html.escape(result.reason)}</i>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"{pos_line}\n"
            f"🏆 <b>Session PnL:</b> ${total:,.4f} ({tc} trades)\n"
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
        print("❌ Missing Alpaca API credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        sys.exit(1)

    global_state = load_state()

    if args.symbol:
        if args.symbol not in ASSET_CONFIG:
            print(f"❌ Unknown symbol: {args.symbol}. Available: {list(ASSET_CONFIG.keys())}")
            sys.exit(1)
        cfg = ASSET_CONFIG[args.symbol]
        bot = AssetBot(args.symbol, cfg, global_state)
        bot.run()
    else:
        # Run all assets as separate daemon threads
        threads = []
        for sym, cfg in ASSET_CONFIG.items():
            bot = AssetBot(sym, cfg, global_state)
            t   = threading.Thread(target=bot.run, name=f"Bot-{sym}", daemon=True)
            t.start()
            threads.append(t)
            print(f"✅ Started thread for {sym}  ({cfg['strategy'].NAME})")
            time.sleep(2)  # stagger startup

        print(f"\n🚀 All {len(threads)} asset bots running in parallel. Press Ctrl+C to stop.\n")
        try:
            while True:
                time.sleep(30)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down all bots...")


if __name__ == "__main__":
    main()
