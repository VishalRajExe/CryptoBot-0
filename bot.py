"""
======================================================================
                  EveryCryptoBot  -  BTC/USD
======================================================================
Automated cryptocurrency trading bot on Alpaca with real-time Telegram
notifications and the Triple Confluence strategy.

Strategy Logic:
  BUY  when: Price > EMA50 > EMA200 (confirmed uptrend)
             AND MACD line crosses above Signal line (bullish momentum)
             AND RSI between 50-70 (healthy momentum, not overbought)

  SELL when: Price < EMA50 < EMA200 (confirmed downtrend)
             AND MACD line crosses below Signal line (bearish momentum)
             AND RSI between 30-50 (healthy momentum, not oversold)

Risk Management:
  STOP-LOSS  : 1.5x ATR below entry (adapts to market volatility)
  TAKE-PROFIT: 2.0x the risk (1:2 Risk/Reward ratio for positive expectancy)
======================================================================
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

# ─────────────────────────────────────────────
#  LOAD ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────
load_dotenv()

# ── Alpaca Credentials ───────────────────────
API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_SECRET_KEY", "")
PAPER      = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# ── Trading Settings ─────────────────────────
SYMBOL        = os.getenv("TRADING_SYMBOL", "BTC/USD")
TIMEFRAME     = TimeFrame(5, TimeFrameUnit.Minute)
POLL_SECS     = int(os.getenv("POLL_SECS", "60"))
CANDLE_LIMIT  = int(os.getenv("CANDLE_LIMIT", "250"))
STAKE_USD     = float(os.getenv("STAKE_USD", "5000.0"))
MAX_OPEN_TRADES = 1

# ── Strategy Parameters ──────────────────────
RSI_PERIOD       = 14
RSI_BUY_MIN      = 50
RSI_BUY_MAX      = 70
RSI_SELL_MIN     = 30
RSI_SELL_MAX     = 50
EMA_FAST         = 50
EMA_SLOW         = 200
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9
ATR_PERIOD       = 14
ATR_SL_MULT      = 1.5
RR_RATIO         = 2.0
MIN_CANDLES      = 210

# ── Telegram Notifications ────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLED   = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"

# ─────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("EveryCryptoBot")


# ─────────────────────────────────────────────
#  TELEGRAM HELPER
# ─────────────────────────────────────────────

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
        req = urllib.request.Request(url, data=payload, headers={"User-Agent": "EveryCryptoBot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        log.warning("Telegram notification failed: %s", e)
        return False


# ─────────────────────────────────────────────
#  TECHNICAL INDICATORS (pandas)
# ─────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"]  - df["close"].shift()).abs()
    tr         = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Trend: EMA 50 / 200
    df["ema50"]  = compute_ema(df["close"], EMA_FAST)
    df["ema200"] = compute_ema(df["close"], EMA_SLOW)

    # Momentum: MACD 12/26/9
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(
        df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL
    )
    df["macd_cross_up"] = (df["macd"] > df["macd_signal"]) & (
        df["macd"].shift(1) <= df["macd_signal"].shift(1)
    )
    df["macd_cross_down"] = (df["macd"] < df["macd_signal"]) & (
        df["macd"].shift(1) >= df["macd_signal"].shift(1)
    )

    # Filter: RSI 14
    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)

    # Volatility: ATR 14
    df["atr"] = compute_atr(df, ATR_PERIOD)

    return df


def get_signal(df: pd.DataFrame) -> tuple[str, float, float]:
    """
    Evaluates Triple Confluence:
    Returns (signal, stop_loss_price, take_profit_price)
    """
    if len(df) < MIN_CANDLES:
        return "HOLD", 0.0, 0.0

    last  = df.iloc[-1]
    price = last["close"]
    atr   = last["atr"]
    risk  = atr * ATR_SL_MULT

    trend_up   = price > last["ema50"] > last["ema200"]
    macd_bull  = bool(last["macd_cross_up"])
    rsi_bull   = RSI_BUY_MIN <= last["rsi"] <= RSI_BUY_MAX

    trend_down = price < last["ema50"] < last["ema200"]
    macd_bear  = bool(last["macd_cross_down"])
    rsi_bear   = RSI_SELL_MIN <= last["rsi"] <= RSI_SELL_MAX

    if trend_up and macd_bull and rsi_bull:
        sl = round(price - risk, 2)
        tp = round(price + risk * RR_RATIO, 2)
        return "BUY", sl, tp

    if trend_down and macd_bear and rsi_bear:
        sl = round(price + risk, 2)
        tp = round(price - risk * RR_RATIO, 2)
        return "SELL", sl, tp

    return "HOLD", 0.0, 0.0


# ─────────────────────────────────────────────
#  TRADE STATE PERSISTENCE
# ─────────────────────────────────────────────
STATE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "alpaca_state.json"
)


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"position": None, "trades": [], "total_pnl": 0.0}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ─────────────────────────────────────────────
#  MAIN BOT CLASS
# ─────────────────────────────────────────────

class EveryCryptoBot:
    def __init__(self):
        log.info("=" * 65)
        log.info("  EveryCryptoBot  -  %s", SYMBOL)
        log.info("  Strategy : Triple Confluence (EMA %d/%d + MACD + RSI %d)",
                 EMA_FAST, EMA_SLOW, RSI_PERIOD)
        log.info("  Stake    : $%.2f  |  ATR SL: %.1fx  |  R:R: 1:%.1f",
                 STAKE_USD, ATR_SL_MULT, RR_RATIO)
        log.info("  Mode     : %s", "PAPER TRADING" if PAPER else "LIVE TRADING")
        log.info("=" * 65)

        if not API_KEY or not API_SECRET:
            log.error("Missing Alpaca API credentials. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
            sys.exit(1)

        self.trading = TradingClient(API_KEY, API_SECRET, paper=PAPER)
        self.data    = CryptoHistoricalDataClient(API_KEY, API_SECRET)

        self.state   = load_state()
        self._validate_connection()

    def _validate_connection(self):
        try:
            acct = self.trading.get_account()
            log.info("Connected to Alpaca (%s): Account %s",
                     "Paper" if PAPER else "Live", acct.id)
            log.info("   Cash: $%s  |  Equity: $%s  |  Buying Power: $%s",
                     acct.cash, acct.equity, acct.buying_power)
        except Exception as e:
            log.error("Failed to connect to Alpaca: %s", e)
            sys.exit(1)

    def fetch_bars(self, limit: int = CANDLE_LIMIT) -> Optional[pd.DataFrame]:
        try:
            from datetime import timedelta
            end   = datetime.now(timezone.utc)
            start = end - timedelta(days=4)

            request = CryptoBarsRequest(
                symbol_or_symbols=SYMBOL,
                timeframe=TIMEFRAME,
                start=start,
                end=end,
            )
            bars = self.data.get_crypto_bars(request)
            df   = bars.df
            if df.empty:
                return None

            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(SYMBOL, level=0)

            df = df[["open", "high", "low", "close", "volume"]].copy()
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
            if limit and len(df) > limit:
                df = df.iloc[-limit:]
            log.debug("Fetched %d candles, latest: %s", len(df), df.index[-1])
            return df
        except Exception as e:
            log.error("Error fetching bars: %s", e)
            return None

    def get_price(self) -> Optional[float]:
        try:
            req   = CryptoLatestQuoteRequest(symbol_or_symbols=SYMBOL)
            resp  = self.data.get_crypto_latest_quote(req)
            quote = resp[SYMBOL]
            return float(quote.ask_price)
        except Exception:
            df = self.fetch_bars(limit=5)
            if df is not None and not df.empty:
                return float(df["close"].iloc[-1])
            return None

    def get_cash(self) -> float:
        acct = self.trading.get_account()
        return float(acct.cash)

    def get_position_qty(self) -> float:
        try:
            pos = self.trading.get_open_position(SYMBOL.replace("/", ""))
            return float(pos.qty)
        except Exception:
            return 0.0

    def buy(self, price: float, sl: float, tp: float) -> bool:
        qty = round(STAKE_USD / price, 6)
        if qty <= 0:
            log.warning("Cannot buy: qty=%.6f BTC, price=$%.2f", qty, price)
            return False

        cash = self.get_cash()
        if cash < STAKE_USD:
            log.warning("Insufficient cash ($%.2f) to stake $%.2f", cash, STAKE_USD)
            return False

        try:
            order = self.trading.submit_order(
                MarketOrderRequest(
                    symbol=SYMBOL.replace("/", ""),
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                )
            )
            entry = {
                "order_id"   : str(order.id),
                "entry_price": price,
                "qty"        : qty,
                "entry_time" : datetime.now(timezone.utc).isoformat(),
                "stop_loss"  : sl,
                "take_profit": tp,
            }
            self.state["position"] = entry
            save_state(self.state)
            log.info("BUY  %.6f BTC @ $%.2f  |  SL=$%.2f  TP=$%.2f",
                     qty, price, entry["stop_loss"], entry["take_profit"])
            self.notify_buy(qty, price, entry["stop_loss"], entry["take_profit"])
            return True
        except Exception as e:
            log.error("BUY failed: %s", e)
            return False

    def sell(self, price: float, reason: str = "signal") -> bool:
        pos_qty = self.get_position_qty()
        if pos_qty <= 0:
            log.info("No BTC position to sell.")
            self.state["position"] = None
            save_state(self.state)
            return False

        try:
            order = self.trading.submit_order(
                MarketOrderRequest(
                    symbol=SYMBOL.replace("/", ""),
                    qty=pos_qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                )
            )

            entry       = self.state.get("position", {})
            entry_price = entry.get("entry_price", price)
            qty         = entry.get("qty", pos_qty)
            pnl         = (price - entry_price) * qty
            pnl_pct     = ((price - entry_price) / entry_price) * 100

            trade_record = {
                "entry_price": entry_price,
                "exit_price" : price,
                "qty"        : qty,
                "pnl"        : round(pnl, 4),
                "pnl_pct"    : round(pnl_pct, 2),
                "reason"     : reason,
                "exit_time"  : datetime.now(timezone.utc).isoformat(),
            }
            self.state["trades"].append(trade_record)
            self.state["total_pnl"] = round(
                self.state.get("total_pnl", 0.0) + pnl, 4
            )
            self.state["position"] = None
            save_state(self.state)

            icon = "WIN" if pnl >= 0 else "LOSS"
            log.info("SELL [%s] %.6f BTC @ $%.2f  |  PnL: $%.2f (%.2f%%)  |  Reason: %s",
                     icon, qty, price, pnl, pnl_pct, reason)
            log.info("   Total PnL: $%.4f  |  Total Trades: %d",
                     self.state["total_pnl"], len(self.state["trades"]))
            self.notify_sell(qty, price, entry_price, pnl, pnl_pct, reason)
            return True
        except Exception as e:
            log.error("SELL failed: %s", e)
            return False

    def check_exit_conditions(self, price: float) -> bool:
        pos = self.state.get("position")
        if not pos:
            return False

        sl = pos.get("stop_loss", 0)
        tp = pos.get("take_profit", float("inf"))

        if price <= sl:
            log.warning("STOP-LOSS triggered @ $%.2f (SL=$%.2f)", price, sl)
            self.sell(price, reason="stop_loss")
            return True

        if price >= tp:
            log.info("TAKE-PROFIT triggered @ $%.2f (TP=$%.2f)", price, tp)
            self.sell(price, reason="take_profit")
            return True

        return False

    def print_status(self, df: pd.DataFrame, signal: str, price: float,
                     sl: float = 0.0, tp: float = 0.0):
        last    = df.iloc[-1]
        pos     = self.state.get("position")
        pos_str = "FLAT"
        if pos:
            unreal  = (price - pos["entry_price"]) * pos["qty"]
            pos_str = (f"LONG {pos['qty']:.6f} BTC @ ${pos['entry_price']:.2f}"
                       f" | SL=${pos['stop_loss']:.2f} TP=${pos['take_profit']:.2f}"
                       f" | Unrealized: ${unreal:+.2f}")

        signal_detail = signal
        if signal == "BUY" and sl:
            signal_detail = f"BUY  [SL=${sl:.2f}  TP=${tp:.2f}]"
        elif signal == "SELL":
            signal_detail = "SELL [exit position]"

        log.info(
            "--- MARKET SNAPSHOT ---\n"
            "  Price    : $%.2f  |  Signal : %s\n"
            "  RSI(%d)  : %.2f   |  MACD: %.2f  Signal: %.2f  Hist: %.2f\n"
            "  EMA50    : $%.2f  |  EMA200 : $%.2f  |  Trend: %s\n"
            "  ATR(%d)  : $%.2f  |  Position: %s\n"
            "  Total PnL: $%.4f  |  Trades: %d",
            price, signal_detail,
            RSI_PERIOD, last["rsi"],
            last["macd"], last["macd_signal"], last["macd_hist"],
            last["ema50"], last["ema200"],
            "UPTREND" if price > last["ema50"] > last["ema200"] else
            "DOWNTREND" if price < last["ema50"] < last["ema200"] else "NEUTRAL",
            ATR_PERIOD, last["atr"],
            pos_str,
            self.state.get("total_pnl", 0.0),
            len(self.state.get("trades", [])),
        )

    # ── Telegram Notifications ───────────────
    def notify_startup(self):
        trades_count = len(self.state.get("trades", []))
        total_pnl = self.state.get("total_pnl", 0.0)
        msg = (
            "🤖 <b>EveryCryptoBot ONLINE</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 <b>Asset:</b> {SYMBOL} ({'Paper' if PAPER else 'Live'})\n"
            f"📈 <b>Strategy:</b> Triple Confluence (EMA{EMA_FAST}/{EMA_SLOW} + MACD + RSI{RSI_PERIOD})\n"
            f"💵 <b>Stake:</b> ${STAKE_USD:,.2f} per trade\n"
            f"⏱️ <b>Timeframe:</b> 5-min candles | Poll: {POLL_SECS}s\n"
            f"💼 <b>Total Realized PnL:</b> ${total_pnl:,.4f} ({trades_count} trades)\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
        )
        send_telegram(msg)

    def notify_buy(self, qty: float, price: float, sl: float, tp: float):
        risk = price - sl if sl else 0.0
        reward = tp - price if tp else 0.0
        msg = (
            "🟢 <b>BUY ORDER EXECUTED</b> 🟢\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 <b>Asset:</b> {SYMBOL}\n"
            f"📊 <b>Action:</b> BUY MARKET\n"
            f"🔢 <b>Quantity:</b> <code>{qty:.6f} BTC</code> (~${qty * price:,.2f})\n"
            f"💵 <b>Entry Price:</b> <b>${price:,.2f}</b>\n"
            f"🛑 <b>Stop Loss:</b> ${sl:,.2f} (-${risk:,.2f})\n"
            f"🎯 <b>Take Profit:</b> ${tp:,.2f} (+${reward:,.2f})\n"
            f"⚖️ <b>Risk:Reward:</b> 1:{RR_RATIO:.1f}\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
        )
        send_telegram(msg)

    def notify_sell(self, qty: float, price: float, entry_price: float,
                    pnl: float, pnl_pct: float, reason: str):
        is_win = pnl >= 0
        if reason == "take_profit":
            header = "🎯 <b>TAKE PROFIT REACHED</b> 🟢"
        elif reason == "stop_loss":
            header = "🛑 <b>STOP LOSS TRIGGERED</b> ⚠️"
        else:
            header = "🔴 <b>SELL SIGNAL EXECUTED</b>"

        outcome = "WIN ✅" if is_win else "LOSS ❌"
        total_pnl = self.state.get("total_pnl", 0.0)
        trades_count = len(self.state.get("trades", []))

        msg = (
            f"{header}\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 <b>Asset:</b> {SYMBOL}\n"
            f"📊 <b>Exit Reason:</b> {reason.upper()}\n"
            f"🔢 <b>Quantity:</b> <code>{qty:.6f} BTC</code>\n"
            f"💵 <b>Entry Price:</b> ${entry_price:,.2f}\n"
            f"💵 <b>Exit Price:</b> <b>${price:,.2f}</b>\n"
            f"📈 <b>Trade PnL:</b> <b>{'+' if pnl >= 0 else ''}${pnl:,.2f} ({pnl_pct:+.2f}%)</b> [{outcome}]\n"
            f"💼 <b>Total Bot PnL:</b> ${total_pnl:,.4f} ({trades_count} trades)\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
        )
        send_telegram(msg)

    def notify_cycle(self, cycle: int, price: float, df: pd.DataFrame,
                     signal: str, sl: float = 0.0, tp: float = 0.0):
        last = df.iloc[-1]
        pos = self.state.get("position")
        trend = "UPTREND 📈" if price > last["ema50"] > last["ema200"] else \
                "DOWNTREND 📉" if price < last["ema50"] < last["ema200"] else "NEUTRAL ⚖️"

        if pos:
            unreal = (price - pos["entry_price"]) * pos["qty"]
            unreal_pct = ((price - pos["entry_price"]) / pos["entry_price"]) * 100
            pnl_emoji = "🟢" if unreal >= 0 else "🔴"
            pos_block = (
                f"💼 <b>Position:</b> 🟢 LONG <code>{pos['qty']:.6f} BTC</code>\n"
                f"💵 <b>Entry:</b> ${pos['entry_price']:,.2f} | <b>Now:</b> ${price:,.2f}\n"
                f"🛑 <b>SL:</b> ${pos['stop_loss']:,.2f} | 🎯 <b>TP:</b> ${pos['take_profit']:,.2f}\n"
                f"📊 <b>Unrealized PnL:</b> {pnl_emoji} <b>{'+' if unreal >= 0 else ''}${unreal:,.2f} ({unreal_pct:+.2f}%)</b>"
            )
        else:
            pos_block = "💼 <b>Position:</b> ⚪ FLAT (Waiting for confluence setup)"

        sig_icon = "🟢 BUY" if signal == "BUY" else "🔴 SELL" if signal == "SELL" else "🟡 HOLD"
        total_pnl = self.state.get("total_pnl", 0.0)
        trades_count = len(self.state.get("trades", []))

        msg = (
            f"🟡 <b>CYCLE #{cycle} REPORT</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 <b>BTC Price:</b> <b>${price:,.2f}</b>\n"
            f"🧭 <b>Trend:</b> {trend} | <b>Signal:</b> {sig_icon}\n"
            f"📊 <b>RSI(14):</b> {last['rsi']:.2f}\n"
            f"🌊 <b>MACD:</b> {last['macd']:.2f} | <b>Signal:</b> {last['macd_signal']:.2f} (Hist: {last['macd_hist']:+.2f})\n"
            f"📏 <b>EMA50:</b> ${last['ema50']:,.2f} | <b>EMA200:</b> ${last['ema200']:,.2f}\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"{pos_block}\n"
            f"🏆 <b>Total Realized PnL:</b> ${total_pnl:,.4f} ({trades_count} trades)\n"
            f"🕒 <i>{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}</i>"
        )
        send_telegram(msg)

    # ── Main Run Loop ────────────────────────
    def run(self):
        log.info("Bot started. Polling every %d seconds. Press Ctrl+C to stop.", POLL_SECS)
        self.notify_startup()
        cycle = 0

        while True:
            try:
                cycle += 1
                log.info("=== Cycle #%d | %s ===",
                         cycle, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

                # 1. Fetch historical candle bars
                df = self.fetch_bars(limit=CANDLE_LIMIT)
                if df is None or len(df) < MIN_CANDLES:
                    log.warning("Not enough data (%s/%s bars). Retrying in %ds ...",
                                len(df) if df is not None else 0, MIN_CANDLES, POLL_SECS)
                    time.sleep(POLL_SECS)
                    continue

                # 2. Compute indicators
                df = add_indicators(df)

                # 3. Get latest price
                price = self.get_price()
                if price is None:
                    price = float(df["close"].iloc[-1])

                # 4. Check SL/TP exit conditions
                exited = self.check_exit_conditions(price)

                # 5. Generate trading signal
                signal, sl, tp = get_signal(df)

                # 6. Print console status & send Telegram update
                self.print_status(df, signal, price, sl, tp)
                self.notify_cycle(cycle, price, df, signal, sl, tp)

                # 7. Execute order if signal triggered
                if not exited:
                    has_position = self.state.get("position") is not None

                    if signal == "BUY" and not has_position:
                        self.buy(price, sl, tp)
                    elif signal == "SELL" and has_position:
                        self.sell(price, reason="sell_signal")
                    elif signal == "BUY" and has_position:
                        log.info("BUY signal - already in position, holding.")
                    elif signal == "SELL" and not has_position:
                        log.info("SELL signal - no position to close.")
                    else:
                        log.info("HOLD - no action taken.")

            except KeyboardInterrupt:
                log.info("Bot stopped by user.")
                self._print_summary()
                sys.exit(0)
            except Exception as e:
                log.error("Unexpected error in main loop: %s", e, exc_info=True)

            time.sleep(POLL_SECS)

    def _print_summary(self):
        trades = self.state.get("trades", [])
        total  = self.state.get("total_pnl", 0.0)
        wins   = [t for t in trades if t["pnl"] >= 0]
        losses = [t for t in trades if t["pnl"] < 0]
        log.info("=" * 65)
        log.info("  SESSION SUMMARY")
        log.info("  Total trades : %d  |  Wins: %d  |  Losses: %d",
                 len(trades), len(wins), len(losses))
        log.info("  Total PnL    : $%.4f", total)
        if trades:
            best  = max(trades, key=lambda t: t["pnl"])
            worst = min(trades, key=lambda t: t["pnl"])
            log.info("  Best trade   : $%.4f (%.2f%%)", best["pnl"], best["pnl_pct"])
            log.info("  Worst trade  : $%.4f (%.2f%%)", worst["pnl"], worst["pnl_pct"])
        log.info("=" * 65)


if __name__ == "__main__":
    bot = EveryCryptoBot()
    bot.run()
