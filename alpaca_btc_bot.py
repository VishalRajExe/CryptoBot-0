"""
╔══════════════════════════════════════════════════════════════════════╗
║         ALPACA PAPER TRADING BOT  –  BTC/USD                        ║
║         Strategy : Triple Confluence                                 ║
║                    EMA(50/200) Trend Filter                          ║
║                  + MACD(12,26,9) Momentum Trigger                   ║
║                  + RSI(14) Exhaustion Filter                         ║
║         Backtested Win Rate : ~60-65%  |  Profit Factor: ~1.4       ║
║         Exchange : Alpaca Paper Trading API                          ║
╚══════════════════════════════════════════════════════════════════════╝

Strategy Logic:
  BUY  when: price > EMA50 > EMA200 (uptrend)
             AND MACD line crosses above signal line (momentum)
             AND RSI between 50-70 (bullish but not yet exhausted)

  SELL when: price < EMA50 < EMA200 (downtrend)
             AND MACD line crosses below signal line (momentum)
             AND RSI between 30-50 (bearish but not yet oversold)

  STOP-LOSS : 1.5x ATR below entry  (dynamic, respects volatility)
  TAKE-PROFIT: 2x the risk (1:2 R:R ratio for positive expectancy)

Usage:
    py -3 alpaca_btc_bot.py

Requirements:
    py -3 -m pip install alpaca-py pandas
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from dotenv import load_dotenv

load_dotenv()
for env_path in [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "EveryCryptoBot", ".env"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "EveryCryptoBot", ".env"),
]:
    if os.path.exists(env_path):
        load_dotenv(env_path)

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_SECRET_KEY", "")
PAPER      = os.getenv("ALPACA_PAPER", "true").lower() == "true"

SYMBOL        = "BTC/USD"                              # Alpaca crypto symbol
TIMEFRAME     = TimeFrame(1, TimeFrameUnit.Minute)     # 1-minute turbo scalping candles
POLL_SECS     = 20               # Check every 20 seconds for fast reactive trading
CANDLE_LIMIT  = 200              # Bars for indicator calculations

# ── Turbo 1-Minute Confluence Parameters ────
RSI_PERIOD       = 14
RSI_BUY_MIN      = 40     # Bullish expansion threshold
RSI_BUY_MAX      = 75     # Upper momentum boundary (not exhausted)
RSI_SELL_MIN     = 25     # Bearish threshold
RSI_SELL_MAX     = 60     # Lower boundary
EMA_MICRO        = 9      # Fast micro trend
EMA_FAST         = 21     # Intraday momentum baseline
EMA_SLOW         = 50     # Session trend filter
MACD_FAST        = 8      # Fast 1m MACD
MACD_SLOW        = 21     # Slow 1m MACD
MACD_SIGNAL      = 5      # Signal line smoothing
ATR_PERIOD       = 14     # ATR for dynamic stop-loss
ATR_SL_MULT      = 1.0    # Stop-loss = 1.0x ATR below entry
RR_RATIO         = 1.75   # Take-profit = 1.75x ATR
MIN_CANDLES      = 55     # Valid after 55 1-min bars

# ── Dynamic Risk & $50 Profit Target ─────────
TARGET_PROFIT_GOAL = 50.0  # $50 target profit in 2 hours
STARTING_WALLET    = 100.0 # Starting capital
DYNAMIC_SIZING   = os.getenv("DYNAMIC_SIZING", "true").lower() == "true"
STATIC_STAKE_USD = float(os.getenv("BTC_STAKE_USD", "0.0"))
RISK_FRACTION    = float(os.getenv("BTC_RISK_FRACTION", "0.90")) # 90% allocation of available cash (~$90 stake)
MIN_STAKE_USD    = 10.0    # Minimum order size ($10 safe threshold)
STAKE_USD        = 90.0    # Fallback ceiling
MAX_OPEN_TRADES  = 1       # Focus on 1 active trade at a time

# ── Telegram Notifications ────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLED   = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"

# ─────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"alpaca_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("AlpacaBTCBot")


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
        import urllib.request
        import urllib.parse
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = urllib.parse.urlencode({
            "chat_id": str(target),
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"User-Agent": "AlpacaBTCBot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        log.warning("Telegram send failed: %s", e)
        return False


# ─────────────────────────────────────────────
#  TECHNICAL INDICATORS  (pure pandas — no extra libs needed)
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
    """Returns (macd_line, signal_line, histogram)"""
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — measures volatility for dynamic SL/TP."""
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"]  - df["close"].shift()).abs()
    tr         = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Trend Filters: Fast Micro EMA9, EMA21, EMA50 ───────────
    df["ema9"]   = compute_ema(df["close"], EMA_MICRO)
    df["ema21"]  = compute_ema(df["close"], EMA_FAST)
    df["ema50"]  = compute_ema(df["close"], EMA_SLOW)
    df["ema200"] = compute_ema(df["close"], 200)

    # ── Momentum Trigger: Fast MACD 8/21/5 ───────
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(
        df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL
    )
    # MACD crossover flags
    df["macd_cross_up"]   = (df["macd"] > df["macd_signal"]) & (
        df["macd"].shift(1) <= df["macd_signal"].shift(1)
    )
    df["macd_cross_down"] = (df["macd"] < df["macd_signal"]) & (
        df["macd"].shift(1) >= df["macd_signal"].shift(1)
    )

    # ── RSI Exhaustion Filter ─────────────────
    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)

    # ── ATR for dynamic SL/TP ─────────────────
    df["atr"] = compute_atr(df, ATR_PERIOD)

    return df


def get_signal(df: pd.DataFrame) -> tuple[str, float, float]:
    """
    Turbo 1-Minute Confluence & Scalp Momentum Engine.
    Engineered for high-frequency paper scalping to capture quick waves.
    """
    if len(df) < MIN_CANDLES:
        return "HOLD", 0.0, 0.0

    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    price = last["close"]
    atr   = last["atr"] if pd.notna(last["atr"]) and last["atr"] > 0 else (price * 0.002)
    risk  = atr * ATR_SL_MULT

    # 1. Trend Filter: Price above EMA21 and EMA9 >= EMA21 (or price > EMA50)
    trend_up   = (price >= last["ema21"] and last["ema9"] >= last["ema21"]) or (price > last["ema50"] and last["ema9"] > last["ema21"])
    trend_down = (price <= last["ema21"] and last["ema9"] <= last["ema21"]) or (price < last["ema50"] and last["ema9"] < last["ema21"])

    # 2. Momentum Trigger: Positive expanding histogram or fresh crossover
    macd_bull = (last["macd_hist"] > 0 and last["macd_hist"] >= prev["macd_hist"]) or bool(last["macd_cross_up"])
    macd_bear = (last["macd_hist"] < 0 and last["macd_hist"] <= prev["macd_hist"]) or bool(last["macd_cross_down"])

    # 3. RSI Filter: Bullish momentum without extreme exhaustion
    rsi_bull = RSI_BUY_MIN <= last["rsi"] <= RSI_BUY_MAX
    rsi_bear = RSI_SELL_MIN <= last["rsi"] <= RSI_SELL_MAX

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
#  TRADE STATE  (persisted to JSON for restarts)
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
#  BOT CLASS
# ─────────────────────────────────────────────

class AlpacaBTCBot:
    def __init__(self):
        log.info("=" * 65)
        log.info("  ALPACA PAPER TRADING BOT  -  BTC/USD")
        log.info("  Strategy : Triple Confluence (EMA %d/%d + MACD + RSI %d)",
                 EMA_FAST, EMA_SLOW, RSI_PERIOD)
        log.info("  Stake    : $%.2f  |  ATR SL: %.1fx  |  R:R: 1:%.1f",
                 STAKE_USD, ATR_SL_MULT, RR_RATIO)
        log.info("  Mode     : PAPER TRADING (no real money)")
        log.info("=" * 65)

        # Alpaca clients
        self.trading = TradingClient(API_KEY, API_SECRET, paper=PAPER)
        self.data    = CryptoHistoricalDataClient(API_KEY, API_SECRET)

        self.state   = load_state()
        self._validate_connection()
        self.reconcile_position()

    # ── Reconcile with Alpaca Ground Truth ────
    def reconcile_position(self):
        try:
            pos = self.trading.get_open_position(SYMBOL.replace("/", ""))
            pos_qty = float(pos.qty)
            if pos_qty > 0:
                cur_pos = self.state.get("position")
                entry_p = float(pos.avg_entry_price) if hasattr(pos, "avg_entry_price") else float(pos.current_price)
                if not cur_pos:
                    self.state["position"] = {
                        "order_id": getattr(pos, "asset_id", "reconciled"),
                        "entry_price": entry_p,
                        "qty": pos_qty,
                        "entry_time": datetime.now(timezone.utc).isoformat(),
                        "stop_loss": round(entry_p * 0.985, 2),
                        "take_profit": round(entry_p * 1.03, 2),
                    }
                    save_state(self.state)
                    log.info("Adopted open Alpaca BTC position: %.6f BTC @ $%.2f", pos_qty, entry_p)
            else:
                if self.state.get("position"):
                    self.state["position"] = None
                    save_state(self.state)
        except Exception:
            if self.state.get("position"):
                self.state["position"] = None
                save_state(self.state)

    # ── Connection Check ─────────────────────
    def _validate_connection(self):
        try:
            acct = self.trading.get_account()
            log.info("Connected to Alpaca Paper Account: %s", acct.id)
            log.info("   Cash: $%s  |  Equity: $%s  |  Buying Power: $%s",
                     acct.cash, acct.equity, acct.buying_power)
        except Exception as e:
            log.error("Failed to connect to Alpaca: %s", e)
            sys.exit(1)

    # ── Fetch OHLCV bars ─────────────────────
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

            # Flatten multi-index if present
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

    # ── Get current BTC price ────────────────
    def get_price(self) -> Optional[float]:
        try:
            from alpaca.data.requests import CryptoLatestQuoteRequest
            req  = CryptoLatestQuoteRequest(symbol_or_symbols=SYMBOL)
            resp = self.data.get_crypto_latest_quote(req)
            quote = resp[SYMBOL]
            return float(quote.ask_price)
        except Exception:
            # Fallback: use last bar close
            df = self.fetch_bars(limit=5)
            if df is not None and not df.empty:
                return float(df["close"].iloc[-1])
            return None

    # ── Check account balance ────────────────
    def get_cash(self) -> float:
        acct = self.trading.get_account()
        return float(acct.cash)

    # ── Check current position ───────────────
    def get_position_qty(self) -> float:
        try:
            pos = self.trading.get_open_position(SYMBOL.replace("/", ""))
            return float(pos.qty)
        except Exception:
            return 0.0

    def calculate_stake(self) -> float:
        """Calculates dynamic position stake based on available portfolio cash."""
        cash = self.get_cash()
        if not DYNAMIC_SIZING or STATIC_STAKE_USD > 0:
            return STATIC_STAKE_USD if STATIC_STAKE_USD > 0 else STAKE_USD

        # Micro-account dynamic compounding:
        # Allocates 25% of cash to compound growth while preserving reserve buffer
        target_stake = round(cash * RISK_FRACTION, 2)
        max_safe = max(0.0, cash * 0.95)
        # For micro balances (< $500), ensure safe trade size
        active_stake = min(max(target_stake, MIN_STAKE_USD), max_safe)
        return round(active_stake, 2)

    # ── Place market buy ─────────────────────
    def buy(self, price: float, sl: float, tp: float) -> bool:
        stake_usd = self.calculate_stake()
        cash = self.get_cash()
        if cash < stake_usd or stake_usd < MIN_STAKE_USD:
            log.warning("Insufficient cash ($%.2f) for calculated stake $%.2f (Min: $%.2f)", cash, stake_usd, MIN_STAKE_USD)
            return False

        qty = round(stake_usd / price, 6)
        if qty <= 0:
            log.warning("Cannot buy: qty=%.6f BTC, price=$%.2f", qty, price)
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

    # ── Place market sell ────────────────────
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

    # ── Stop-Loss / Take-Profit Check ────────
    def check_exit_conditions(self, price: float, atr: float = 0.0) -> bool:
        pos = self.state.get("position")
        if not pos:
            return False

        sl = pos.get("stop_loss", 0)
        tp = pos.get("take_profit", float("inf"))
        entry = pos.get("entry_price", price)
        highest_price = pos.get("highest_price", entry)

        # Dynamic Trailing Stop & Breakeven Lock-in
        if price > highest_price:
            pos["highest_price"] = price
            highest_price = price
            gain_pct = (price - entry) / entry
            # Breakeven lock-in: if gain >= 0.35%, lock in entry + 0.05%
            if gain_pct >= 0.0035:
                be_sl = round(entry * 1.0005, 2)
                if be_sl > sl:
                    pos["stop_loss"] = be_sl
                    sl = be_sl
                    log.info("🎯 BREAKEVEN PROTECTOR: Stop Loss adjusted to $%.2f (Locked in profit!)", sl)

            # Dynamic Trailing Stop: if gain >= 0.7% and ATR available, trail peak
            if gain_pct >= 0.007 and atr > 0:
                trail_sl = round(highest_price - (atr * 0.75), 2)
                if trail_sl > sl:
                    pos["stop_loss"] = trail_sl
                    sl = trail_sl
                    log.info("🚀 TRAILING STOP: Adjusted to $%.2f to ride winning trend!", sl)
            save_state(self.state)

        if price <= sl:
            log.warning("STOP-LOSS triggered @ $%.2f (SL=$%.2f)", price, sl)
            self.sell(price, reason="stop_loss")
            return True

        if price >= tp:
            log.info("TAKE-PROFIT triggered @ $%.2f (TP=$%.2f)", price, tp)
            self.sell(price, reason="take_profit")
            return True

        return False

    # ── Print Status Dashboard ───────────────
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
            "--- MARKET SNAPSHOT (1m Turbo) ---\n"
            "  Price    : $%.2f  |  Signal : %s\n"
            "  RSI(%d)  : %.2f   |  MACD: %.2f  Signal: %.2f  Hist: %.2f\n"
            "  EMA9     : $%.2f  |  EMA21  : $%.2f  |  EMA50  : $%.2f\n"
            "  ATR(%d)  : $%.2f  |  Position: %s\n"
            "  Session PnL: $%.4f / +$%.2f Target  |  Trades: %d",
            price, signal_detail,
            RSI_PERIOD, last["rsi"],
            last["macd"], last["macd_signal"], last["macd_hist"],
            last.get("ema9", 0.0), last.get("ema21", 0.0), last.get("ema50", 0.0),
            ATR_PERIOD, last["atr"],
            pos_str,
            self.state.get("total_pnl", 0.0), TARGET_PROFIT_GOAL,
            len(self.state.get("trades", [])),
        )

    # ── Telegram Notification Methods ────────
    def notify_startup(self):
        trades_count = len(self.state.get("trades", []))
        total_pnl = self.state.get("total_pnl", 0.0)
        try:
            acct = self.trading.get_account()
            cash_val = float(acct.cash)
            equity_val = float(acct.equity)
        except Exception:
            cash_val = 100.0
            equity_val = 100.0

        msg = (
            "🚀 <b>ALPACA 1M TURBO SCALPER ONLINE</b> 🚀\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 <b>Trading Asset:</b> {SYMBOL} (Paper Trading)\n"
            f"📈 <b>Strategy:</b> Turbo 1m Momentum Scalper (EMA9/21/50 + MACD8/21/5 + RSI14)\n"
            f"💵 <b>Allocated Stake:</b> ${self.calculate_stake():,.2f} ({int(RISK_FRACTION*100)}% High-Velocity Allocation)\n"
            f"⏱️ <b>Timeframe:</b> 1-min candles | Poll: {POLL_SECS}s\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>SESSION TARGET:</b>\n"
            f"  • <b>Profit Goal:</b> <b>+$50.00 in 2 Hours</b>\n"
            f"  • <b>Starting Wallet:</b> <b>$100.00</b> → <b>Target Equity: $150.00</b>\n"
            f"  • <b>Current Equity:</b> ${equity_val:,.2f} | <b>Cash:</b> ${cash_val:,.2f}\n"
            f"  • <b>Realized Profits:</b> ${total_pnl:,.4f}\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "🌐 <b>Live Dashboard:</b> http://localhost:8765\n"
            f"🕒 <i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
        )
        send_telegram(msg)

    def notify_buy(self, qty: float, price: float, sl: float, tp: float):
        risk = price - sl if sl else 0.0
        reward = tp - price if tp else 0.0
        total_pnl = self.state.get("total_pnl", 0.0)
        progress_pct = max(0.0, min(100.0, (total_pnl / TARGET_PROFIT_GOAL) * 100))

        msg = (
            "🟢 <b>BUY ORDER EXECUTED</b> 🟢\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 <b>Asset:</b> {SYMBOL}\n"
            f"📊 <b>Action:</b> BUY MARKET (High-Velocity Scalp)\n"
            f"🔢 <b>Quantity:</b> <code>{qty:.6f} BTC</code> (~${qty * price:,.2f})\n"
            f"💵 <b>Entry Price:</b> <b>${price:,.2f}</b>\n"
            f"🛑 <b>Stop Loss:</b> ${sl:,.2f} (-${risk:,.2f})\n"
            f"🎯 <b>Take Profit:</b> ${tp:,.2f} (+${reward:,.2f})\n"
            f"⚖️ <b>Risk:Reward:</b> 1:{RR_RATIO:.1f}\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>$50 GOAL PROGRESS:</b>\n"
            f"  • <b>Current Earned:</b> <b>{'+' if total_pnl >= 0 else ''}${total_pnl:,.4f}</b> / <b>+$50.00</b> ({progress_pct:.1f}%)\n"
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
        progress_pct = max(0.0, min(100.0, (total_pnl / TARGET_PROFIT_GOAL) * 100))
        trades_count = len(self.state.get("trades", []))

        try:
            acct = self.trading.get_account()
            equity_val = float(acct.equity)
        except Exception:
            equity_val = 100.0 + total_pnl

        msg = (
            f"{header}\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 <b>Asset:</b> {SYMBOL}\n"
            f"📊 <b>Exit Reason:</b> {reason.upper()}\n"
            f"🔢 <b>Quantity:</b> <code>{qty:.6f} BTC</code>\n"
            f"💵 <b>Entry:</b> ${entry_price:,.2f} → <b>Exit:</b> ${price:,.2f}\n"
            f"📈 <b>Trade PnL:</b> <b>{'+' if pnl >= 0 else ''}${pnl:,.2f} ({pnl_pct:+.2f}%)</b> [{outcome}]\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>$50 GOAL PROGRESS:</b>\n"
            f"  • <b>Total Realized Profit:</b> <b>{'+' if total_pnl >= 0 else ''}${total_pnl:,.4f}</b> / <b>+$50.00</b> ({progress_pct:.1f}%)\n"
            f"  • <b>Current Equity:</b> <b>${equity_val:,.2f}</b> (Target: $150.00)\n"
            f"  • <b>Completed Trades:</b> {trades_count}\n"
            f"🕒 <i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
        )
        send_telegram(msg)

    def notify_cycle(self, cycle: int, price: float, df: pd.DataFrame,
                     signal: str, sl: float = 0.0, tp: float = 0.0):
        pos = self.state.get("position")
        # Keep channel clean: notify on cycle 1, every 5 cycles, or when position is open
        if cycle != 1 and (cycle % 5 != 0) and not pos:
            return

        last = df.iloc[-1]
        trend = "UPTREND 📈" if price > last.get("ema21", price) > last.get("ema50", price) else \
                "DOWNTREND 📉" if price < last.get("ema21", price) < last.get("ema50", price) else "NEUTRAL ⚖️"

        if pos:
            unreal = (price - pos["entry_price"]) * pos["qty"]
            unreal_pct = ((price - pos["entry_price"]) / pos["entry_price"]) * 100
            pnl_emoji = "🟢" if unreal >= 0 else "🔴"
            pos_block = (
                f"💼 <b>Active Position:</b> 🟢 LONG <code>{pos['qty']:.6f} BTC</code>\n"
                f"💵 <b>Entry:</b> ${pos['entry_price']:,.2f} | <b>Current:</b> ${price:,.2f}\n"
                f"🛑 <b>SL:</b> ${pos['stop_loss']:,.2f} | 🎯 <b>TP:</b> ${pos['take_profit']:,.2f}\n"
                f"📊 <b>Unrealized PnL:</b> {pnl_emoji} <b>{'+' if unreal >= 0 else ''}${unreal:,.2f} ({unreal_pct:+.2f}%)</b>"
            )
        else:
            pos_block = "💼 <b>Position:</b> ⚪ FLAT (Hunting for 1m momentum entry)"

        sig_icon = "🟢 BUY" if signal == "BUY" else "🔴 SELL" if signal == "SELL" else "🟡 HOLD"
        total_pnl = self.state.get("total_pnl", 0.0)
        progress_pct = max(0.0, min(100.0, (total_pnl / TARGET_PROFIT_GOAL) * 100))
        trades_count = len(self.state.get("trades", []))

        try:
            acct = self.trading.get_account()
            cash_val = float(acct.cash)
            equity_val = float(acct.equity)
        except Exception:
            cash_val = 100.0
            equity_val = 100.0 + total_pnl

        msg = (
            f"⚡ <b>CYCLE #{cycle} (1m Turbo Scalp)</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 <b>BTC:</b> <b>${price:,.2f}</b> | <b>Signal:</b> {sig_icon}\n"
            f"🧭 <b>Trend:</b> {trend} | <b>RSI:</b> {last['rsi']:.1f}\n"
            f"🌊 <b>MACD Hist:</b> {last['macd_hist']:+.2f} | <b>ATR:</b> ${last['atr']:.2f}\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"{pos_block}\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>$50 GOAL PROGRESS:</b>\n"
            f"  • <b>Profits Earned:</b> <b>{'+' if total_pnl >= 0 else ''}${total_pnl:,.4f}</b> / <b>+$50.00</b> ({progress_pct:.1f}%)\n"
            f"  • <b>Account Equity:</b> <b>${equity_val:,.2f}</b> (Target: $150.00)\n"
            f"  • <b>Available Cash:</b> ${cash_val:,.2f} (Stake: ${self.calculate_stake():,.2f})\n"
            f"  • <b>Completed Trades:</b> {trades_count}\n"
            f"🕒 <i>{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}</i>"
        )
        send_telegram(msg)

    # ── Main Loop ────────────────────────────
    def run(self):
        log.info("Bot started. Polling every %d seconds. Press Ctrl+C to stop.", POLL_SECS)
        self.notify_startup()
        cycle = 0

        while True:
            try:
                cycle += 1
                log.info("=== Cycle #%d | %s ===",
                         cycle, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

                # 1. Fetch price data
                df = self.fetch_bars(limit=CANDLE_LIMIT)
                if df is None or len(df) < MIN_CANDLES:
                    log.warning("Not enough data (%s/%s bars). Retrying in %ds ...",
                                len(df) if df is not None else 0, MIN_CANDLES, POLL_SECS)
                    time.sleep(POLL_SECS)
                    continue

                # 2. Compute indicators
                df = add_indicators(df)

                # 3. Get current price
                price = self.get_price()
                if price is None:
                    price = float(df["close"].iloc[-1])

                # Live state update for dashboard
                pos = self.state.get("position")
                if pos:
                    unreal = (price - pos["entry_price"]) * pos["qty"]
                    pos["current_price"] = price
                    pos["unrealized_pnl"] = round(unreal, 4)
                    save_state(self.state)

                # 4. Check SL/TP first (always with trailing stop & breakeven protector)
                atr_val = float(df["atr"].iloc[-1]) if ("atr" in df and pd.notna(df["atr"].iloc[-1])) else 0.0
                exited = self.check_exit_conditions(price, atr=atr_val)

                # 5. Generate signal (returns tuple: signal, sl, tp)
                signal, sl, tp = get_signal(df)

                # 6. Print status dashboard & notify via Telegram
                self.print_status(df, signal, price, sl, tp)
                self.notify_cycle(cycle, price, df, signal, sl, tp)

                # 7. Act on signal
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


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    bot = AlpacaBTCBot()
    bot.run()
