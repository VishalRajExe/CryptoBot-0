"""
EveryCryptoBot — SQLite Trade Memory & Persistence Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Provides ACID-compliant persistent memory for all trading bots:
- Survives crashes, reboots, and terminal restarts.
- Auto-reconciles with Alpaca API to adopt existing open positions on reboot.
- Stores full trade telemetry (RSI, MACD, ATR, volume at entry) for AI learning.
- Tracks consecutive losses, rolling win rates, and cooling periods.
"""

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading_memory.db")
_db_lock = threading.RLock()
log = logging.getLogger("ECB.Memory")


class TradeMemory:
    """ACID-compliant SQLite trade storage and adaptive learning memory."""

    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high-concurrency multi-threaded access
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        return conn

    def _init_db(self):
        with _db_lock, self._get_conn() as conn:
            conn.executescript("""
                -- Active Positions (synced with Alpaca)
                CREATE TABLE IF NOT EXISTS active_positions (
                    symbol TEXT PRIMARY KEY,
                    order_id TEXT,
                    qty REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    strategy TEXT,
                    features_json TEXT
                );

                -- Completed Trades Journal (Persistent History)
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    side TEXT NOT NULL DEFAULT 'BUY',
                    qty REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    exit_reason TEXT NOT NULL,
                    features_json TEXT
                );

                -- Adaptive Learning & Health Metrics per Symbol
                CREATE TABLE IF NOT EXISTS symbol_metrics (
                    symbol TEXT PRIMARY KEY,
                    consecutive_losses INTEGER DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0.0,
                    last_loss_time TEXT,
                    cooldown_until TEXT,
                    risk_multiplier REAL DEFAULT 1.0,
                    updated_at TEXT
                );

                -- Machine Learning Training Dataset (Persisted across reboots)
                CREATE TABLE IF NOT EXISTS ml_training_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    outcome INTEGER NOT NULL, -- 1=Win, 0=Loss
                    pnl REAL NOT NULL,
                    pnl_pct REAL NOT NULL
                );

                -- Mistake Memory & Loss Regimes (Catalog of failed trade setups)
                CREATE TABLE IF NOT EXISTS mistake_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    loss_amount REAL NOT NULL,
                    loss_pct REAL NOT NULL,
                    exit_reason TEXT NOT NULL
                );
            """)

    # ─────────────────────────────────────────────────────────────────────────
    #  POSITION PERSISTENCE & RECONCILIATION
    # ─────────────────────────────────────────────────────────────────────────
    def save_position(self, symbol: str, order_id: str, qty: float,
                      entry_price: float, stop_loss: float, take_profit: float,
                      strategy: str = "", features: Optional[dict] = None) -> None:
        """Record a new open position in SQLite."""
        with _db_lock, self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO active_positions 
                (symbol, order_id, qty, entry_price, stop_loss, take_profit, entry_time, strategy, features_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, order_id, qty, entry_price, stop_loss, take_profit,
                datetime.now(timezone.utc).isoformat(), strategy,
                json.dumps(features or {})
            ))
            conn.commit()
        log.info("Saved open position to SQLite: %s (Qty: %.6f @ $%.4f)", symbol, qty, entry_price)

    def get_position(self, symbol: str) -> Optional[dict]:
        """Retrieve active position for symbol."""
        with _db_lock, self._get_conn() as conn:
            row = conn.execute("SELECT * FROM active_positions WHERE symbol = ?", (symbol,)).fetchone()
            if row:
                return dict(row)
        return None

    def remove_position(self, symbol: str) -> None:
        """Clear active position upon exit."""
        with _db_lock, self._get_conn() as conn:
            conn.execute("DELETE FROM active_positions WHERE symbol = ?", (symbol,))
            conn.commit()

    def update_stop_loss(self, symbol: str, stop_loss: float) -> None:
        """Update stop loss dynamically for breakeven or trailing stop."""
        with _db_lock, self._get_conn() as conn:
            conn.execute("UPDATE active_positions SET stop_loss = ? WHERE symbol = ?", (stop_loss, symbol))
            conn.commit()

    def get_active_positions(self) -> List[dict]:
        """Returns all currently open positions across all coins."""
        with _db_lock, self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM active_positions").fetchall()
            return [dict(r) for r in rows]

    def reconcile_with_alpaca(self, trading_client, symbol: str, fallback_sl_pct: float = 0.015,
                              fallback_tp_pct: float = 0.03) -> Optional[dict]:
        """
        Ensures state survives restarts:
        Checks Alpaca ground truth. If Alpaca has an open position but SQLite was wiped or restarting,
        this restores it automatically with safe default stops.
        """
        clean_sym = symbol.replace("/", "")
        alpaca_pos = None
        try:
            alpaca_pos = trading_client.get_open_position(clean_sym)
        except Exception:
            alpaca_pos = None

        db_pos = self.get_position(symbol)

        if alpaca_pos and float(alpaca_pos.qty) > 0:
            qty = float(alpaca_pos.qty)
            entry_price = float(alpaca_pos.avg_entry_price)

            if db_pos:
                # Match found, update qty if partial fill
                return db_pos
            else:
                # Bot restarted while in a trade! Reconstruct position in memory
                sl = round(entry_price * (1 - fallback_sl_pct), 4)
                tp = round(entry_price * (1 + fallback_tp_pct), 4)

                # Check if alpaca_state.json has calibrated SL/TP for BTC/USD
                if symbol == "BTC/USD":
                    try:
                        state_path = os.path.join(os.path.dirname(__file__), "alpaca_state.json")
                        if os.path.exists(state_path):
                            with open(state_path, "r", encoding="utf-8") as f:
                                btc_state = json.load(f)
                            pos_info = btc_state.get("position")
                            if pos_info and pos_info.get("stop_loss"):
                                sl = float(pos_info["stop_loss"])
                                tp = float(pos_info.get("take_profit", tp))
                    except Exception:
                        pass

                log.warning("🔄 RECOVERED ORPHAN POSITION on %s: Qty=%.6f @ $%.4f. Setting SL=$%.4f TP=$%.4f",
                            symbol, qty, entry_price, sl, tp)
                self.save_position(
                    symbol=symbol,
                    order_id="recovered_from_alpaca",
                    qty=qty,
                    entry_price=entry_price,
                    stop_loss=sl,
                    take_profit=tp,
                    strategy="ScalpMacdRsi" if symbol == "BTC/USD" else "RecoveredPosition"
                )
                return self.get_position(symbol)
        else:
            # Alpaca says flat: clear stale position in DB if any
            if db_pos:
                log.info("Position on %s closed externally or filled while bot was offline. Clearing DB.", symbol)
                self.remove_position(symbol)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    #  TRADE JOURNALING & LEARNING FEEDBACK
    # ─────────────────────────────────────────────────────────────────────────
    def record_closed_trade(self, symbol: str, strategy: str, qty: float,
                            entry_price: float, exit_price: float, entry_time: str,
                            reason: str, features: Optional[dict] = None) -> dict:
        """
        Saves completed trade, updates adaptive learning metrics,
        and triggers consecutive-loss cooling if necessary.
        """
        pnl = (exit_price - entry_price) * qty
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price else 0.0
        exit_time = datetime.now(timezone.utc).isoformat()
        is_win = pnl >= 0

        with _db_lock, self._get_conn() as conn:
            # 1. Insert into history
            conn.execute("""
                INSERT INTO trade_history 
                (symbol, strategy, side, qty, entry_price, exit_price, entry_time, exit_time, pnl, pnl_pct, exit_reason, features_json)
                VALUES (?, ?, 'BUY', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, strategy, qty, entry_price, exit_price, entry_time, exit_time,
                round(pnl, 6), round(pnl_pct, 4), reason, json.dumps(features or {})
            ))

            # 2. Update active positions
            conn.execute("DELETE FROM active_positions WHERE symbol = ?", (symbol,))

            # 3. Update Adaptive Metrics & Learning
            row = conn.execute("SELECT * FROM symbol_metrics WHERE symbol = ?", (symbol,)).fetchone()
            consec_losses = 0
            tot_trades = 0
            wins = 0
            losses = 0
            total_pnl = 0.0
            risk_mult = 1.0
            cooldown_until = None

            if row:
                consec_losses = row["consecutive_losses"]
                tot_trades = row["total_trades"]
                wins = row["wins"]
                losses = row["losses"]
                total_pnl = row["total_pnl"]
                risk_mult = row["risk_multiplier"]

            tot_trades += 1
            total_pnl += pnl

            if is_win:
                wins += 1
                consec_losses = 0
                # Reward: restore standard risk multiplier
                risk_mult = min(1.2, risk_mult + 0.05)
            else:
                losses += 1
                consec_losses += 1
                # Penalty: scale down risk multiplier after losses
                risk_mult = max(0.5, risk_mult - 0.20)

                # If 2 consecutive losses occur, trigger a 20-minute cooldown
                if consec_losses >= 2:
                    cooldown_dt = datetime.now(timezone.utc).timestamp() + (20 * 60)
                    cooldown_until = datetime.fromtimestamp(cooldown_dt, timezone.utc).isoformat()
                    log.warning("🧠 LEARNING TRIGGER: %s incurred %d consecutive losses. Cooling down until %s.",
                                symbol, consec_losses, cooldown_until)

            conn.execute("""
                INSERT OR REPLACE INTO symbol_metrics
                (symbol, consecutive_losses, total_trades, wins, losses, total_pnl, last_loss_time, cooldown_until, risk_multiplier, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, consec_losses, tot_trades, wins, losses, round(total_pnl, 6),
                exit_time if not is_win else (row["last_loss_time"] if row else None),
                cooldown_until, round(risk_mult, 2), exit_time
            ))

            # 4. Record sample into persistent ML training dataset
            conn.execute("""
                INSERT INTO ml_training_samples 
                (symbol, strategy, timestamp, features_json, outcome, pnl, pnl_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, strategy, exit_time, json.dumps(features or {}),
                1 if is_win else 0, round(pnl, 6), round(pnl_pct, 4)
            ))

            # 5. If loss, catalog into mistake_patterns memory
            if not is_win:
                conn.execute("""
                    INSERT INTO mistake_patterns
                    (symbol, strategy, timestamp, features_json, loss_amount, loss_pct, exit_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, strategy, exit_time, json.dumps(features or {}),
                    abs(round(pnl, 6)), abs(round(pnl_pct, 4)), reason
                ))
            conn.commit()

        return {
            "pnl": round(pnl, 4),
            "pnl_pct": round(pnl_pct, 2),
            "is_win": is_win,
            "consecutive_losses": consec_losses,
            "risk_multiplier": risk_mult,
            "cooldown_until": cooldown_until,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  ADAPTIVE LEARNING QUERIES
    # ─────────────────────────────────────────────────────────────────────────
    def is_in_cooldown(self, symbol: str) -> bool:
        """Checks if symbol is in cooling-off period after consecutive losses."""
        with _db_lock, self._get_conn() as conn:
            row = conn.execute("SELECT cooldown_until FROM symbol_metrics WHERE symbol = ?", (symbol,)).fetchone()
            if row and row["cooldown_until"]:
                try:
                    cd_time = datetime.fromisoformat(row["cooldown_until"])
                    if datetime.now(timezone.utc) < cd_time:
                        return True
                except Exception:
                    pass
        return False

    def get_risk_multiplier(self, symbol: str) -> float:
        """Returns adaptive stake multiplier (0.5x to 1.2x) based on learning history."""
        with _db_lock, self._get_conn() as conn:
            row = conn.execute("SELECT risk_multiplier FROM symbol_metrics WHERE symbol = ?", (symbol,)).fetchone()
            if row and row["risk_multiplier"]:
                return float(row["risk_multiplier"])
        return 1.0

    def get_learning_summary(self) -> Dict[str, Any]:
        """Provides full metrics across all symbols for the dashboard."""
        with _db_lock, self._get_conn() as conn:
            metrics_rows = conn.execute("SELECT * FROM symbol_metrics").fetchall()
            trades_count = conn.execute("SELECT COUNT(*) as c FROM trade_history").fetchone()["c"]
            active_count = conn.execute("SELECT COUNT(*) as c FROM active_positions").fetchone()["c"]

            summary = {
                "total_trades_stored": trades_count,
                "active_positions_stored": active_count,
                "symbols": {}
            }
            for r in metrics_rows:
                sym = r["symbol"]
                wr = round((r["wins"] / r["total_trades"] * 100), 1) if r["total_trades"] > 0 else 0.0
                is_cooling = False
                if r["cooldown_until"]:
                    try:
                        cd_time = datetime.fromisoformat(r["cooldown_until"])
                        if datetime.now(timezone.utc) < cd_time:
                            is_cooling = True
                    except Exception:
                        pass
                summary["symbols"][sym] = {
                    "total_trades": r["total_trades"],
                    "wins": r["wins"],
                    "losses": r["losses"],
                    "win_rate_pct": wr,
                    "total_pnl": r["total_pnl"],
                    "consecutive_losses": r["consecutive_losses"],
                    "risk_multiplier": r["risk_multiplier"],
                    "cooldown_until": r["cooldown_until"],
                    "cooling_active": is_cooling
                }
            return summary

    def get_recent_trades(self, limit: int = 50) -> List[dict]:
        """Returns recent completed trades from database."""
        with _db_lock, self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM trade_history 
                ORDER BY id DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_all_ml_samples(self) -> List[dict]:
        """Returns all persisted machine learning training samples."""
        with _db_lock, self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM ml_training_samples 
                ORDER BY id ASC
            """).fetchall()
            return [dict(r) for r in rows]

    def get_recent_mistakes(self, limit: int = 50) -> List[dict]:
        """Returns recent losing market regimes for mistake pattern avoidance."""
        with _db_lock, self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM mistake_patterns 
                ORDER BY id DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_mistake_summary(self) -> dict:
        """Returns statistics on learned mistakes and losing patterns."""
        with _db_lock, self._get_conn() as conn:
            total_mistakes = conn.execute("SELECT COUNT(*) as c FROM mistake_patterns").fetchone()["c"]
            recent = conn.execute("""
                SELECT symbol, strategy, loss_amount, loss_pct, exit_reason, timestamp 
                FROM mistake_patterns 
                ORDER BY id DESC LIMIT 5
            """).fetchall()
            return {
                "total_mistakes_remembered": total_mistakes,
                "recent_mistakes": [dict(r) for r in recent]
            }

    def get_symbol_total_pnl(self, symbol: str) -> float:
        """Returns total realized PnL for a given symbol from symbol_metrics."""
        with _db_lock, self._get_conn() as conn:
            row = conn.execute("SELECT total_pnl FROM symbol_metrics WHERE symbol = ?", (symbol,)).fetchone()
            return float(row["total_pnl"]) if row and row["total_pnl"] is not None else 0.0

    def get_total_realized_pnl(self) -> float:
        """Returns sum of all realized trade PnL across all altcoin bots."""
        with _db_lock, self._get_conn() as conn:
            row = conn.execute("SELECT SUM(pnl) as s FROM trade_history").fetchone()
            return float(row["s"]) if row and row["s"] is not None else 0.0

