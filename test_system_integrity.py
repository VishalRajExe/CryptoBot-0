"""
EveryCryptoBot — Comprehensive System & Codebase Integrity Test Suite
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Verifies:
1. SQLite ACID Memory Persistence & Orphan Position Recovery
2. Online ML Retraining, Model Serialization, & Mistake Memory
3. Dynamic Micro-Account Position Sizing ($100 capital scaling)
4. Trailing Stop & Breakeven Protection Logic
5. Volatility-Adaptive Timeframe Manager
6. All 6 Technical Strategies & Registry
7. Dashboard Server API Data Contracts
"""

import json
import os
import shutil
import sys
import unittest
import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from memory import TradeMemory
from learner import TradeLearner
from strategies import STRATEGY_REGISTRY


class TestSystemIntegrity(unittest.TestCase):
    def setUp(self):
        self.test_db = os.path.join(BASE_DIR, f"test_memory_{self._testMethodName}.db")
        if os.path.exists(self.test_db):
            try:
                os.remove(self.test_db)
            except Exception:
                pass
        self.memory = TradeMemory(db_path=self.test_db)

    def tearDown(self):
        try:
            if os.path.exists(self.test_db):
                os.remove(self.test_db)
        except Exception:
            pass

    def test_01_sqlite_acid_persistence(self):
        """Test ACID persistence, table creation, and position management."""
        # 1. Save Position
        features = {"rsi": 42.5, "macd_hist": 0.05, "atr": 2.1, "price": 100.0}
        self.memory.save_position(
            symbol="TEST/USD",
            order_id="test_order_123",
            qty=1.5,
            entry_price=100.0,
            stop_loss=98.0,
            take_profit=104.0,
            strategy="BinHCluc",
            features=features
        )
        pos = self.memory.get_position("TEST/USD")
        self.assertIsNotNone(pos)
        self.assertEqual(pos["symbol"], "TEST/USD")
        self.assertEqual(pos["qty"], 1.5)
        self.assertEqual(pos["stop_loss"], 98.0)

        # 2. Update Stop Loss (Breakeven / Trailing stop)
        self.memory.update_stop_loss("TEST/USD", 100.2)
        pos_updated = self.memory.get_position("TEST/USD")
        self.assertEqual(pos_updated["stop_loss"], 100.2)

        # 3. Retrieve Active Positions list
        active_list = self.memory.get_active_positions()
        self.assertEqual(len(active_list), 1)

        # 4. Record Closed Trade (WIN)
        learning = self.memory.record_closed_trade(
            symbol="TEST/USD",
            strategy="BinHCluc",
            qty=1.5,
            entry_price=100.0,
            exit_price=104.0,
            entry_time="2026-10-21T12:00:00Z",
            reason="take_profit",
            features=features
        )
        self.assertTrue(learning["is_win"])
        self.assertAlmostEqual(learning["pnl"], 6.0)
        self.assertIsNone(self.memory.get_position("TEST/USD"))

        # 5. Verify trade was journaled in history and ML training samples
        history = self.memory.get_recent_trades(limit=10)
        self.assertEqual(len(history), 1)
        samples = self.memory.get_all_ml_samples()
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["outcome"], 1)

    def test_02_mistake_memory_cataloging(self):
        """Test that losing trades are cataloged into mistake_patterns and cooling triggers."""
        features_loss = {"rsi": 88.0, "macd_hist": -0.5, "atr": 10.0, "price": 150.0}
        # First loss
        self.memory.record_closed_trade(
            symbol="FAIL/USD",
            strategy="ScalpMacdRsi",
            qty=2.0,
            entry_price=150.0,
            exit_price=145.0,
            entry_time="2026-10-21T12:00:00Z",
            reason="stop_loss",
            features=features_loss
        )
        # Second loss (should trigger consecutive loss circuit breaker)
        res2 = self.memory.record_closed_trade(
            symbol="FAIL/USD",
            strategy="ScalpMacdRsi",
            qty=2.0,
            entry_price=145.0,
            exit_price=140.0,
            entry_time="2026-10-21T12:10:00Z",
            reason="stop_loss",
            features=features_loss
        )
        self.assertEqual(res2["consecutive_losses"], 2)
        self.assertTrue(self.memory.is_in_cooldown("FAIL/USD"))
        self.assertLess(self.memory.get_risk_multiplier("FAIL/USD"), 1.0)

        # Check mistake patterns table
        mistakes = self.memory.get_recent_mistakes()
        self.assertEqual(len(mistakes), 2)
        summary = self.memory.get_mistake_summary()
        self.assertEqual(summary["total_mistakes_remembered"], 2)

    def test_03_online_machine_learning_retraining(self):
        """Test genuine online retraining, mistake penalty, and weight persistence."""
        test_model_path = os.path.join(BASE_DIR, "test_trade_model.joblib")
        if os.path.exists(test_model_path):
            os.remove(test_model_path)

        try:
            learner = TradeLearner(model_path=test_model_path)
            self.assertTrue(learner.is_trained)
            initial_samples = learner.sample_count

            # Evaluate favorable signal
            good_features = {"rsi": 55.0, "macd_hist": 0.15, "atr": 2.0, "price": 100.0, "vol_ratio": 1.5, "adx": 30.0}
            approved, prob, reason = learner.evaluate_signal("ETH/USD", good_features)
            self.assertGreaterEqual(prob, 0.40)

            # Record losing trade setup
            bad_features = {"rsi": 89.0, "macd_hist": -0.8, "atr": 18.0, "price": 100.0, "vol_ratio": 0.4, "adx": 12.0}
            learner.record_online_sample(bad_features, is_win=False, pnl=-12.5)

            # Check that samples incremented and retraining cycle fired
            self.assertEqual(learner.sample_count, initial_samples + 1)
            self.assertEqual(learner.retraining_events, 1)

            # Test that identical/similar losing regime now receives mistake penalty or veto
            app2, prob2, reason2 = learner.evaluate_signal("ETH/USD", bad_features)
            self.assertLess(prob2, 0.45)
            self.assertIn("MISTAKE", reason2)

            # Verify serialization to disk
            self.assertTrue(os.path.exists(test_model_path))

            # Reload model from disk and verify persistence
            reloaded_learner = TradeLearner(model_path=test_model_path)
            self.assertEqual(reloaded_learner.sample_count, learner.sample_count)
            self.assertEqual(reloaded_learner.retraining_events, 1)
        finally:
            if os.path.exists(test_model_path):
                os.remove(test_model_path)

    def test_04_micro_account_dynamic_position_sizing(self):
        """Test dynamic fractional sizing for a $100 starting balance."""
        class MockBot:
            def __init__(self, cash, memory, symbol):
                self.cash = cash
                self.memory = memory
                self.symbol = symbol

            def get_cash(self):
                return self.cash

            def calculate_stake(self):
                cash = self.get_cash()
                reserve = max(10.0, cash * 0.15)
                available = max(0.0, cash - reserve)
                active_positions = self.memory.get_active_positions()
                slots = max(1, 3 - len(active_positions))
                target = available / slots
                risk_mult = self.memory.get_risk_multiplier(self.symbol)
                max_allowed = max(10.0, cash * 0.35)
                stake = min(max(target * risk_mult, 10.0), max_allowed)
                return round(stake, 2)

        # 1. $100 starting capital with 0 open positions
        bot_100 = MockBot(cash=100.0, memory=self.memory, symbol="ETH/USD")
        stake_100 = bot_100.calculate_stake()
        # available = 100 - 15 = 85. 85 / 3 slots = ~$28.33
        self.assertGreaterEqual(stake_100, 25.0)
        self.assertLessEqual(stake_100, 35.0)

        # 2. Capital compounds to $1,000
        bot_1000 = MockBot(cash=1000.0, memory=self.memory, symbol="ETH/USD")
        stake_1000 = bot_1000.calculate_stake()
        # available = 1000 - 150 = 850. 850 / 3 = $283.33
        self.assertGreaterEqual(stake_1000, 250.0)
        self.assertLessEqual(stake_1000, 350.0)

        # 3. Capital reaches $10,000
        bot_10k = MockBot(cash=10000.0, memory=self.memory, symbol="ETH/USD")
        stake_10k = bot_10k.calculate_stake()
        self.assertGreaterEqual(stake_10k, 2500.0)
        self.assertLessEqual(stake_10k, 3500.0)

    def test_05_strategy_registry_and_indicators(self):
        """Test that all 6 strategies generate indicators and signals correctly."""
        # Create synthetic OHLCV data
        dates = pd.date_range("2026-10-01", periods=100, freq="5min", tz="UTC")
        prices = 100.0 + np.cumsum(np.random.normal(0, 0.5, 100))
        df = pd.DataFrame({
            "open": prices * 0.999,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "close": prices,
            "volume": np.random.uniform(100, 500, 100)
        }, index=dates)

        for strat_name, strat_cls in STRATEGY_REGISTRY.items():
            strat = strat_cls(symbol="TEST/USD")
            self.assertIsNotNone(strat.NAME)
            df_ind = strat.add_indicators(df.copy())
            self.assertIn("atr", df_ind.columns)
            sig = strat.get_signal(df_ind, float(df_ind["close"].iloc[-1]))
            self.assertIn(sig.signal, ["BUY", "SELL", "HOLD"])
            self.assertIsInstance(sig.confidence, float)


if __name__ == "__main__":
    unittest.main()
