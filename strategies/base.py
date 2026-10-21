"""Abstract base class for all trading strategies."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class SignalResult:
    signal: str          # "BUY", "SELL", "HOLD"
    stop_loss: float     # price for SL
    take_profit: float   # price for TP
    confidence: float    # 0.0 – 1.0 (for logging)
    reason: str          # Human-readable why


class BaseStrategy(ABC):
    """All strategies must subclass this."""

    NAME: str = "BaseStrategy"
    TIMEFRAME_MINUTES: int = 5         # default bar size
    MIN_CANDLES: int = 50

    def __init__(self, symbol: str, params: Optional[dict] = None):
        self.symbol = symbol
        self.params = params or {}

    @abstractmethod
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to df and return it."""

    @abstractmethod
    def get_signal(self, df: pd.DataFrame, current_price: float) -> SignalResult:
        """Evaluate latest bar and return a SignalResult."""

    def requires_bars(self) -> int:
        return self.MIN_CANDLES

    def timeframe_minutes(self) -> int:
        return self.TIMEFRAME_MINUTES
