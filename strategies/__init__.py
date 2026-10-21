"""Strategy registry for EveryCryptoBot."""
from .base import BaseStrategy
from .scalp_macd_rsi import ScalpMacdRsi
from .momentum_ema_ribbon import MomentumEmaRibbon
from .vwap_breakout import VwapBreakout

STRATEGY_REGISTRY = {
    "ScalpMacdRsi": ScalpMacdRsi,
    "MomentumEmaRibbon": MomentumEmaRibbon,
    "VwapBreakout": VwapBreakout,
}

__all__ = ["BaseStrategy", "ScalpMacdRsi", "MomentumEmaRibbon", "VwapBreakout", "STRATEGY_REGISTRY"]
