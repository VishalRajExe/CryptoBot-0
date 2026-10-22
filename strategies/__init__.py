"""Strategy registry for EveryCryptoBot."""
from .base import BaseStrategy
from .scalp_macd_rsi import ScalpMacdRsi
from .momentum_ema_ribbon import MomentumEmaRibbon
from .vwap_breakout import VwapBreakout
from .binh_cluc import BinHCluc
from .supertrend import Supertrend
from .smooth_scalp import SmoothScalp
from .rsi_bb_squeeze import RsiBbSqueeze
from .ema_rsi_scalper import EmaRsiScalper

STRATEGY_REGISTRY = {
    "ScalpMacdRsi": ScalpMacdRsi,
    "MomentumEmaRibbon": MomentumEmaRibbon,
    "VwapBreakout": VwapBreakout,
    "BinHCluc": BinHCluc,
    "Supertrend": Supertrend,
    "SmoothScalp": SmoothScalp,
    "RsiBbSqueeze": RsiBbSqueeze,
    "EmaRsiScalper": EmaRsiScalper,
}

__all__ = [
    "BaseStrategy",
    "ScalpMacdRsi",
    "MomentumEmaRibbon",
    "VwapBreakout",
    "BinHCluc",
    "Supertrend",
    "SmoothScalp",
    "RsiBbSqueeze",
    "EmaRsiScalper",
    "STRATEGY_REGISTRY"
]
