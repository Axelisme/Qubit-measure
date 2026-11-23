"""Core-only two-line conditional spectroscopy; GUI attachment is explicit."""

from .core import SpectroscopyCfg, SpectroscopyExp, SpectroscopyResult
from .core_heralded import (
    HeraldedSpectroscopyCfg,
    HeraldedSpectroscopyExp,
    HeraldedSpectroscopyResult,
)

__all__ = [
    "HeraldedSpectroscopyCfg",
    "HeraldedSpectroscopyExp",
    "HeraldedSpectroscopyResult",
    "SpectroscopyCfg",
    "SpectroscopyExp",
    "SpectroscopyResult",
]
