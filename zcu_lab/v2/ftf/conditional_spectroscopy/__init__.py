"""Core-only two-line conditional spectroscopy; GUI attachment is explicit."""

from .core import (
    SpectroscopyCfg,
    SpectroscopyExp,
    SpectroscopyModules,
    SpectroscopyResult,
)

__all__ = [
    "SpectroscopyCfg",
    "SpectroscopyExp",
    "SpectroscopyModules",
    "SpectroscopyResult",
]
