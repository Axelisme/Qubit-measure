"""Public six-fluxonium material parser and YAML loader."""

from pathlib import Path

from .schema import SixFluxoniumSettings

__all__ = ["SixFluxoniumSettings", "load_six_fluxonium", "parse_six_fluxonium"]


def parse_six_fluxonium(document: object) -> SixFluxoniumSettings:
    """Validate decoded six-fluxonium v1 material, raising ValueError if invalid."""
    raise NotImplementedError("six-fluxonium material validation is not implemented")


def load_six_fluxonium(path: str | Path) -> SixFluxoniumSettings:
    """Read UTF-8 physical YAML at path (relative to cwd), then validate it."""
    raise NotImplementedError("six-fluxonium YAML loading is not implemented")
