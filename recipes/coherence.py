"""Coherence recipes using calibrated library pulses."""

from typing import Any

from zcu_tools.mcp.measure.recipe_context import RecipeContext


def t1(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    """Run one calibrated T1 delay sweep, save raw data and Primary analysis."""
    raise NotImplementedError("T1 calibration validation is not implemented")
