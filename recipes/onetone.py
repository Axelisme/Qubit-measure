"""Single-run onetone recipes using GUI-owned calibration and defaults."""

from typing import Any

from zcu_tools.mcp.measure.recipe_context import RecipeContext


def onetone_spectrum(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    """Prepare a calibrated spectrum, run once and save raw and analysis."""
    raise NotImplementedError("Onetone frequency preparation is not implemented")
