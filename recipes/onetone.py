"""Single-run onetone recipes using GUI-owned calibration and defaults."""

from math import isfinite
from typing import Any

from zcu_tools.mcp.measure.recipe_context import MissingParameter, RecipeContext


def _finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(value)
    )


def onetone_spectrum(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    """Prepare a calibrated spectrum, run once and save raw and analysis."""
    sources = ctx.rpc("context.snapshot", {})
    ctx.prepare_tab("onetone/freq", arguments.get("reuse_tab_id"))
    missing = []
    if arguments.get("center_mhz") is None and not _finite(sources["md"].get("r_f")):
        missing.append(MissingParameter("center_mhz", "No finite r_f calibration"))
    width = sources["md"].get("rf_w")
    if arguments.get("span_mhz") is None and (not _finite(width) or width <= 0):
        missing.append(MissingParameter("span_mhz", "No positive rf_w calibration"))
    if missing:
        ctx.needs_parameters(missing)
        return
    raise NotImplementedError("Onetone Run preparation is not implemented")
