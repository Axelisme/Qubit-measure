"""Single-run drive recipes using GUI-owned configuration sources."""

from math import isfinite
from typing import Any, TypeGuard

from zcu_tools.mcp.measure.recipe_context import MissingParameter, RecipeContext


def _finite(value: object) -> TypeGuard[int | float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(value)
    )


def twotone_spectrum(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    """Report missing sources; successful execution is not yet implemented."""
    sources = ctx.rpc("context.snapshot", {})
    ctx.prepare_tab("twotone/freq", arguments.get("reuse_tab_id"))
    missing = []
    if arguments.get("center_mhz") is None and not _finite(sources["md"].get("q_f")):
        missing.append(MissingParameter("center_mhz", "No finite q_f calibration"))
    width = sources["md"].get("qf_w")
    if arguments.get("span_mhz") is None and (not _finite(width) or width <= 0):
        missing.append(MissingParameter("span_mhz", "No calibrated qubit linewidth"))
    if arguments.get("readout_ref") is None and not _finite(sources["md"].get("r_f")):
        missing.append(
            MissingParameter("readout_ref", "Provide readout_ref or calibrated r_f")
        )
    if missing:
        ctx.needs_parameters(missing)
        return
    raise NotImplementedError("Two-tone execution is not yet implemented")
