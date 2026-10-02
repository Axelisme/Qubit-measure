"""Single-Run lookback recipe with finite inputs; GUI owns cfg defaults."""

from math import isfinite
from typing import Any

from zcu_tools.mcp.measure.recipe_context import MissingParameter, RecipeContext


def lookback(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    """Prepare one lookback tab, save raw data, then analyze without accepting."""
    reuse_tab_id = arguments.get("reuse_tab_id")
    if reuse_tab_id is not None and (
        not isinstance(reuse_tab_id, str) or not reuse_tab_id
    ):
        raise ValueError("reuse_tab_id must be a non-empty string or null")
    sources = ctx.rpc("context.snapshot", {})
    ctx.prepare_tab("lookback", reuse_tab_id)
    calibrated = sources["md"].get("r_f")
    if (
        arguments.get("frequency_mhz") is None
        and arguments.get("readout_ref") is None
        and (
            isinstance(calibrated, bool)
            or not isinstance(calibrated, (float, int))
            or not isfinite(calibrated)
        )
    ):
        ctx.needs_parameters(
            [
                MissingParameter(
                    "frequency_mhz", "No calibrated readout frequency is available"
                )
            ]
        )
        return
    raise NotImplementedError("Lookback measurement is not implemented")
