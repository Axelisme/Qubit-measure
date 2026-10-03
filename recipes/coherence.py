"""Coherence recipes using calibrated library pulses."""

from typing import Any

from zcu_tools.mcp.measure.recipe_context import MissingParameter, RecipeContext


def t1(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    """Run one calibrated T1 delay sweep, save raw data and Primary analysis."""
    sources = ctx.rpc("context.snapshot", {})
    publication = ctx.prepare_tab("twotone/t1", arguments.get("reuse_tab_id"))
    pulse = publication["tree"]["children"]["modules"]["children"]["pi_pulse"]
    if pulse.get("ref") not in sources["ml"]["modules"]:
        ctx.needs_parameters(
            [MissingParameter("pi_ref", "Provide a calibrated library pi pulse")]
        )
        return
    raise NotImplementedError("T1 calibrated Run is not implemented")
