"""Single-shot GE calibration recipe."""

from typing import Any

from zcu_tools.mcp.measure.recipe_context import MissingParameter, RecipeContext

from .cfg_sources import cfg_node


def singleshot_ge(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    sources = ctx.rpc("context.snapshot", {})
    publication = ctx.prepare_tab("singleshot/ge", arguments.get("reuse_tab_id"))
    pulse = cfg_node(publication, "modules", "probe_pulse")
    if pulse.get("ref") not in sources["ml"]["modules"]:
        ctx.needs_parameters(
            [MissingParameter("pi_ref", "Provide a calibrated library probe_pulse")]
        )
        return
    raise NotImplementedError("GE acquisition is not implemented")
