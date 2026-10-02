"""Single-Run lookback recipe with finite inputs; GUI owns cfg defaults."""

from typing import Any

from zcu_tools.mcp.measure.recipe_context import RecipeContext


def lookback(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    """Prepare one lookback tab, save raw data, then analyze without accepting."""
    raise NotImplementedError("Lookback preparation is not implemented")
