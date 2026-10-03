"""Single-shot GE calibration recipe."""

from typing import Any

from zcu_tools.mcp.measure.recipe_context import RecipeContext


def singleshot_ge(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    raise NotImplementedError("GE recipe is not implemented")
