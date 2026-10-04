"""Explicit user recipe declarations supplied only by composition roots."""

from zcu_tools.mcp.measure.recipe import RecipeDefinition

from .lookback import DEFINITION as LOOKBACK

RECIPES: tuple[RecipeDefinition, ...] = (LOOKBACK,)
