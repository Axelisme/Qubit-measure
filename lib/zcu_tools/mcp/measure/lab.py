"""Lab declarations consumed by the measure MCP entry point."""

from collections.abc import Callable, Sequence
from typing import Protocol

from zcu_tools.mcp.measure.recipe_api import RecipeDeclaration
from zcu_tools.resources.entry.registry import ComponentRegistry


class MeasureMcpLab(Protocol):
    """Supply component registration and async recipes without acquiring hardware.

    Lab descriptions and synthetic test descriptions implement this interface.
    Registration failures propagate before the stdio server starts.
    """

    @property
    def register_components(self) -> Callable[[ComponentRegistry], None]:
        """Install component models, roles and schema policy in the process registry."""
        ...

    @property
    def recipes(self) -> Sequence[RecipeDeclaration]:
        """Return async tool declarations; an empty sequence exposes no recipes."""
        ...
