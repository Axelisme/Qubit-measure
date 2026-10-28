"""Lab definitions consumed by the dispersive analysis GUI entry."""

from collections.abc import Callable
from typing import Protocol

from zcu_tools.resources.entry.registry import ComponentRegistry


class DispersiveGuiLab(Protocol):
    """Component registration supplied by a lab or synthetic lab description.

    Registration installs models and roles without acquiring hardware. Failures
    propagate from behavior construction, before any entry is opened.
    """

    @property
    def register_components(self) -> Callable[[ComponentRegistry], None]:
        """Install component models/roles into the process component registry."""
        ...
