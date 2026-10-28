"""Lab definitions consumed by the flux-dependence GUI entry."""

from collections.abc import Callable
from typing import Protocol

from zcu_tools.gui.app.analysis_project.flux import ComponentFluxProvider
from zcu_tools.resources.entry.registry import ComponentRegistry

from .export_plan import ExportPlanner


class FluxDepGuiLab(Protocol):
    """Component definitions and analysis policies from a lab or synthetic lab.

    Registration installs models and roles without acquiring hardware. Failures
    propagate from behavior construction, before any entry is opened.
    """

    @property
    def register_components(self) -> Callable[[ComponentRegistry], None]:
        """Install component models/roles into the process component registry."""
        ...

    @property
    def fluxdep_export_planner(self) -> ExportPlanner | None:
        """Plan detached setup writes; None leaves viewing enabled but refuses export."""
        ...

    @property
    def component_flux_context_provider(self) -> ComponentFluxProvider | None:
        """Project a component frame/device choices; None refuses context-dependent work."""
        ...
