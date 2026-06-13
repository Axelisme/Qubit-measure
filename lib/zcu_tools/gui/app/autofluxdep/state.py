"""State SSOT for autofluxdep-gui.

Holds the workflow definition the user edits — the ordered Node placements, the
flux sweep, and the project handle — plus the Setup resources (soc / flux device
/ predictor) and the per-Node run Results. Mirrors the fluxdep/dispersive State
pattern (frozen project info + mutable working set guarded by a VersionTable),
but the domain content is the Node graph, not spectra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zcu_tools.gui.app.autofluxdep.nodes.builder import PlacedNode
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import ExpContext

if TYPE_CHECKING:
    from zcu_tools.gui.app.autofluxdep.tools import Predictor

# VersionTable resource keys (optimistic concurrency, shared mechanism). The
# table itself is inherited from SessionState (one shared table, decision 6);
# these workflow keys bump the same table as the inherited session keys.
WORKFLOW_VERSION_KEY = "workflow"
FLUX_VERSION_KEY = "flux"


@dataclass(frozen=True)
class ProjectInfo:
    """Project handle — chip/qubit identity + where params.json lives.

    ``params_path`` is where a real FluxoniumPredictor would be loaded from when
    predictor loading is wired (the run uses a SimplePredictor stand-in).
    """

    chip_name: str
    qub_name: str
    result_dir: str
    params_path: str


class AutoFluxDepState(SessionState):
    """Mutable working set: the workflow the user is assembling + run resources.

    Extends ``SessionState`` (the active ``ExpContext`` + device set + startup
    prefs + the shared ``VersionTable``) with autofluxdep's experiment slice: the
    ordered Node placements, the flux sweep, the (transitional) Setup resources,
    and the per-Node run Results. Workflow version keys bump the same shared table
    as the inherited session keys (decision 6).

    ``run_results`` maps a provider name → its sweep-lived Result (the
    accumulated domain data — read by the Plotter, by saving, by an Info dialog
    — so it goes in State even though it is not serialisable). The main thread
    builds the empty Result containers at Run start; the worker fills their rows
    in place. Filling pre-allocated numpy rows is NOT a State semantic write (no
    key add/remove, no version bump), so it does not violate the "State writes
    only on the main thread" invariant. Cleared/rebuilt at each Run start.

    The soc / soccfg / predictor the run needs live in the inherited
    ``exp_context`` (the session SSOT, written by Setup / the session services);
    ``run_predictor`` holds the adaptive predictor the current run was built with.
    """

    def __init__(self, ctx: ExpContext, project: ProjectInfo | None = None) -> None:
        super().__init__(ctx)
        self.project: ProjectInfo | None = project
        self.nodes: list[PlacedNode] = []
        self.flux_values: list[float] = []
        # Which connected device the flux sweep is applied through (its unit
        # labels the flux axis; recorded for the run cfg's flux ``dev`` entry).
        # None = unset (the flux values are then bare numbers).
        self.flux_device_name: str | None = None
        self.run_results: dict[str, Any] = {}
        # The adaptive predictor the current/last run was built with (made per-run
        # from ``exp_context.predictor`` in ``Controller._build_tools``). Run-lived
        # and non-serialisable — like ``run_results`` — so an Info dialog / a test
        # can inspect the predictor the run calibrated.
        self.run_predictor: Predictor | None = None

    @property
    def has_setup(self) -> bool:
        """Whether a SoC is connected — the run prerequisite. Reads the active
        ``exp_context`` (the session SSOT), not a separate resources bundle."""
        return self.exp_context.has_soc()

    def node_names(self) -> list[str]:
        return [n.name for n in self.nodes]
