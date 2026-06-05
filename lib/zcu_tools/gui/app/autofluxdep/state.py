"""State SSOT for autofluxdep-gui (skeleton).

Holds the workflow definition the user edits — the ordered Node instances, the
flux sweep, and the project handle — plus run results once Phase B/C add
execution. Mirrors the fluxdep/dispersive State pattern (frozen project info +
mutable working set guarded by a VersionTable), but the domain content is the
Node graph, not spectra.

This is a skeleton: only the workflow-definition fields exist. Run progress,
per-point/per-Node results, and predictor state are Phase B/C additions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Optional

from zcu_tools.gui.app.autofluxdep.nodes.spec import NodeInstance
from zcu_tools.gui.version_table import VersionTable

# VersionTable resource keys (optimistic concurrency, shared mechanism).
WORKFLOW_VERSION_KEY = "workflow"
FLUX_VERSION_KEY = "flux"
SETUP_VERSION_KEY = "setup"


@dataclass(frozen=True)
class ProjectInfo:
    """Project handle — chip/qubit identity + where params.json lives.

    The predictor (FluxoniumPredictor) is loaded from ``params_path`` in
    Phase B; here we only carry the locations.
    """

    chip_name: str
    qub_name: str
    result_dir: str
    params_path: str


@dataclass
class SetupResources:
    """The run prerequisites built by the Setup step (held as live objects).

    In the prototype these are fakes; Phase B binds the real soc/soccfg
    (make_soc_proxy), ml (ModuleLibrary), predictor (FluxoniumPredictor), and md
    (metadata). device is a global singleton (GlobalDeviceManager), not held
    here — State only records that setup completed via ``AutoFluxDepState
    .has_setup``. These objects stay owned by the main thread; the run worker
    uses them by reference.
    """

    soc: Any = None
    soccfg: Any = None
    ml: Any = None
    predictor: Any = None
    md: Any = None


@dataclass
class AutoFluxDepState:
    """Mutable working set: the workflow the user is assembling + run resources."""

    project: Optional[ProjectInfo] = None
    nodes: list[NodeInstance] = field(default_factory=list)
    flux_values: list[float] = field(default_factory=list)
    resources: Optional[SetupResources] = None
    version: VersionTable = field(default_factory=VersionTable)

    @property
    def has_setup(self) -> bool:
        return self.resources is not None

    def node_names(self) -> list[str]:
        return [n.name for n in self.nodes]
