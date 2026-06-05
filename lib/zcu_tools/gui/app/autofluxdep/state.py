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

from typing_extensions import Optional

from zcu_tools.gui.app.autofluxdep.nodes.spec import NodeInstance
from zcu_tools.gui.version_table import VersionTable

# VersionTable resource keys (optimistic concurrency, shared mechanism).
WORKFLOW_VERSION_KEY = "workflow"
FLUX_VERSION_KEY = "flux"


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
class AutoFluxDepState:
    """Mutable working set: the workflow the user is assembling."""

    project: Optional[ProjectInfo] = None
    nodes: list[NodeInstance] = field(default_factory=list)
    flux_values: list[float] = field(default_factory=list)
    version: VersionTable = field(default_factory=VersionTable)

    def node_names(self) -> list[str]:
        return [n.name for n in self.nodes]
