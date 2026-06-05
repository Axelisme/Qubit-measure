"""Controller façade for autofluxdep-gui (skeleton).

State + EventBus coordinator, mirroring fluxdep/dispersive. Skeleton scope:
workflow-definition commands (add/remove/reorder Nodes, set flux) and a dry-run
that exercises the orchestrator on fake data so the dependency model is
demonstrable without hardware. Phase B adds the real hardware-driven run.
"""

from __future__ import annotations

from typing_extensions import Any, Optional

from zcu_tools.gui.app.autofluxdep.derivation import DerivationService
from zcu_tools.gui.app.autofluxdep.event_bus import Event, EventBus, EventType
from zcu_tools.gui.app.autofluxdep.nodes.spec import NodeInstance, NodeSpec
from zcu_tools.gui.app.autofluxdep.orchestrator import (
    InfoStore,
    ModuleSource,
    Orchestrator,
    PrePoint,
    RunNode,
)
from zcu_tools.gui.app.autofluxdep.state import (
    FLUX_VERSION_KEY,
    WORKFLOW_VERSION_KEY,
    AutoFluxDepState,
)
from zcu_tools.gui.app.autofluxdep.tools import Tools


class Controller:
    def __init__(self, state: AutoFluxDepState, bus: EventBus) -> None:
        self._state = state
        self._bus = bus

    # --- workflow definition (the only writes the user makes) ---

    def add_node(self, spec: NodeSpec, **params: Any) -> NodeInstance:
        node = NodeInstance(spec=spec, params=dict(params))
        self._state.nodes.append(node)
        self._state.version.bump(WORKFLOW_VERSION_KEY)
        self._bus.emit(Event(EventType.WORKFLOW_CHANGED, node.name))
        return node

    def remove_node(self, name: str) -> None:
        self._state.nodes = [n for n in self._state.nodes if n.name != name]
        self._state.version.bump(WORKFLOW_VERSION_KEY)
        self._bus.emit(Event(EventType.WORKFLOW_CHANGED, name))

    def set_flux_values(self, values: list[float]) -> None:
        self._state.flux_values = list(values)
        self._state.version.bump(FLUX_VERSION_KEY)
        self._bus.emit(Event(EventType.FLUX_CHANGED, len(values)))

    # --- dry run (skeleton: fake per-Node execution, no hardware) ---

    def dry_run(
        self,
        run_node: RunNode,
        pre_point: Optional[PrePoint] = None,
        tools: Optional[Tools] = None,
        ml: Optional[ModuleSource] = None,
        derivations: Optional[list[DerivationService]] = None,
    ) -> InfoStore:
        """Exercise dependency resolution on fake data.

        ``run_node`` is the injected fake per-Node execution; ``pre_point``
        seeds the per-point values Nodes depend on (e.g. predict_freq), the way
        the real orchestrator's before-each would. ``tools`` (the predictor)
        defaults to a fresh skeleton ``Tools``; ``ml`` is the module-library
        fallback for module deps; smoothing is auto-built from the Nodes'
        declarations, ``derivations`` are any extra producers. Returns the final
        InfoStore. Nodes run in their list order (no topo sort).
        """
        orch = Orchestrator(
            nodes=list(self._state.nodes),
            run_node=run_node,
            tools=tools or Tools(),
            ml=ml,
            derivations=derivations or [],
        )
        return orch.run(self._state.flux_values, pre_point=pre_point)
