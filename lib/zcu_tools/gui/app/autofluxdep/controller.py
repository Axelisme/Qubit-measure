"""Controller façade for autofluxdep-gui.

State + EventBus coordinator, mirroring fluxdep/dispersive. Owns the workflow
definition commands (add/remove/reorder Nodes, set flux, set Node params), a
Setup that builds a MockSoc + FakeDevice flux board (the prototype's offline
resources), and a cancellable run that drives each Node through its build_cfg +
run_body (synthetic acquire + real fit) and emits run-lifecycle events. The
hardware-free ``dry_run`` remains for direct testing of the dependency model.
"""

from __future__ import annotations

from typing_extensions import Any, Optional

from zcu_tools.gui.app.autofluxdep.derivation import DerivationService
from zcu_tools.gui.app.autofluxdep.event_bus import Event, EventBus, EventType
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch
from zcu_tools.gui.app.autofluxdep.nodes.spec import NodeInstance, NodeSpec
from zcu_tools.gui.app.autofluxdep.orchestrator import (
    InfoStore,
    ModuleSource,
    Orchestrator,
    PrePoint,
    RunNode,
)
from zcu_tools.gui.app.autofluxdep.registry import create_instance
from zcu_tools.gui.app.autofluxdep.state import (
    FLUX_VERSION_KEY,
    SETUP_VERSION_KEY,
    WORKFLOW_VERSION_KEY,
    AutoFluxDepState,
    SetupResources,
)
from zcu_tools.gui.app.autofluxdep.tools import Tools


class Controller:
    def __init__(self, state: AutoFluxDepState, bus: EventBus) -> None:
        self._state = state
        self._bus = bus
        self._stop = False  # cooperative run-cancel flag
        self._running = False
        self._cur_idx = 0  # current flux index during a run (for NODE_STARTED)

    # --- read-only accessors for the UI ---

    @property
    def state(self) -> AutoFluxDepState:
        return self._state

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def is_running(self) -> bool:
        return self._running

    # --- workflow definition (the only writes the user makes) ---

    def add_node(self, spec: NodeSpec, **params: Any) -> NodeInstance:
        node = NodeInstance(spec=spec, params=dict(params))
        self._state.nodes.append(node)
        self._state.version.bump(WORKFLOW_VERSION_KEY)
        self._bus.emit(Event(EventType.WORKFLOW_CHANGED, node.name))
        return node

    def add_node_by_type(self, type_name: str) -> NodeInstance:
        """Add a fresh Node of ``type_name`` (from the registry) to the end."""
        node = create_instance(type_name)
        self._state.nodes.append(node)
        self._state.version.bump(WORKFLOW_VERSION_KEY)
        self._bus.emit(Event(EventType.WORKFLOW_CHANGED, node.name))
        return node

    def remove_node(self, name: str) -> None:
        self._state.nodes = [n for n in self._state.nodes if n.name != name]
        self._state.version.bump(WORKFLOW_VERSION_KEY)
        self._bus.emit(Event(EventType.WORKFLOW_CHANGED, name))

    def reorder(self, index: int, delta: int) -> int:
        """Move the Node at ``index`` by ``delta`` (±1). Returns the new index."""
        nodes = self._state.nodes
        new_index = index + delta
        if not (0 <= index < len(nodes) and 0 <= new_index < len(nodes)):
            return index  # out of range → no-op
        nodes[index], nodes[new_index] = nodes[new_index], nodes[index]
        self._state.version.bump(WORKFLOW_VERSION_KEY)
        self._bus.emit(Event(EventType.WORKFLOW_CHANGED, None))
        return new_index

    def set_node_params(self, index: int, params: dict[str, Any]) -> None:
        """Replace the tuned params of the Node at ``index``."""
        self._state.nodes[index].params = dict(params)
        self._state.version.bump(WORKFLOW_VERSION_KEY)
        self._bus.emit(Event(EventType.WORKFLOW_CHANGED, None))

    def set_flux_values(self, values: list[float]) -> None:
        self._state.flux_values = list(values)
        self._state.version.bump(FLUX_VERSION_KEY)
        self._bus.emit(Event(EventType.FLUX_CHANGED, len(values)))

    # --- setup (prototype: MockSoc + FakeDevice, no hardware) ---

    def setup(
        self,
        resources: Optional[SetupResources] = None,
        *,
        use_mock: bool = True,
    ) -> None:
        """Build the run prerequisites. Prototype: ``use_mock`` builds a MockSoc
        + a FakeDevice flux board (registered as "flux_dev" in the global device
        manager). Real make_soc_proxy / ml / predictor wiring is Phase B; pass an
        explicit ``resources`` to inject them."""
        if resources is None:
            resources = self._build_mock_resources() if use_mock else SetupResources()
        self._state.resources = resources
        self._state.version.bump(SETUP_VERSION_KEY)
        self._bus.emit(Event(EventType.SETUP_DONE, None))

    @staticmethod
    def _build_mock_resources() -> SetupResources:
        """MockSoc + soccfg + a FakeDevice flux board (no hardware)."""
        from zcu_tools.device import FakeDevice, GlobalDeviceManager
        from zcu_tools.program.v2.mocksoc import make_mock_soc, make_mock_soccfg

        soc = make_mock_soc()
        soccfg = make_mock_soccfg()
        if "flux_dev" not in GlobalDeviceManager.get_all_devices():
            GlobalDeviceManager.register_device("flux_dev", FakeDevice(fast_mode=True))
        return SetupResources(soc=soc, soccfg=soccfg, ml=None, predictor=None)

    # --- run control (cancellable; prototype uses fake per-Node execution) ---

    def stop_run(self) -> None:
        """Request cooperative cancellation of an in-progress run."""
        self._stop = True

    def start_run(self) -> InfoStore:
        """Run flux × Nodes, emitting run events. Drives each Node through its
        build_cfg + run_body (synthetic acquire + real fit) when available, else
        a fake fallback.

        Blocks until the sweep finishes or is stopped — the UI calls this on a
        worker thread. Emits RUN_STARTED, NODE_STARTED(name, idx) before each
        Node, POINT_DONE(idx) after each flux point, and RUN_FINISHED /
        RUN_STOPPED at the end.
        """
        self._stop = False
        self._running = True
        self._bus.emit(Event(EventType.RUN_STARTED, None))

        soc = self._state.resources.soc if self._state.resources else None

        def run_node(node: NodeInstance, snapshot, tools) -> Patch:
            self._bus.emit(Event(EventType.NODE_STARTED, (node.name, self._cur_idx)))
            spec = node.spec
            if spec.build_cfg is not None and spec.run_body is not None:
                cfg = spec.build_cfg(snapshot, node.params, tools)
                if cfg is None:
                    return Patch()  # build_cfg asked to skip this Node this point
                return spec.run_body(cfg, snapshot, tools, soc)
            # declaration-only Node (no body yet) → fake fallback
            data = {k: (0.05 if "kappa" in k else 1.0) for k in spec.provides}
            mods = {m: f"<{node.name}:{m}>" for m in spec.provides_modules}
            return Patch(data, mods)

        def pre_point(idx: int, flux, info: InfoStore, tools) -> None:
            self._cur_idx = idx
            # seed predict_freq (the pre-step's job): real predictor if set up,
            # else a flux-dependent stand-in so the synthetic Lorentzian moves.
            predictor = (
                self._state.resources.predictor if self._state.resources else None
            )
            if predictor is not None:
                info.point["predict_freq"] = float(predictor.predict_freq(flux))
            else:
                info.point["predict_freq"] = 5000.0 + 50.0 * idx

        def on_point(idx: int, _flux, _info) -> None:
            self._bus.emit(Event(EventType.POINT_DONE, idx))

        self._cur_idx = 0
        orch = Orchestrator(
            nodes=list(self._state.nodes), run_node=run_node, tools=Tools()
        )
        result = orch.run(
            self._state.flux_values,
            pre_point=pre_point,
            on_point=on_point,
            should_stop=lambda: self._stop,
        )
        self._running = False
        self._bus.emit(
            Event(EventType.RUN_STOPPED if self._stop else EventType.RUN_FINISHED, None)
        )
        return result

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
