"""Controller façade for autofluxdep-gui.

State + EventBus coordinator, mirroring fluxdep/dispersive. Owns the workflow
definition commands (add/remove/reorder Nodes, set flux, set Node params), a
Setup that builds a MockSoc + FakeDevice flux board (the prototype's offline
resources), and a cancellable run that drives the orchestrator over the user's
ordered providers (with the predictor Service prepended). Each provider's Node
``produce`` synthesises a signal, fits it, fills the provider's sweep Result in
place, and notifies the main thread to redraw — no hardware. ``dry_run`` runs
the same orchestrator headless (no Results / notify) for direct testing of the
dependency model.
"""

from __future__ import annotations

import logging

from typing_extensions import Any, Optional

from zcu_tools.gui.app.autofluxdep.derivation import DerivationService
from zcu_tools.gui.app.autofluxdep.event_bus import (
    EventBus,
    FluxChangedPayload,
    NodeEnteredPayload,
    PointDonePayload,
    RunFinishedPayload,
    RunStartedPayload,
    RunStoppedPayload,
    SetupDonePayload,
    WorkflowChangedPayload,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, PlacedNode
from zcu_tools.gui.app.autofluxdep.nodes.predictor import PredictorBuilder
from zcu_tools.gui.app.autofluxdep.nodes.synth import (
    DEFAULT_ACQUIRE_DELAY,
    DEFAULT_ROUNDS,
)
from zcu_tools.gui.app.autofluxdep.orchestrator import (
    InfoStore,
    ModuleSource,
    Notify,
    Orchestrator,
)
from zcu_tools.gui.app.autofluxdep.registry import create_placement
from zcu_tools.gui.app.autofluxdep.state import (
    FLUX_VERSION_KEY,
    SETUP_VERSION_KEY,
    WORKFLOW_VERSION_KEY,
    AutoFluxDepState,
    SetupRequest,
    SetupResources,
)
from zcu_tools.gui.app.autofluxdep.tools import SimplePredictor, Tools

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, state: AutoFluxDepState, bus: EventBus) -> None:
        self._state = state
        self._bus = bus
        self._stop = False  # cooperative run-cancel flag
        self._running = False
        self._cur_idx = 0  # current flux index during a run (for POINT_DONE)

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

    def _unique_name(self, base: str, *, exclude: Optional[PlacedNode] = None) -> str:
        """A workflow-unique instance name from ``base`` (append _2, _3, … if taken).

        ``exclude`` is a placement allowed to keep its own name (for rename — a
        no-op rename to the current name must not bump to _2).
        """
        taken = {n.name for n in self._state.nodes if n is not exclude}
        if base not in taken:
            return base
        i = 2
        while f"{base}_{i}" in taken:
            i += 1
        return f"{base}_{i}"

    def add_node(self, builder: Builder, **params: Any) -> PlacedNode:
        name = self._unique_name(builder.name)
        node = PlacedNode(builder=builder, name=name, params=dict(params))
        self._state.nodes.append(node)
        self._state.version.bump(WORKFLOW_VERSION_KEY)
        logger.debug("add_node: %r (type=%r) params=%s", name, builder.name, params)
        self._bus.emit(WorkflowChangedPayload(name=node.name))
        return node

    def add_node_by_type(self, type_name: str) -> PlacedNode:
        """Add a fresh provider of ``type_name`` (from the registry) to the end.

        The instance name defaults to the type name, de-duped within the
        workflow (a second ``mist`` becomes ``mist_2``); the user can rename it.
        A Node is seeded with the default ``acquire_delay`` / ``rounds`` so the
        GUI run paces the synthetic liveplot visibly and shows the per-round
        accumulation (the user can tune both).
        """
        node = create_placement(type_name)
        node.name = self._unique_name(node.name)
        if "acquire_delay" in node.builder.base_params:
            node.params.setdefault("acquire_delay", DEFAULT_ACQUIRE_DELAY)
        if "rounds" in node.builder.base_params:
            node.params.setdefault("rounds", DEFAULT_ROUNDS)
        self._state.nodes.append(node)
        self._state.version.bump(WORKFLOW_VERSION_KEY)
        logger.debug("add_node_by_type: %r -> %r", type_name, node.name)
        self._bus.emit(WorkflowChangedPayload(name=node.name))
        return node

    def rename_node(self, index: int, new_name: str) -> str:
        """Rename the placement at ``index`` to a workflow-unique ``new_name``.

        Returns the actual name applied (de-duped + stripped). A blank name is
        rejected (kept unchanged) — fast-fail on the empty case rather than
        silently naming it the type. Used to distinguish repeated placements
        (e.g. two ``mist`` → ``g_mist`` / ``e_mist``).
        """
        node = self._state.nodes[index]
        cleaned = new_name.strip()
        if not cleaned:
            logger.debug(
                "rename_node[%d]: blank name rejected, kept %r", index, node.name
            )
            return node.name  # blank → no-op, keep current name
        old = node.name
        node.name = self._unique_name(cleaned, exclude=node)
        self._state.version.bump(WORKFLOW_VERSION_KEY)
        logger.debug("rename_node[%d]: %r -> %r", index, old, node.name)
        self._bus.emit(WorkflowChangedPayload(name=node.name))
        return node.name

    def remove_node(self, name: str) -> None:
        before = len(self._state.nodes)
        self._state.nodes = [n for n in self._state.nodes if n.name != name]
        self._state.version.bump(WORKFLOW_VERSION_KEY)
        logger.debug("remove_node: %r (%d -> %d)", name, before, len(self._state.nodes))
        self._bus.emit(WorkflowChangedPayload(name=name))

    def reorder(self, index: int, delta: int) -> int:
        """Move the Node at ``index`` by ``delta`` (±1). Returns the new index."""
        nodes = self._state.nodes
        new_index = index + delta
        if not (0 <= index < len(nodes) and 0 <= new_index < len(nodes)):
            return index  # out of range → no-op
        nodes[index], nodes[new_index] = nodes[new_index], nodes[index]
        self._state.version.bump(WORKFLOW_VERSION_KEY)
        logger.debug("reorder: %d <-> %d", index, new_index)
        self._bus.emit(WorkflowChangedPayload(name=None))
        return new_index

    def set_node_params(self, index: int, params: dict[str, Any]) -> None:
        """Replace the tuned params of the Node at ``index``."""
        self._state.nodes[index].params = dict(params)
        self._state.version.bump(WORKFLOW_VERSION_KEY)
        logger.debug(
            "set_node_params[%d] (%r): %s", index, self._state.nodes[index].name, params
        )
        self._bus.emit(WorkflowChangedPayload(name=None))

    def set_flux_values(self, values: list[float]) -> None:
        self._state.flux_values = list(values)
        self._state.version.bump(FLUX_VERSION_KEY)
        if values:
            logger.debug(
                "set_flux_values: n=%d range=[%g, %g]",
                len(values),
                values[0],
                values[-1],
            )
        else:
            logger.debug("set_flux_values: cleared")
        self._bus.emit(FluxChangedPayload(count=len(values)))

    # --- setup (MockSoc + FakeDevice, or a real ZCU + YOKO + predictor) ---

    def setup(
        self,
        request: Optional[SetupRequest] = None,
        *,
        use_mock: bool = True,
    ) -> None:
        """Build the run prerequisites from a ``SetupRequest``.

        With no ``request`` a mock setup is built (the convenience for tests /
        the ``use_mock`` shorthand). Otherwise the request's ``use_mock`` picks
        the path: mock (MockSoc + FakeDevice + SimplePredictor) or real (the soc
        proxy + a YOKOGS200 flux source + a FluxoniumPredictor from params.json).

        A real connection that fails raises (Fast Fail — the dialog reports it
        and stays open); a missing / unloadable predictor file degrades to a
        SimplePredictor with a warning rather than blocking the run.
        """
        if request is None:
            request = SetupRequest(use_mock=use_mock)
        resources = (
            self._build_mock_resources()
            if request.use_mock
            else self._build_real_resources(request)
        )
        self._state.resources = resources
        self._state.version.bump(SETUP_VERSION_KEY)
        logger.info(
            "setup done: soc=%s predictor=%s (use_mock=%s)",
            type(resources.soc).__name__ if resources.soc is not None else None,
            type(resources.predictor).__name__
            if resources.predictor is not None
            else None,
            request.use_mock,
        )
        self._bus.emit(SetupDonePayload())

    @staticmethod
    def _build_mock_resources() -> SetupResources:
        """MockSoc + soccfg + a FakeDevice flux board + a SimplePredictor."""
        from zcu_tools.device import FakeDevice, GlobalDeviceManager
        from zcu_tools.program.v2.mocksoc import make_mock_soc, make_mock_soccfg

        soc = make_mock_soc()
        soccfg = make_mock_soccfg()
        if "flux_dev" not in GlobalDeviceManager.get_all_devices():
            GlobalDeviceManager.register_device("flux_dev", FakeDevice(fast_mode=True))
        return SetupResources(
            soc=soc, soccfg=soccfg, ml=None, predictor=SimplePredictor()
        )

    @staticmethod
    def _build_real_resources(request: SetupRequest) -> SetupResources:
        """Connect a real ZCU + YOKOGS200 flux source + load the predictor.

        Each step uses the existing library API directly (no gui-external module
        is modified). A connection failure propagates (Fast Fail). The flux
        device is registered as "flux_dev" in the global manager so the run finds
        it; the predictor degrades to a SimplePredictor if params.json is absent
        or unloadable.
        """
        from zcu_tools.remote import make_soc_proxy

        logger.info("connecting real ZCU at %s:%d", request.ip, request.port)
        soc, soccfg = make_soc_proxy(request.ip, request.port)

        Controller._connect_flux_device(request.flux_device_address)
        predictor = Controller._load_predictor(request.params_path)

        return SetupResources(soc=soc, soccfg=soccfg, ml=None, predictor=predictor)

    @staticmethod
    def _connect_flux_device(address: str) -> None:
        """Connect a YOKOGS200 at ``address`` and register it as "flux_dev".

        Raises if the address is blank (a real run needs a flux source) or the
        VISA connection fails (Fast Fail — surfaced by the dialog).
        """
        from zcu_tools.device import GlobalDeviceManager
        from zcu_tools.device.yoko import YOKOGS200

        if not address.strip():
            raise ValueError("A flux device address is required for a real setup")
        import pyvisa  # type: ignore[import-untyped]

        logger.info("connecting YOKOGS200 flux source at %s", address)
        device = YOKOGS200(address.strip(), pyvisa.ResourceManager())
        GlobalDeviceManager.register_device("flux_dev", device)

    @staticmethod
    def _load_predictor(params_path: str) -> object:
        """Load a FluxoniumPredictor from ``params_path``; degrade on failure.

        A blank path or an unreadable / invalid params.json logs a warning and
        falls back to the SimplePredictor stand-in, so a missing fit file does
        not block the run (the prediction is just a linear stand-in then).
        """
        if not params_path.strip():
            logger.warning("no params.json given; using SimplePredictor stand-in")
            return SimplePredictor()
        try:
            from zcu_tools.gui.app.autofluxdep.tools import FluxoniumPredictorAdapter
            from zcu_tools.simulate.fluxonium import FluxoniumPredictor

            fluxonium = FluxoniumPredictor.from_file(params_path.strip())
            logger.info("loaded FluxoniumPredictor from %s", params_path)
            # wrap into the Predictor interface (physical prediction + IDW error
            # correction) so qubit_freq's calibrate drives the same feedback loops
            return FluxoniumPredictorAdapter(fluxonium=fluxonium)
        except Exception as exc:
            logger.warning(
                "could not load predictor from %s (%s); using SimplePredictor",
                params_path,
                exc,
            )
            return SimplePredictor()

    # --- run control (cancellable) ---

    def stop_run(self) -> None:
        """Request cooperative cancellation of an in-progress run."""
        logger.info("stop_run requested (at flux idx %d)", self._cur_idx)
        self._stop = True

    def _build_providers(self) -> list[PlacedNode]:
        """The execution sequence: the predictor Service prepended to the user's
        providers. The Service is loaded because qubit_freq requires
        ``predict_freq`` — it is not in the user's list, but it runs first each
        point so its ``predict_freq`` is this-point-available downstream."""
        service = PlacedNode(builder=PredictorBuilder())
        return [service, *self._state.nodes]

    def _build_tools(self) -> Tools:
        predictor = self._state.resources.predictor if self._state.resources else None
        return Tools(predictor=predictor)

    def _allocate_results(self, flux: "Any") -> dict[str, Any]:
        """Pre-allocate each user provider's sweep Result on the main thread.

        ``flux`` is the full flux axis (filled into each Result up front for the
        Plotter's x). A Service (predictor) has no Result (``make_init_result``
        returns None), so it is absent from the map — the orchestrator curries no
        Result for it and the UI builds no figure, with no ``isinstance``.
        """
        results: dict[str, Any] = {}
        for node in self._state.nodes:
            result = node.builder.make_init_result(node.params, flux)
            if result is not None:
                results[node.name] = result
        return results

    def start_run(self, notify: Optional[Notify] = None) -> InfoStore:
        """Run flux × providers, emitting run events. Each provider's Node
        ``produce`` synthesises a signal, fits it, fills its sweep Result in
        place, and fires ``notify(name, idx)`` so the main thread redraws.

        ``notify`` is the row-updated callback (the UI passes one bound to its
        Plotters); None for a headless run. Blocks until the sweep finishes or
        is stopped — the UI calls this on a worker thread. Emits RUN_STARTED,
        POINT_DONE(idx) after each flux point, and RUN_FINISHED / RUN_STOPPED.
        """
        self._stop = False
        self._running = True
        self._cur_idx = 0
        logger.info(
            "run start: %d user Node(s) %s over %d flux point(s)",
            len(self._state.nodes),
            self._state.node_names(),
            len(self._state.flux_values),
        )
        self._bus.emit(RunStartedPayload())

        soc = self._state.resources.soc if self._state.resources else None
        # The UI pre-allocates Results (+ binds Plotters) before starting the
        # worker; a headless caller has not, so allocate here. Either way the
        # worker fills these exact containers in place.
        if not self._state.run_results:
            self.prepare_run_results()
        results = self._state.run_results

        def on_point(idx: int, flux: float, info: InfoStore) -> None:
            del flux, info  # POINT_DONE carries only the index
            self._cur_idx = idx
            self._bus.emit(PointDonePayload(idx=idx))

        user_node_names = {n.name for n in self._state.nodes}

        def on_node(name: str, idx: int) -> None:
            # a provider is about to run → let the UI auto-follow to its run tab.
            # The orchestrator fires for every provider (it does not distinguish a
            # Service); the controller knows the Service boundary, so it only
            # forwards user-list Nodes — the predictor Service has no list row to
            # navigate to.
            if name in user_node_names:
                self._bus.emit(NodeEnteredPayload(name=name, idx=idx))

        orch = Orchestrator(
            providers=self._build_providers(),
            tools=self._build_tools(),
            soc=soc,
            results=results,
            notify=notify,
        )
        info = orch.run(
            self._state.flux_values,
            on_point=on_point,
            on_node=on_node,
            should_stop=lambda: self._stop,
        )
        self._running = False
        if self._stop:
            logger.info("run stopped at flux idx %d", self._cur_idx)
        else:
            logger.info("run finished: %d flux point(s)", len(self._state.flux_values))
        self._bus.emit(RunStoppedPayload() if self._stop else RunFinishedPayload())
        return info

    def prepare_run_results(self) -> dict[str, Any]:
        """Allocate + store this run's Results in State (main thread, Run start).

        The UI calls this before starting the worker so the Plotters bind to the
        same Result objects the worker fills. Returns the name→Result map.
        """
        import numpy as np

        flux = np.asarray(self._state.flux_values or [0.0], dtype=np.float64)
        self._state.run_results = self._allocate_results(flux)
        return self._state.run_results

    # --- dry run (headless: real orchestrator, no Results / notify) ---

    def dry_run(
        self,
        tools: Optional[Tools] = None,
        ml: Optional[ModuleSource] = None,
        derivations: Optional[list[DerivationService]] = None,
    ) -> InfoStore:
        """Exercise the dependency model headless: the same orchestrator over the
        same providers (predictor Service prepended), but with no Results and no
        notify (nothing to draw). ``tools`` defaults to a fresh ``Tools`` (a
        SimplePredictor if none bound); ``ml`` is the module-library fallback;
        smoothing is auto-built from declarations, ``derivations`` are extra
        producers. Returns the final InfoStore. Providers run in list order (no
        topo sort)."""
        orch = Orchestrator(
            providers=self._build_providers(),
            tools=tools or Tools(predictor=SimplePredictor()),
            ml=ml,
            derivations=derivations or [],
        )
        return orch.run(self._state.flux_values)
