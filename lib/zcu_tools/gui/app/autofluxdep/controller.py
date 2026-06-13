"""Controller façade for autofluxdep-gui.

State + EventBus coordinator, mirroring fluxdep/dispersive. Owns the workflow
definition commands (add/remove/reorder Nodes, set flux, set Node params), a
Setup that builds a MockSoc + FakeDevice flux board (the prototype's offline
resources), and a cancellable run that drives the orchestrator over the user's
ordered providers (with the predictor Service prepended). Each provider's Node
``produce`` runs a real acquire (against a flux-aware MockSoc offline or real
hardware), fits it, fills the provider's sweep Result in place, and notifies the
main thread to redraw. ``dry_run`` runs
the same orchestrator headless (no Results / notify) for direct testing of the
dependency model.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from zcu_tools.gui.app.autofluxdep.background import BackgroundService
from zcu_tools.gui.app.autofluxdep.derivation import DerivationService
from zcu_tools.gui.app.autofluxdep.events.run import (
    NodeEnteredPayload,
    PointDonePayload,
    RunFailedPayload,
    RunFinishedPayload,
    RunStartedPayload,
    RunStoppedPayload,
)
from zcu_tools.gui.app.autofluxdep.events.workflow import (
    FluxChangedPayload,
    WorkflowChangedPayload,
)
from zcu_tools.gui.app.autofluxdep.nodes.acquire import DEFAULT_ROUNDS
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, PlacedNode
from zcu_tools.gui.app.autofluxdep.nodes.predictor import PredictorBuilder
from zcu_tools.gui.app.autofluxdep.operation_gate import OperationGate
from zcu_tools.gui.app.autofluxdep.orchestrator import (
    InfoStore,
    ModuleSource,
    Notify,
    Orchestrator,
)
from zcu_tools.gui.app.autofluxdep.registry import create_placement
from zcu_tools.gui.app.autofluxdep.state import (
    FLUX_VERSION_KEY,
    WORKFLOW_VERSION_KEY,
    AutoFluxDepState,
)
from zcu_tools.gui.app.autofluxdep.tools import (
    FluxoniumPredictorAdapter,
    SimplePredictor,
    Tools,
)
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.services.build import build_session_services
from zcu_tools.gui.session.services.io_manager import IOManager
from zcu_tools.gui.session.services.progress import ProgressService

if TYPE_CHECKING:
    from zcu_tools.device.base import BaseDeviceInfo
    from zcu_tools.gui.session.pbar_host import ProgressBarModel
    from zcu_tools.gui.session.ports import ProgressTransport
    from zcu_tools.gui.session.services.connection import (
        ConnectRequest,
        LoadPredictorRequest,
        PredictFreqRequest,
    )
    from zcu_tools.gui.session.services.device import (
        ConnectDeviceRequest,
        DeviceEntry,
        DeviceSetupSnapshot,
        DeviceSnapshot,
        DisconnectDeviceRequest,
        SetupDeviceRequest,
    )
    from zcu_tools.gui.session.services.startup import (
        PersistedStartup,
        StartupConnectionRequest,
        StartupProjectRequest,
    )
    from zcu_tools.gui.session.types import SocCfgHandle
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary

logger = logging.getLogger(__name__)


class _MlModuleSource:
    """Transparent ``ModuleLibrary`` proxy honouring the ``ModuleSource`` contract.

    Two consumers share the ml threaded into the run env: ``project_snapshot``
    wants the orchestrator's ``ModuleSource`` contract — ``get_module(name)``
    returns None if absent (the snapshot then falls back to a Node-produced
    module / the dependency default) — whereas a node's cfg builder wants the full
    ``ModuleLibrary`` surface (``get_waveform`` / ``make_cfg``, which *should*
    raise on a missing reference). This proxy serves both: it overrides only
    ``get_module``'s raise-on-absent (``ModuleLibrary`` raises ``ValueError``) into
    None-on-absent, and forwards every other attribute to the wrapped library.
    """

    def __init__(self, ml: ModuleLibrary) -> None:
        self._ml = ml

    def get_module(self, name: str) -> Any:
        if name not in self._ml.modules:
            return None
        return self._ml.get_module(name)

    def __getattr__(self, attr: str) -> Any:
        # forward get_waveform / make_cfg / etc. to the real library (called only
        # for attributes not defined on this proxy; _ml is a real attribute).
        return getattr(self._ml, attr)


class Controller:
    def __init__(
        self,
        state: AutoFluxDepState,
        bus: EventBus,
        *,
        io_manager: IOManager | None = None,
        progress_transport: ProgressTransport | None = None,
        project_root: str | None = None,
    ) -> None:
        self._state = state
        self._bus = bus
        self._stop = False  # cooperative run-cancel flag
        self._running = False
        self._cur_idx = 0  # current flux index during a run (for POINT_DONE)

        # Base directory default result/database paths are anchored under (the
        # entry script injects the repo root). None falls back to cwd — fine for
        # tests / a `python -m` run from the repo root.
        import os

        self._project_root = project_root if project_root is not None else os.getcwd()

        # --- session-core infrastructure (this app owns its gate + executor) ---
        # autofluxdep composes the shared session services (connection / context /
        # device / startup) by injecting its own concrete infra through the session
        # ports (ADR-0019, session-core extraction decision 3): an app-local
        # OperationGate (conflict policy) + thin BackgroundService (no figure
        # routing) alongside the shared OperationHandles / ProgressService /
        # IOManager. The progress transport defaults to the Qt marshal so a GUI /
        # agent process works without the entry point wiring it; tests inject a
        # synchronous fake.
        self._io_manager = io_manager if io_manager is not None else IOManager()
        transport: ProgressTransport
        if progress_transport is not None:
            transport = progress_transport
        else:
            from zcu_tools.gui.session.adapters.qt_progress_transport import (
                QtProgressTransport,
            )

            transport = QtProgressTransport()
        self._operation_gate = OperationGate()
        self._operation_handles = OperationHandles()
        self._background_svc = BackgroundService()
        self._progress_svc = ProgressService(transport)
        session = build_session_services(
            state=state,
            bus=bus,
            gate=self._operation_gate,
            handles=self._operation_handles,
            background=self._background_svc,
            progress=self._progress_svc,
            io_manager=self._io_manager,
        )
        self._session = session
        self._conn_svc = session.connection
        self._ctx_svc = session.context
        self._dev_svc = session.device
        self._startup_svc = session.startup

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

    # ------------------------------------------------------------------
    # SessionControllerPort — the surface the shared setup / device / inspect
    # dialogs depend on (gui/session/controller_port). Each method delegates to
    # a session service; the dialogs never reach a concrete service, only this
    # Protocol. pyright verifies conformance at the dialog call sites (S4-c).
    # ------------------------------------------------------------------

    def get_bus(self) -> EventBus:
        return self._bus

    # -- setup dialog: project / startup --
    def apply_startup_project(self, req: StartupProjectRequest) -> bool:
        self._startup_svc.apply_project(req)
        return True

    def get_persisted_startup(self) -> PersistedStartup:
        return self._startup_svc.get_persisted()

    def get_project_root(self) -> str:
        return self._project_root

    # -- setup dialog: context switching --
    def use_context(self, label: str) -> None:
        self._ctx_svc.use_context(label)

    def new_context(
        self,
        bind_device: str | None = None,
        clone_from: str | None = None,
    ) -> None:
        """Create a new flux context, optionally bound to a flux device.

        ``bind_device`` (a connected device name) decides the flux unit/value:
        the unit comes from the device-type whitelist (Fast-Fail if unknown) and
        the value is *read* from the device's current state (never set).
        ``bind_device=None`` makes an unbound context. ``clone_from`` clones an
        existing context's ml/md.
        """
        if bind_device is not None:
            unit = self._dev_svc.get_device_unit_strict(bind_device)
            value = self._dev_svc.get_device_value_for_new_context(bind_device)
        else:
            unit, value = "none", None
        self._ctx_svc.new_context(value=value, unit=unit, clone_from=clone_from)

    def get_context_labels(self) -> list[str]:
        return self._ctx_svc.get_context_labels()

    def get_active_context_label(self) -> str | None:
        return self._ctx_svc.get_active_context_label()

    # -- setup dialog: connection --
    def start_connect(self, req: ConnectRequest) -> int:
        return self._conn_svc.start_connect(req)

    def bind_connection_outcome(
        self,
        on_finished: Callable[[], None],
        on_failed: Callable[[str], None],
    ) -> None:
        """Bind the single connection observer (the open SetupDialog) without
        exposing the service. Drops all prior slots first so a re-created dialog
        does not leak the previous observer (a no-arg ``disconnect()`` clears
        every connection)."""
        for signal in (
            self._conn_svc.connection_finished,
            self._conn_svc.connection_failed,
        ):
            try:
                signal.disconnect()
            except (TypeError, RuntimeError):
                pass  # no existing connections
        self._conn_svc.connection_finished.connect(on_finished)
        self._conn_svc.connection_failed.connect(on_failed)

    def remember_startup_connection(self, req: StartupConnectionRequest) -> None:
        self._startup_svc.remember_connection(req)

    def get_soccfg(self) -> SocCfgHandle | None:
        return self._conn_svc.get_soccfg()

    def get_device_unit(self, name: str) -> str:
        return self._dev_svc.get_device_unit(name)

    # -- predictor dialog: load / clear / predict (the shared PredictorDialog) --
    def load_predictor(self, req: LoadPredictorRequest) -> None:
        self._conn_svc.load_predictor(req)

    def clear_predictor(self) -> None:
        self._conn_svc.clear_predictor()

    def predict_freq(self, req: PredictFreqRequest) -> float:
        return self._conn_svc.predict_freq(req)

    def get_predictor_info(self) -> dict | None:
        return self._conn_svc.get_predictor_info()

    # -- device dialog: lifecycle --
    def start_connect_device(self, req: ConnectDeviceRequest) -> int:
        return self._dev_svc.start_connect_device(req)

    def start_disconnect_device(self, req: DisconnectDeviceRequest) -> int:
        return self._dev_svc.start_disconnect_device(req)

    def start_reconnect_device(self, name: str) -> None:
        self._dev_svc.start_reconnect_device(name)

    def start_setup_device(self, req: SetupDeviceRequest) -> int:
        return self._dev_svc.start_setup_device(req)

    def forget_device(self, name: str) -> None:
        self._dev_svc.forget_device(name)

    def cancel_device_operation(self, name: str) -> None:
        self._dev_svc.cancel_device_operation(name)

    # -- device dialog: queries --
    def list_devices(self) -> list[DeviceEntry]:
        return self._dev_svc.list_devices()

    def get_device_snapshot(self, name: str) -> DeviceSnapshot | None:
        return self._dev_svc.get_device_snapshot(name)

    def get_device_info(self, name: str) -> BaseDeviceInfo | None:
        return self._dev_svc.get_device_info(name)

    def is_memory_device(self, name: str) -> bool:
        return self._dev_svc.is_memory_device(name)

    def get_active_device_setup(self) -> DeviceSetupSnapshot | None:
        return self._dev_svc.get_active_setup()

    # -- device dialog: progress --
    def attach_progress(
        self, owner_id: str, listener: Callable[[], None]
    ) -> Callable[[], None]:
        return self._progress_svc.attach_by_owner(owner_id, listener)

    def progress_bars(self, owner_id: str) -> tuple[tuple[int, ProgressBarModel], ...]:
        return self._progress_svc.bars_for_owner(owner_id)

    # -- inspect dialog: md edit + ml view/rename/delete --
    def get_current_md(self) -> MetaDict:
        return self._ctx_svc.get_current_md()

    def get_current_ml(self) -> ModuleLibrary:
        return self._ctx_svc.get_current_ml()

    # -- cfg form (node detail pane): the LiveModel env contract --
    # The reused measure LiveModel fields fetch their environment through this
    # surface (ControllerProtocol). autofluxdep's node knobs are flat scalars /
    # sweeps with no md-reference / module-ref / device-ref, so these two never
    # fire for our schemas — they exist only so the controller structurally
    # conforms to the shared LiveModelEnv contract.
    def has_soc(self) -> bool:
        return self._state.exp_context.has_soc()

    def list_device_names(self) -> list[str]:
        return [entry.name for entry in self._dev_svc.list_devices()]

    def coerce_md_value(self, key: str, text: str) -> Any:
        return self._ctx_svc.coerce_md_value(key, text)

    def set_md_attr(self, key: str, value: Any) -> None:
        self._ctx_svc.set_md_attr(key, value)

    def del_md_attr(self, key: str) -> None:
        self._ctx_svc.del_md_attr(key)

    def rename_ml_module(self, old: str, new: str) -> None:
        self._ctx_svc.rename_ml_module(old, new)

    def rename_ml_waveform(self, old: str, new: str) -> None:
        self._ctx_svc.rename_ml_waveform(old, new)

    def del_ml_module(self, name: str) -> None:
        self._ctx_svc.del_ml_module(name)

    def del_ml_waveform(self, name: str) -> None:
        self._ctx_svc.del_ml_waveform(name)

    # --- workflow definition (the only writes the user makes) ---

    def _unique_name(self, base: str, *, exclude: PlacedNode | None = None) -> str:
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
        node = PlacedNode(builder=builder, name=name, overrides=params)
        self._state.nodes.append(node)
        self._state.version.bump(WORKFLOW_VERSION_KEY)
        logger.debug("add_node: %r (type=%r) params=%s", name, builder.name, params)
        self._bus.emit(WorkflowChangedPayload(name=node.name))
        return node

    def add_node_by_type(self, type_name: str) -> PlacedNode:
        """Add a fresh provider of ``type_name`` (from the registry) to the end.

        The instance name defaults to the type name, de-duped within the
        workflow (a second ``mist`` becomes ``mist_2``); the user can rename it.
        A Node is seeded with the GUI's default acquire ``rounds`` so the run
        averages a sensible number of passes (the user can tune it) — written
        through the placement's schema (the SSOT) rather than a params dict.
        """
        node = create_placement(type_name)
        node.name = self._unique_name(node.name)
        if "rounds" in node.schema.keys:
            node.schema.set_field("rounds", DEFAULT_ROUNDS)
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

    def set_node_params(self, index: int, params: Mapping[str, Any]) -> None:
        """Write one-or-more knob leaves of the Node at ``index`` into its schema SSOT.

        Phase 160b typed entry: each incoming key writes directly into the
        placement's own ``NodeCfgSchema`` (the per-placement value tree, the SSOT).
        A scalar value is coerced to the field's declared type; a ``SweepValue`` is
        accepted for a ``SweepSpec`` knob (the typed sweep widget now edits those).
        An unknown key fast-fails — the form only renders declared knobs, so an
        undeclared key is a real typo, not a silent extra. State writes happen on
        the main thread (the UI calls this), preserving the State main-thread
        invariant; the workflow version bumps + a ``WorkflowChangedPayload`` fires
        so dependents refresh, exactly as before.
        """
        node = self._state.nodes[index]
        for key, value in params.items():
            node.schema.set_field(key, value)  # fast-fails unknown key / wrong type
        self._state.version.bump(WORKFLOW_VERSION_KEY)
        logger.debug(
            "set_node_params[%d] (%r): keys=%s", index, node.name, list(params)
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

    def set_flux_device(self, name: str | None) -> None:
        """Designate which connected device the flux sweep is applied through.

        Its unit labels the flux axis (and the value is recorded into the run
        cfg's flux ``dev`` entry when a real acquire is wired). ``None`` clears the
        selection — the flux values are then bare numbers.
        """
        self._state.flux_device_name = name or None
        logger.debug("set_flux_device: %r", self._state.flux_device_name)

    def get_flux_device(self) -> str | None:
        return self._state.flux_device_name

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
        """Build the sweep's adaptive predictor from the active context.

        ``exp_context.predictor`` holds the raw ``FluxoniumPredictor`` (loaded at
        setup / by ConnectionService) or None. A real predictor is wrapped into
        the adaptive ``FluxoniumPredictorAdapter``; with none loaded we fall back
        to the ``SimplePredictor`` stand-in, so a mock / unconfigured run still
        drives the same calibrate loop.
        """
        raw = self._state.exp_context.predictor
        predictor = (
            FluxoniumPredictorAdapter(fluxonium=raw)
            if raw is not None
            else SimplePredictor()
        )
        return Tools(predictor=predictor)

    def _allocate_results(self, flux: Any) -> dict[str, Any]:
        """Pre-allocate each user provider's sweep Result on the main thread.

        ``flux`` is the full flux axis (filled into each Result up front for the
        Plotter's x). A Service (predictor) has no Result (``make_init_result``
        returns None), so it is absent from the map — the orchestrator curries no
        Result for it and the UI builds no figure, with no ``isinstance``.
        """
        results: dict[str, Any] = {}
        for node in self._state.nodes:
            result = node.builder.make_init_result(node.schema, flux)
            if result is not None:
                results[node.name] = result
        return results

    def start_run(self, notify: Notify | None = None) -> InfoStore:
        """Run flux × providers, emitting run events. Each provider's Node
        ``produce`` runs a real acquire (flux-aware MockSoc offline or real
        hardware), fits it, fills its sweep Result in place, and fires
        ``notify(name, idx)`` so the main thread redraws.

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

        ctx = self._state.exp_context
        soc, soccfg, md = ctx.soc, ctx.soccfg, ctx.md
        # Adapt the ModuleLibrary to the orchestrator's ModuleSource contract
        # (None on absent rather than raise) before threading it in; the proxy
        # still forwards make_cfg / get_waveform for a node's cfg builder.
        ml = _MlModuleSource(ctx.ml)
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

        # Build the sweep's adaptive predictor once and stash it as run-lived
        # state (like run_results) so an Info dialog / a test can inspect the
        # predictor the run calibrated.
        tools = self._build_tools()
        self._state.run_predictor = tools.predictor
        orch = Orchestrator(
            providers=self._build_providers(),
            tools=tools,
            ml=ml,
            soc=soc,
            soccfg=soccfg,
            md=md,
            # The user's flux-source pick reaches each RunEnv so a real acquire
            # writes this point's flux into cfg.dev[flux_device] (RB-0b).
            flux_device=self._state.flux_device_name,
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
        # A produce error is a terminal RUN_FAILED (distinct from a cooperative
        # stop): the orchestrator caught it so the worker QThread never aborts.
        # Either terminal state unlocks the UI; the failure carries its message.
        if orch.run_error is not None:
            logger.error("run failed at flux idx %d: %s", self._cur_idx, orch.run_error)
            self._bus.emit(RunFailedPayload(message=str(orch.run_error)))
        elif self._stop:
            logger.info("run stopped at flux idx %d", self._cur_idx)
            self._bus.emit(RunStoppedPayload())
        else:
            logger.info("run finished: %d flux point(s)", len(self._state.flux_values))
            self._bus.emit(RunFinishedPayload())
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
        tools: Tools | None = None,
        ml: ModuleSource | None = None,
        derivations: list[DerivationService] | None = None,
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
