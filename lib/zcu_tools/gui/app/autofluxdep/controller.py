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
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zcu_tools.gui.app.autofluxdep.cfg.schema import NodeCfgPersistenceError
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
from zcu_tools.gui.app.autofluxdep.operation_gate import OperationGate, OperationKind
from zcu_tools.gui.app.autofluxdep.orchestrator import (
    InfoStore,
    ModuleSource,
    Notify,
    Orchestrator,
)
from zcu_tools.gui.app.autofluxdep.registry import create_placement
from zcu_tools.gui.app.autofluxdep.services.persistence_types import (
    AppPersistedState,
    PersistedFluxSweep,
    PersistedNode,
    PersistedWorkflow,
    PersistenceError,
    RestoreIssue,
    RestoreReport,
)
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
from zcu_tools.gui.background import BackgroundRunner
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.controller_mixin import SessionControllerMixin
from zcu_tools.gui.session.operation_handles import (
    AwaitResult,
    OperationHandles,
    OperationOutcome,
)
from zcu_tools.gui.session.operation_runner import (
    BgResult,
    ExclusionRequest,
    OperationRunner,
    OperationSpec,
    SettleFn,
)
from zcu_tools.gui.session.scopes import progress_ambient
from zcu_tools.gui.session.services.build import build_session_services
from zcu_tools.gui.session.services.io_manager import IOManager
from zcu_tools.gui.session.services.progress import ProgressService

if TYPE_CHECKING:
    from zcu_tools.gui.app.autofluxdep.services.caretaker import (
        PersistenceCaretaker,
        RestoreOutcome,
    )
    from zcu_tools.gui.session.ports import ProgressTransport
    from zcu_tools.gui.session.services.startup import (
        StartupProjectRequest,
    )
    from zcu_tools.meta_tool import ModuleLibrary

logger = logging.getLogger(__name__)

_RUN_OWNER_ID = "autofluxdep-run"


@dataclass(frozen=True)
class _RunOutcome:
    """Worker return value for one autofluxdep sweep operation."""

    info: InfoStore
    run_error: Exception | None
    stopped: bool


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


class Controller(SessionControllerMixin):
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
        self._cur_idx = 0  # current flux index during a run (for POINT_DONE)
        self._active_run_token: int | None = None
        self._run_stop_event: threading.Event | None = None
        self._last_run_info: InfoStore | None = None
        self._caretaker: PersistenceCaretaker | None = None

        # Base directory default result/database paths are anchored under (the
        # entry script injects the repo root). None falls back to cwd — fine for
        # tests / a `python -m` run from the repo root.
        import os

        self._project_root = project_root if project_root is not None else os.getcwd()

        # --- session-core infrastructure (this app owns its gate + executor) ---
        # autofluxdep composes the shared session services (connection / context /
        # device / startup) by injecting its own concrete infra through the session
        # ports (ADR-0019, session-core extraction decision 3): an app-local
        # OperationGate (conflict policy) + the shared BackgroundRunner (no figure
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
        self._background_svc = BackgroundRunner()
        self._progress_svc = ProgressService(transport)
        self._runner = OperationRunner(
            self._operation_gate,
            self._operation_handles,
            self._progress_svc,
            self._background_svc,
        )
        session = build_session_services(
            state=state,
            bus=bus,
            gate=self._operation_gate,
            handles=self._operation_handles,
            background=self._background_svc,
            progress=self._progress_svc,
            io_manager=self._io_manager,
            runner=self._runner,
            project_root=self._project_root,
        )
        self._session = session
        self._soc_svc = session.soc_connection
        self._pred_svc = session.predictor
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
        return self._active_run_token is not None

    @property
    def active_run_token(self) -> int | None:
        """The operation token for the live RUN, or None when idle."""
        return self._active_run_token

    @property
    def last_run_info(self) -> InfoStore | None:
        """The latest terminal sweep InfoStore, if the worker produced one."""
        return self._last_run_info

    # ------------------------------------------------------------------
    # SessionControllerPort — the surface the shared setup / device / inspect
    # dialogs depend on (gui/session/controller_port). The identical forwards live
    # in SessionControllerMixin (read through the _*_svc accessors, supplied below
    # in __init__); only the methods whose body diverges are kept here. pyright
    # verifies conformance at the dialog call sites.
    # ------------------------------------------------------------------

    def get_bus(self) -> EventBus:
        return self._bus

    # -- Memento Originator (workflow persistence) -----------------------------

    def attach_caretaker(self, caretaker: PersistenceCaretaker) -> None:
        """Wire the app-level persistence caretaker built by the composition root."""
        self._caretaker = caretaker

    def capture_persisted_state(self) -> AppPersistedState:
        """Snapshot the persistent autofluxdep workflow state, without disk I/O."""
        nodes = tuple(
            PersistedNode(
                type_name=node.type_name,
                name=node.name,
                cfg_raw=node.schema.to_persisted_raw(),
            )
            for node in self._state.nodes
        )
        return AppPersistedState(
            workflow=PersistedWorkflow(nodes=nodes),
            flux=PersistedFluxSweep(
                start_expr=self._state.flux_start_expr,
                stop_expr=self._state.flux_stop_expr,
                npts_expr=self._state.flux_npts_expr,
                values=tuple(float(v) for v in self._state.flux_values),
            ),
        )

    def restore_persisted_state(self, state: AppPersistedState) -> RestoreReport:
        """Apply a persisted workflow snapshot, rejecting only invalid nodes."""
        rejected: list[RestoreIssue] = []
        self._state.nodes = []

        for index, persisted in enumerate(state.workflow.nodes):
            subject = f"node[{index}] {persisted.name!r}"
            try:
                node = create_placement(persisted.type_name)
                node.name = self._unique_name(persisted.name or node.name)
                node.schema.restore_persisted_raw(persisted.cfg_raw)
            except (
                KeyError,
                NodeCfgPersistenceError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as exc:
                rejected.append(RestoreIssue(subject=subject, message=str(exc)))
                continue
            self._state.nodes.append(node)

        self._state.flux_start_expr = state.flux.start_expr
        self._state.flux_stop_expr = state.flux.stop_expr
        self._state.flux_npts_expr = state.flux.npts_expr
        self._state.flux_values = [float(v) for v in state.flux.values]
        self._state.run_results = {}
        self._state.version.bump(WORKFLOW_VERSION_KEY)
        self._state.version.bump(FLUX_VERSION_KEY)
        self._bus.emit(WorkflowChangedPayload(name=None))
        self._bus.emit(FluxChangedPayload(count=len(self._state.flux_values)))
        return RestoreReport(
            restored_nodes=len(self._state.nodes),
            rejected_nodes=tuple(rejected),
        )

    def restore_all(self, *, load: bool = True) -> RestoreOutcome | None:
        if self._caretaker is None:
            return None
        outcome = self._caretaker.restore_all(load=load)
        if outcome.load_error is not None:
            logger.warning("%s", outcome.load_error)
        report = outcome.report
        if isinstance(report, RestoreReport) and report.rejected_nodes:
            logger.warning(
                "autofluxdep persistence rejected %d node(s): %s",
                len(report.rejected_nodes),
                "; ".join(f"{i.subject}: {i.message}" for i in report.rejected_nodes),
            )
        return outcome

    def persist_all(self) -> None:
        if self._caretaker is None:
            return
        try:
            self._caretaker.flush()
        except PersistenceError:
            logger.exception("autofluxdep settings save failed")

    # -- setup dialog: project / startup --
    # apply_startup_project diverges (autofluxdep returns bool; measure returns the
    # resolved-project dict per WIRE-48) so it stays a per-app override. get_bus /
    # get_project_root also stay per-app (app EventBus subtype / app state). Every
    # other SessionControllerPort forward lives in SessionControllerMixin.
    def apply_startup_project(self, req: StartupProjectRequest) -> bool:
        self._startup_svc.apply_project(req)
        return True

    def get_project_root(self) -> str:
        return self._project_root

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

    def set_flux_sweep_expressions(
        self, start_expr: str, stop_expr: str, npts_expr: str
    ) -> None:
        """Remember the user's editable flux sweep expressions."""
        self._state.flux_start_expr = start_expr
        self._state.flux_stop_expr = stop_expr
        self._state.flux_npts_expr = npts_expr
        self._state.version.bump(FLUX_VERSION_KEY)

    def get_flux_sweep_expressions(self) -> tuple[str, str, str]:
        return (
            self._state.flux_start_expr,
            self._state.flux_stop_expr,
            self._state.flux_npts_expr,
        )

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

    def stop_run(self, reason: str = "Autofluxdep run stop requested") -> bool:
        """Request cooperative cancellation of an in-progress run."""
        logger.info("stop_run requested (at flux idx %d)", self._cur_idx)
        token = self._active_run_token
        if token is None:
            event = self._run_stop_event
            if event is not None:
                event.set()
            return False
        self._operation_handles.stop(token, reason=reason)
        return True

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
        setup / by PredictorService) or None. A real predictor is wrapped into
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
        md = self._state.exp_context.md
        for node in self._state.nodes:
            result = node.builder.make_init_result(node.schema, flux, md=md)
            if result is not None:
                results[node.name] = result
        return results

    def start_run(self, notify: Notify | None = None) -> int:
        """Start an async flux × providers RUN operation.

        Each provider's Node
        ``produce`` runs a real acquire (flux-aware MockSoc offline or real
        hardware), fits it, fills its sweep Result in place, and fires
        ``notify(name, idx)`` so the main thread redraws.

        ``notify`` is the row-updated callback (the UI passes one bound to its
        Plotters); None for a headless run. Returns the OperationRunner token
        immediately; terminal state is observed through run events or
        ``await_operation``. Emits RUN_STARTED, POINT_DONE(idx) after each flux
        point, and RUN_FINISHED / RUN_STOPPED / RUN_FAILED.
        """
        if self.is_running:
            raise RuntimeError("autofluxdep run is already active")

        self._cur_idx = 0
        self._last_run_info = None
        logger.info(
            "run start: %d user Node(s) %s over %d flux point(s)",
            len(self._state.nodes),
            self._state.node_names(),
            len(self._state.flux_values),
        )

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
        flux_values = list(self._state.flux_values)
        providers = self._build_providers()
        flux_device = self._state.flux_device_name

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
        stop_event = threading.Event()
        self._run_stop_event = stop_event

        def work(factory: Any) -> _RunOutcome:
            from zcu_tools.progress_bar import make_pbar

            with progress_ambient(factory):
                pbar = make_pbar(
                    total=len(flux_values),
                    desc="flux sweep",
                    leave=True,
                )

                def on_point_with_progress(
                    idx: int, flux: float, info: InfoStore
                ) -> None:
                    pbar.update((idx + 1) - pbar.n)
                    on_point(idx, flux, info)

                try:
                    orch = Orchestrator(
                        providers=providers,
                        tools=tools,
                        ml=ml,
                        soc=soc,
                        soccfg=soccfg,
                        md=md,
                        # The user's flux-source pick reaches each RunEnv so a real
                        # acquire writes this point's flux into cfg.dev[flux_device]
                        # (RB-0b).
                        flux_device=flux_device,
                        results=results,
                        notify=notify,
                    )
                    info = orch.run(
                        flux_values,
                        on_point=on_point_with_progress,
                        on_node=on_node,
                        should_stop=stop_event.is_set,
                    )
                    pbar.refresh()
                    return _RunOutcome(
                        info=info,
                        run_error=orch.run_error,
                        stopped=stop_event.is_set(),
                    )
                finally:
                    pbar.close()

        def on_terminal(bg: BgResult, settle: SettleFn) -> None:
            if bg.ok:
                outcome = bg.result
                if not isinstance(outcome, _RunOutcome):
                    _on_run_failed(
                        RuntimeError(
                            f"autofluxdep run returned {type(outcome).__name__}"
                        ),
                        settle,
                    )
                    return
                self._last_run_info = outcome.info
                if outcome.run_error is not None:
                    _on_run_failed(outcome.run_error, settle)
                elif outcome.stopped:
                    _on_run_stopped(settle)
                else:
                    _on_run_finished(settle)
                return

            assert bg.error is not None
            if stop_event.is_set():
                _on_run_stopped(settle)
            else:
                _on_run_failed(bg.error, settle)

        def _clear_active_run() -> None:
            self._active_run_token = None
            self._run_stop_event = None

        def _on_run_finished(settle: SettleFn) -> None:
            logger.info("run finished: %d flux point(s)", len(flux_values))
            _clear_active_run()
            settle(OperationOutcome("finished"))
            self._bus.emit(RunFinishedPayload())

        def _on_run_stopped(settle: SettleFn) -> None:
            logger.info("run stopped at flux idx %d", self._cur_idx)
            _clear_active_run()
            settle(OperationOutcome("cancelled"))
            self._bus.emit(RunStoppedPayload())

        def _on_run_failed(error: Exception, settle: SettleFn) -> None:
            logger.error("run failed at flux idx %d: %s", self._cur_idx, error)
            _clear_active_run()
            settle(OperationOutcome("failed", str(error)))
            self._bus.emit(RunFailedPayload(message=str(error)))

        spec = OperationSpec(
            exclusion=ExclusionRequest(
                kind=OperationKind.RUN,
                owner_id=_RUN_OWNER_ID,
            ),
            owner_id=_RUN_OWNER_ID,
            wants_progress=True,
            cancel_hook=stop_event.set,
            work=work,
            run_in_pool=False,
            on_terminal=on_terminal,
        )

        try:
            token = self._runner.begin(spec)
        except Exception:
            self._run_stop_event = None
            raise
        self._active_run_token = token
        self._bus.emit(RunStartedPayload())
        return token

    def await_operation(self, operation_id: int, timeout: float) -> AwaitResult | None:
        """Block until an async operation settles or a wakeup condition fires."""
        return self._operation_handles.await_outcome(operation_id, timeout)

    def get_operation_progress(self, operation_id: int) -> tuple:
        """Live progress bars for one operation by id."""
        return self._progress_svc.bars_for_operation(operation_id)

    def active_operation_count(self) -> int:
        """How many shared operation handles are live right now."""
        return self._operation_handles.live_count()

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
