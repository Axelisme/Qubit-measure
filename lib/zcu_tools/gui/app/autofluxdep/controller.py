"""Controller façade for autofluxdep-gui.

State + EventBus coordinator, mirroring fluxdep/dispersive. Owns the workflow
definition commands (add/remove/reorder Nodes, set flux, set Node params), setup
resources (including MockSoc + FakeDevice for offline runs), and a cancellable
run that drives the orchestrator over the user's ordered providers (with the
predictor Service prepended). Each provider's Node
``produce`` runs a real acquire (against a flux-aware MockSoc offline or real
hardware), fits it, fills the provider's sweep Result in place, and notifies the
main thread to redraw. ``dry_run`` runs
the same orchestrator headless (no Results / notify) for direct testing of the
dependency model.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from qtpy.QtCore import (
    QObject,
    Qt,
    QThread,
    QTimer,
    Signal,  # type: ignore[attr-defined]
)

from zcu_tools.gui.app.autofluxdep.cfg import (
    CfgSectionValue,
    RunCfgSnapshot,
    validate_override_plan_base_cfg,
)
from zcu_tools.gui.app.autofluxdep.cfg.schema import NodeCfgPersistenceError
from zcu_tools.gui.app.autofluxdep.events.run import (
    NodeEnteredPayload,
    PointDonePayload,
    RunContinuedPayload,
    RunFailedPayload,
    RunFinishedPayload,
    RunPausedPayload,
    RunPauseRequestedPayload,
    RunStartedPayload,
    RunStoppedPayload,
)
from zcu_tools.gui.app.autofluxdep.events.workflow import (
    FluxChangedPayload,
    WorkflowChangedPayload,
)
from zcu_tools.gui.app.autofluxdep.feedback import build_feedback_runtime
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
from zcu_tools.gui.app.autofluxdep.run_locks import (
    GuardedContextControl,
    GuardedDeviceControl,
    GuardedPredictorControl,
    GuardedSetupControl,
)
from zcu_tools.gui.app.autofluxdep.run_session import (
    RunSegmentOutcome,
    RunSession,
    RunSessionStatus,
)
from zcu_tools.gui.app.autofluxdep.services.persistence_types import (
    AppPersistedState,
    PersistedFluxSweep,
    PersistedNode,
    PersistedPredictorDialogState,
    PersistedPredictorModel,
    PersistedUiPrefs,
    PersistedWorkflow,
    PersistenceError,
    RestoreIssue,
    RestoreReport,
)
from zcu_tools.gui.app.autofluxdep.services.run_store import RunStore
from zcu_tools.gui.app.autofluxdep.services.sample_table_export import (
    SampleTableExportResult,
    export_sample_table_from_artifact,
)
from zcu_tools.gui.app.autofluxdep.services.sample_table_export import (
    default_sample_table_path as sample_table_default_path,
)
from zcu_tools.gui.app.autofluxdep.state import (
    FLUX_VERSION_KEY,
    WORKFLOW_VERSION_KEY,
    AutoFluxDepState,
    ProjectInfo,
)
from zcu_tools.gui.app.autofluxdep.tools import (
    FluxoniumPredictorAdapter,
    SimplePredictor,
    Tools,
)
from zcu_tools.gui.background import BackgroundRunner
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.event_bus import BasePayload
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.controller_mixin import SessionControllerMixin
from zcu_tools.gui.session.events import PredictorChangedPayload
from zcu_tools.gui.session.expression import coerce_eval_result, evaluate_numeric_expr
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
from zcu_tools.gui.session.services.build import build_session_services
from zcu_tools.gui.session.services.io_manager import IOManager
from zcu_tools.gui.session.services.predictor import (
    PredictorLoadError,
    SetModelParamsRequest,
)
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.gui.session.state import DEFAULT_LEFT_PANEL_WIDTH
from zcu_tools.meta_tool import QubitParams, QubitParamsError

if TYPE_CHECKING:
    from zcu_tools.gui.app.autofluxdep.services.caretaker import (
        PersistenceCaretaker,
        RestoreOutcome,
    )
    from zcu_tools.gui.session.adapters.qt_shutdown_driver import QtShutdownDriver
    from zcu_tools.gui.session.context_control import ContextControlPort
    from zcu_tools.gui.session.device_control import DeviceControlPort
    from zcu_tools.gui.session.ports import ProgressTransport
    from zcu_tools.gui.session.predictor_control import PredictorControlPort
    from zcu_tools.gui.session.progress_control import ProgressControlPort
    from zcu_tools.gui.session.services.startup import (
        ResolvedStartupProject,
        StartupProjectRequest,
    )
    from zcu_tools.gui.session.setup_control import SetupControlPort
    from zcu_tools.meta_tool import ModuleLibrary

logger = logging.getLogger(__name__)

RUN_PROGRESS_OWNER_ID = "autofluxdep-run"
FLUX_PROGRESS_LABEL = "flux sweep"
_PERSIST_DEBOUNCE_MS = 500


class _MlModuleSource:
    """Transparent ``ModuleLibrary`` proxy honouring the ``ModuleSource`` contract.

    The run resolver wants the orchestrator's ``ModuleSource`` contract:
    ``get_module(name)`` returns None if absent, so the dependency resolver can
    fall back to a Node-produced module or dependency default. A node cfg builder
    still wants the full ``ModuleLibrary`` surface (``get_waveform`` /
    ``make_cfg``), which should raise on a missing reference. This proxy serves
    both: it overrides only ``get_module``'s raise-on-absent behavior
    (``ModuleLibrary`` raises ``ValueError``) into None-on-absent, and forwards
    every other attribute to the wrapped library.
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


@dataclass(frozen=True)
class _RunReadiness:
    reason: str | None
    enabled_nodes: list[PlacedNode]
    flux_values: list[float] | None
    project: ProjectInfo | None


class _RunEventEmitter(QObject):
    """Deliver worker-originated run events on the controller's Qt thread."""

    point_done = Signal(int)
    node_entered = Signal(str, int)
    predictor_changed = Signal()

    def __init__(self, owner: Controller) -> None:
        super().__init__()
        self._owner = owner
        blocking = Qt.ConnectionType.BlockingQueuedConnection
        self.point_done.connect(owner._emit_point_done_on_main, type=blocking)  # type: ignore[call-arg]
        self.node_entered.connect(owner._emit_node_entered_on_main, type=blocking)  # type: ignore[call-arg]
        self.predictor_changed.connect(  # type: ignore[call-arg]
            owner._emit_predictor_changed_on_main,
            type=blocking,  # type: ignore[call-arg]
        )

    def emit_point_done(self, idx: int) -> None:
        if QThread.currentThread() == self.thread():
            self._owner._emit_point_done_on_main(idx)
            return
        self.point_done.emit(idx)

    def emit_node_entered(self, name: str, idx: int) -> None:
        if QThread.currentThread() == self.thread():
            self._owner._emit_node_entered_on_main(name, idx)
            return
        self.node_entered.emit(name, idx)

    def emit_predictor_changed(self) -> None:
        if QThread.currentThread() == self.thread():
            self._owner._emit_predictor_changed_on_main()
            return
        self.predictor_changed.emit()


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
        self._run_events = _RunEventEmitter(self)
        self._active_run_token: int | None = None
        self._run_session: RunSession | None = None
        self._last_run_info: InfoStore | None = None
        self._last_terminal_manifest_path: Path | None = None
        self._last_terminal_status: str | None = None
        self._caretaker: PersistenceCaretaker | None = None
        self._shutdown_driver: QtShutdownDriver | None = None
        self._persist_timer = QTimer()
        self._persist_timer.setSingleShot(True)
        self._persist_timer.timeout.connect(self.persist_all)

        # Base directory default result/database paths are anchored under (the
        # entry script injects the repo root). None falls back to cwd — fine for
        # tests / a `python -m` run from the repo root.
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
            on_project_applied=self._on_startup_project_applied,
        )
        self._session = session
        self._soc_svc = session.soc_connection
        self._pred_svc = session.predictor
        self._progress_control = session.progress_control
        self._ctx_svc = session.context
        self._dev_svc = session.device
        self._context_control = GuardedContextControl(
            session.context_control, self._require_session_mutable
        )
        self._device_control = GuardedDeviceControl(
            session.device_control, self._require_session_mutable
        )
        self._predictor_control = GuardedPredictorControl(
            session.predictor_control,
            self._require_session_mutable,
            on_mutated=self._schedule_persist_all,
        )
        self._setup_control = GuardedSetupControl(
            session.setup_control, self._require_session_mutable
        )
        self._startup_svc = session.startup

    # --- read-only accessors for the UI ---

    @property
    def state(self) -> AutoFluxDepState:
        return self._state

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def device_control(self) -> DeviceControlPort:
        return self._device_control

    @property
    def predictor_control(self) -> PredictorControlPort:
        return self._predictor_control

    @property
    def progress_control(self) -> ProgressControlPort:
        return self._progress_control

    @property
    def context_control(self) -> ContextControlPort:
        return self._context_control

    @property
    def setup_control(self) -> SetupControlPort:
        return self._setup_control

    @property
    def is_running(self) -> bool:
        return self._active_run_token is not None

    @property
    def is_paused(self) -> bool:
        session = self._run_session
        return session is not None and session.status is RunSessionStatus.PAUSED

    @property
    def run_status(self) -> str:
        session = self._run_session
        if session is None:
            return "idle"
        return session.status.value

    @property
    def next_flux_idx(self) -> int | None:
        session = self._run_session
        return None if session is None else session.next_flux_idx

    @property
    def active_run_token(self) -> int | None:
        """The operation token for the live RUN, or None when idle."""
        return self._active_run_token

    @property
    def last_run_info(self) -> InfoStore | None:
        """Testing/support entry: latest terminal sweep InfoStore, if produced."""
        return self._last_run_info

    def can_export_sample_table(self) -> bool:
        path = self._last_terminal_manifest_path
        return (
            self._last_terminal_status in {"finished", "stopped"}
            and path is not None
            and path.exists()
        )

    def default_sample_table_path(self) -> str | None:
        if not self.can_export_sample_table():
            return None
        assert self._last_terminal_manifest_path is not None
        return str(sample_table_default_path(self._last_terminal_manifest_path))

    def export_sample_table(
        self, filepath: str | None = None
    ) -> SampleTableExportResult:
        if not self.can_export_sample_table():
            raise RuntimeError("no finished or stopped autofluxdep run is exportable")
        assert self._last_terminal_manifest_path is not None
        return export_sample_table_from_artifact(
            self._last_terminal_manifest_path,
            filepath=filepath,
        )

    # ------------------------------------------------------------------
    # Shared setup/inspect/device/predictor dialogs use control facets. The
    # remaining identical compatibility forwards live in SessionControllerMixin
    # (read through the _*_svc accessors supplied above in __init__); only the
    # methods whose body diverges are kept here.
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
                enabled=node.enabled,
                cfg_raw=node.schema.to_persisted_raw(),
            )
            for node in self._state.nodes
        )
        return AppPersistedState(
            startup=self._startup_svc.capture_startup(
                left_panel_width=DEFAULT_LEFT_PANEL_WIDTH
            ),
            workflow=PersistedWorkflow(nodes=nodes),
            flux=PersistedFluxSweep(
                start_expr=self._state.flux_start_expr,
                stop_expr=self._state.flux_stop_expr,
                npts_expr=self._state.flux_npts_expr,
                values=tuple(float(v) for v in self._state.flux_values),
            ),
            predictor=self._capture_predictor_model(),
            ui=PersistedUiPrefs(
                auto_follow_tabs=self._state.auto_follow_tabs,
                predictor_dialog=self._state.predictor_dialog_state,
            ),
        )

    def _capture_predictor_model(self) -> PersistedPredictorModel | None:
        info = self._pred_svc.get_predictor_info()
        if info is None:
            return None
        path = info.get("path")
        return PersistedPredictorModel(
            EJ=self._finite_float(info["EJ"], "predictor.EJ"),
            EC=self._finite_float(info["EC"], "predictor.EC"),
            EL=self._finite_float(info["EL"], "predictor.EL"),
            flux_half=self._finite_float(info["flux_half"], "predictor.flux_half"),
            flux_period=self._finite_float(
                info["flux_period"], "predictor.flux_period"
            ),
            flux_bias=self._finite_float(info["flux_bias"], "predictor.flux_bias"),
            path=str(path) if path else None,
        )

    @staticmethod
    def _finite_float(value: object, field: str) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{field} must be a finite number")
        if not isinstance(value, Real):
            raise TypeError(f"{field} must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{field} must be finite")
        return numeric

    def restore_persisted_state(self, state: AppPersistedState) -> RestoreReport:
        """Apply a persisted workflow snapshot, rejecting only invalid nodes."""
        rejected: list[RestoreIssue] = []
        predictor_issue: RestoreIssue | None = None
        restored_predictor = False
        self._startup_svc.restore_startup(state.startup)
        self._state.nodes = []

        try:
            restored_predictor = self._restore_predictor_model(state.predictor)
        except (
            PredictorLoadError,
            TypeError,
            ValueError,
            KeyError,
            RuntimeError,
        ) as exc:
            self._pred_svc.clear_predictor()
            predictor_issue = RestoreIssue(subject="predictor", message=str(exc))

        for index, persisted in enumerate(state.workflow.nodes):
            subject = f"node[{index}] {persisted.name!r}"
            try:
                node = create_placement(persisted.type_name)
                node.name = self._unique_name(persisted.name or node.name)
                node.enabled = persisted.enabled
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
        self._state.auto_follow_tabs = state.ui.auto_follow_tabs
        self._state.predictor_dialog_state = state.ui.predictor_dialog
        self._commit_workflow_edit(changed_name=None)
        self._state.version.bump(FLUX_VERSION_KEY)
        self._bus.emit(FluxChangedPayload(count=len(self._state.flux_values)))
        return RestoreReport(
            restored_nodes=len(self._state.nodes),
            rejected_nodes=tuple(rejected),
            restored_predictor=restored_predictor,
            predictor_issue=predictor_issue,
        )

    def _restore_predictor_model(self, model: PersistedPredictorModel | None) -> bool:
        if model is None:
            self._pred_svc.clear_predictor()
            return False
        self._pred_svc.set_model_params(
            SetModelParamsRequest(
                EJ=self._finite_float(model.EJ, "predictor.EJ"),
                EC=self._finite_float(model.EC, "predictor.EC"),
                EL=self._finite_float(model.EL, "predictor.EL"),
                flux_half=self._finite_float(model.flux_half, "predictor.flux_half"),
                flux_period=self._finite_float(
                    model.flux_period, "predictor.flux_period"
                ),
                flux_bias=self._finite_float(model.flux_bias, "predictor.flux_bias"),
            )
        )
        return True

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
        if isinstance(report, RestoreReport) and report.predictor_issue is not None:
            logger.warning(
                "autofluxdep persistence rejected predictor: %s",
                report.predictor_issue.message,
            )
        return outcome

    def persist_all(self) -> None:
        self._persist_timer.stop()
        if self._caretaker is None:
            return
        try:
            self._caretaker.flush()
        except PersistenceError:
            logger.exception("autofluxdep settings save failed")

    def _schedule_persist_all(self) -> None:
        if self._caretaker is None:
            return
        self._persist_timer.start(_PERSIST_DEBOUNCE_MS)

    def _clear_run_products(self) -> None:
        self._state.run_results = {}
        self._state.run_predictor = None

    def _clear_terminal_sample_artifact(self) -> None:
        self._last_terminal_manifest_path = None
        self._last_terminal_status = None

    def _remember_terminal_sample_artifact(
        self, session: RunSession, status: str
    ) -> None:
        self._last_terminal_manifest_path = session.store.manifest_path
        self._last_terminal_status = status

    def _commit_workflow_edit(self, changed_name: str | None) -> None:
        self._clear_run_products()
        self._state.version.bump(WORKFLOW_VERSION_KEY)
        self._bus.emit(WorkflowChangedPayload(name=changed_name))
        self._schedule_persist_all()

    def clear_run_products(self) -> None:
        """UI start-failure recovery entry: clear run-lived products."""
        self._clear_run_products()

    def _require_workflow_editable(self) -> None:
        if self.is_running:
            raise RuntimeError("autofluxdep workflow is locked while a run is active")
        if self.is_paused:
            raise RuntimeError("autofluxdep workflow is locked while a run is paused")

    def _require_session_mutable(self, subject: str) -> None:
        if self.is_running:
            raise RuntimeError(f"autofluxdep {subject} is locked while a run is active")
        if self.is_paused:
            raise RuntimeError(f"autofluxdep {subject} is locked while a run is paused")

    # -- setup dialog: project / startup --
    # apply_startup_project diverges (autofluxdep returns bool; measure returns the
    # resolved-project dict per WIRE-48) so it stays a per-app override. get_bus /
    # get_project_root also stay per-app (app EventBus subtype / app state). Every
    # other setup-controller forward lives in SessionControllerMixin.
    def apply_startup_project(self, req: StartupProjectRequest) -> bool:
        self._require_workflow_editable()
        resolved = self._startup_svc.apply_project(req)
        self._on_startup_project_applied(resolved)
        return True

    def _on_startup_project_applied(self, project: ResolvedStartupProject) -> None:
        self._state.project = ProjectInfo(
            chip_name=project.chip_name,
            qub_name=project.qub_name,
            result_dir=project.result_dir,
            database_path=project.database_path,
            params_path=project.params_path,
        )
        self._try_auto_load_predictor_from_params(project)

    def _try_auto_load_predictor_from_params(
        self, project: ResolvedStartupProject
    ) -> None:
        params_path = Path(project.params_path)
        if not params_path.is_file():
            logger.debug(
                "autofluxdep predictor auto-load skipped: %s is not a file",
                params_path,
            )
            return
        try:
            fit = QubitParams(params_path, readonly=True).get_fluxdep_fit()
            if fit is None:
                logger.debug(
                    "autofluxdep predictor auto-load skipped: no fluxdep_fit in %s",
                    params_path,
                )
                return
            self._pred_svc.set_model_params(
                SetModelParamsRequest(
                    EJ=fit.EJ,
                    EC=fit.EC,
                    EL=fit.EL,
                    flux_half=fit.flux_half,
                    flux_period=fit.flux_period,
                )
            )
            self._schedule_persist_all()
        except (OSError, QubitParamsError, PredictorLoadError, ValueError) as exc:
            logger.warning(
                "autofluxdep predictor auto-load failed for %s: %s",
                params_path,
                exc,
            )
            return
        logger.info("autofluxdep predictor auto-loaded from %s", params_path)

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
        """Testing/support entry: add an ad-hoc Builder placement."""
        self._require_workflow_editable()
        name = self._unique_name(builder.name)
        node = PlacedNode(
            builder=builder,
            name=name,
            overrides=params,
            default_context=self._state.exp_context,
        )
        self._state.nodes.append(node)
        logger.debug("add_node: %r (type=%r) params=%s", name, builder.name, params)
        self._commit_workflow_edit(changed_name=node.name)
        return node

    def add_node_by_type(self, type_name: str) -> PlacedNode:
        """Add a fresh provider of ``type_name`` (from the registry) to the end.

        The instance name defaults to the type name, de-duped within the
        workflow (a second ``mist`` becomes ``mist_2``); the user can rename it.
        """
        self._require_workflow_editable()
        node = create_placement(type_name, ctx=self._state.exp_context)
        node.name = self._unique_name(node.name)
        self._state.nodes.append(node)
        logger.debug("add_node_by_type: %r -> %r", type_name, node.name)
        self._commit_workflow_edit(changed_name=node.name)
        return node

    def rename_node(self, index: int, new_name: str) -> str:
        """Rename the placement at ``index`` to a workflow-unique ``new_name``.

        Returns the actual name applied (de-duped + stripped). A blank name is
        rejected (kept unchanged) — fast-fail on the empty case rather than
        silently naming it the type. Used to distinguish repeated placements
        (e.g. two ``mist`` → ``g_mist`` / ``e_mist``).
        """
        self._require_workflow_editable()
        node = self._state.nodes[index]
        cleaned = new_name.strip()
        if not cleaned:
            logger.debug(
                "rename_node[%d]: blank name rejected, kept %r", index, node.name
            )
            return node.name  # blank → no-op, keep current name
        old = node.name
        node.name = self._unique_name(cleaned, exclude=node)
        if node.name == old:
            return node.name
        logger.debug("rename_node[%d]: %r -> %r", index, old, node.name)
        self._commit_workflow_edit(changed_name=node.name)
        return node.name

    def remove_node(self, name: str) -> None:
        self._require_workflow_editable()
        before = len(self._state.nodes)
        self._state.nodes = [n for n in self._state.nodes if n.name != name]
        if len(self._state.nodes) == before:
            return
        logger.debug("remove_node: %r (%d -> %d)", name, before, len(self._state.nodes))
        self._commit_workflow_edit(changed_name=name)

    def reorder(self, index: int, delta: int) -> int:
        """Move the Node at ``index`` by ``delta`` (±1). Returns the new index."""
        self._require_workflow_editable()
        nodes = self._state.nodes
        new_index = index + delta
        if not (0 <= index < len(nodes) and 0 <= new_index < len(nodes)):
            return index  # out of range → no-op
        nodes[index], nodes[new_index] = nodes[new_index], nodes[index]
        logger.debug("reorder: %d <-> %d", index, new_index)
        self._commit_workflow_edit(changed_name=None)
        return new_index

    def set_node_enabled(self, index: int, enabled: bool) -> None:
        """Toggle whether a placed node participates in future runs."""
        self._require_workflow_editable()
        node = self._state.nodes[index]
        enabled = bool(enabled)
        if node.enabled == enabled:
            return
        node.enabled = enabled
        logger.debug("set_node_enabled[%d] (%r): %s", index, node.name, enabled)
        self._commit_workflow_edit(changed_name=node.name)

    def set_node_params(self, index: int, params: Mapping[str, Any]) -> None:
        """Testing/support entry: write Node knob leaves into its schema SSOT.

        Each incoming key writes directly into the placement's own
        ``NodeCfgSchema`` (the per-placement value tree, the SSOT). A scalar value
        is coerced to the field's declared type; a ``SweepValue`` is accepted for a
        ``SweepSpec`` knob.
        An unknown key fast-fails — the form only renders declared knobs, so an
        undeclared key is a real typo, not a silent extra. State writes happen on
        the main thread (the UI calls this), preserving the State main-thread
        invariant; the workflow version bumps + a ``WorkflowChangedPayload`` fires
        so dependents refresh, exactly as before.
        """
        self._require_workflow_editable()
        node = self._state.nodes[index]
        for key, value in params.items():
            node.schema.set_field(key, value)  # fast-fails unknown key / wrong type
        logger.debug(
            "set_node_params[%d] (%r): keys=%s", index, node.name, list(params)
        )
        self._commit_workflow_edit(changed_name=node.name)

    def set_node_cfg_value(self, index: int, value: CfgSectionValue) -> None:
        """Replace one Node's complete cfg value tree from the typed form draft."""
        self._require_workflow_editable()
        node = self._state.nodes[index]
        node.schema.replace_value_tree(value)
        logger.debug("set_node_cfg_value[%d] (%r)", index, node.name)
        self._commit_workflow_edit(changed_name=node.name)

    def set_flux_values(self, values: list[float]) -> None:
        self._require_workflow_editable()
        self._state.flux_values = list(values)
        self._clear_run_products()
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
        self._schedule_persist_all()

    def set_flux_sweep_expressions(
        self, start_expr: str, stop_expr: str, npts_expr: str
    ) -> None:
        """Remember the user's editable flux sweep expressions."""
        self._require_workflow_editable()
        self._state.flux_start_expr = start_expr
        self._state.flux_stop_expr = stop_expr
        self._state.flux_npts_expr = npts_expr
        self._clear_run_products()
        self._state.version.bump(FLUX_VERSION_KEY)
        self._schedule_persist_all()

    def get_flux_sweep_expressions(self) -> tuple[str, str, str]:
        return (
            self._state.flux_start_expr,
            self._state.flux_stop_expr,
            self._state.flux_npts_expr,
        )

    def commit_flux_sweep(
        self, start_expr: str, stop_expr: str, npts_expr: str
    ) -> list[float]:
        """Resolve editable flux sweep expressions and commit explicit points."""
        self._require_workflow_editable()
        start = self._resolve_flux_expr("start", start_expr, float)
        stop = self._resolve_flux_expr("stop", stop_expr, float)
        npts = self._resolve_flux_expr("points", npts_expr, int)
        if not isinstance(npts, int):
            raise RuntimeError("Flux sweep points must be an integer")
        if npts < 1:
            raise RuntimeError("Flux sweep points must be at least 1")

        values = np.linspace(float(start), float(stop), npts).tolist()
        self._state.flux_start_expr = start_expr
        self._state.flux_stop_expr = stop_expr
        self._state.flux_npts_expr = npts_expr
        self._state.flux_values = values
        self._clear_run_products()
        self._state.version.bump(FLUX_VERSION_KEY)
        logger.debug(
            "commit_flux_sweep: n=%d range=[%g, %g]",
            len(values),
            values[0],
            values[-1],
        )
        self._bus.emit(FluxChangedPayload(count=len(values)))
        self._schedule_persist_all()
        return values

    def set_auto_follow_tabs(self, enabled: bool) -> None:
        """Persist the UI preference for run-time selection/tab auto-follow."""
        enabled = bool(enabled)
        if self._state.auto_follow_tabs == enabled:
            return
        self._state.auto_follow_tabs = enabled
        self._schedule_persist_all()

    def get_auto_follow_tabs(self) -> bool:
        return self._state.auto_follow_tabs

    def get_predictor_dialog_state(self) -> PersistedPredictorDialogState:
        return self._state.predictor_dialog_state

    def set_predictor_dialog_state(self, state: PersistedPredictorDialogState) -> None:
        if self._state.predictor_dialog_state == state:
            return
        self._state.predictor_dialog_state = state
        self._schedule_persist_all()

    def set_flux_device(self, name: str | None) -> None:
        """Designate which connected device the flux sweep is applied through.

        Its unit labels the flux axis (and the value is recorded into the run
        cfg's flux ``dev`` entry when a real acquire is wired). ``None`` clears the
        selection — the flux values are then bare numbers.
        """
        self._require_workflow_editable()
        self._state.flux_device_name = name or None
        logger.debug("set_flux_device: %r", self._state.flux_device_name)

    def get_flux_device(self) -> str | None:
        return self._state.flux_device_name

    def run_readiness(self) -> str | None:
        """UI readiness reason for the Run button; None means Run may be clicked."""
        return self._evaluate_run_readiness(
            require_setup=True,
            require_flux=False,
            start_messages=False,
        ).reason

    def _evaluate_run_readiness(
        self,
        *,
        require_setup: bool,
        require_flux: bool,
        start_messages: bool,
    ) -> _RunReadiness:
        enabled_nodes = self._enabled_nodes()
        project = self._state.project
        flux_values: list[float] | None = None
        if require_setup and not self._state.has_setup:
            return _RunReadiness("Setup required", enabled_nodes, None, project)
        if start_messages:
            if not enabled_nodes:
                return _RunReadiness(
                    "autofluxdep run requires at least one enabled node",
                    enabled_nodes,
                    None,
                    project,
                )
        else:
            if not self._state.nodes:
                return _RunReadiness(
                    "Add at least one node", enabled_nodes, None, project
                )
            if not enabled_nodes:
                return _RunReadiness(
                    "Enable at least one node", enabled_nodes, None, project
                )
        if require_flux:
            try:
                flux_values = self._current_flux_values_for_run()
            except RuntimeError as exc:
                return _RunReadiness(str(exc), enabled_nodes, None, project)
        if project is None:
            reason = (
                "autofluxdep run requires a configured project"
                if start_messages
                else "Project required"
            )
            return _RunReadiness(reason, enabled_nodes, flux_values, project)
        return _RunReadiness(None, enabled_nodes, flux_values, project)

    def _resolve_flux_expr(self, label: str, expr: str, type_: type) -> int | float:
        try:
            resolved = coerce_eval_result(
                evaluate_numeric_expr(expr.strip(), self.get_current_md()),
                type_,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Flux sweep {label} expression {expr!r}: {exc}"
            ) from exc
        if not math.isfinite(float(resolved)):
            raise RuntimeError(f"Flux sweep {label} value must be finite")
        return resolved

    def _current_flux_values_for_run(self) -> list[float]:
        values = list(self._state.flux_values)
        if not values:
            raise RuntimeError("autofluxdep run requires at least one flux point")
        for idx, value in enumerate(values):
            if not math.isfinite(float(value)):
                raise RuntimeError(
                    f"autofluxdep run flux point at index {idx} must be finite"
                )
        return values

    def _require_start_run_ready(
        self,
    ) -> tuple[list[PlacedNode], list[float], ProjectInfo]:
        readiness = self._evaluate_run_readiness(
            require_setup=False,
            require_flux=True,
            start_messages=True,
        )
        if readiness.reason is not None:
            raise RuntimeError(readiness.reason)
        if readiness.flux_values is None or readiness.project is None:
            raise RuntimeError("autofluxdep run readiness evaluation is incomplete")
        return readiness.enabled_nodes, readiness.flux_values, readiness.project

    # --- run control (cancellable) ---

    def stop_run(self, reason: str = "Autofluxdep run stop requested") -> bool:
        """Request terminal cancellation of an active or paused run session."""
        logger.info("stop_run requested (at flux idx %d)", self._cur_idx)
        token = self._active_run_token
        if token is not None:
            self._operation_handles.stop(token, reason=reason)
            return True
        if self.is_paused:
            self.finalize_paused_run_as_stopped()
            return True
        return False

    def finalize_paused_run_as_stopped(self) -> None:
        """Finalize the paused in-memory session and drop its continue state."""
        if not self.is_paused:
            raise RuntimeError("autofluxdep run is not paused")
        session = self._run_session
        assert session is not None
        try:
            session.finalize("stopped")
        except Exception as exc:
            logger.exception("autofluxdep paused artifact finalize failed")
            self._clear_terminal_sample_artifact()
            self._run_session = None
            self._bus.emit(RunFailedPayload(message=str(exc), stage="paused_finalize"))
            raise
        self._remember_terminal_sample_artifact(session, "stopped")
        self._run_session = None
        self.persist_all()
        self._bus.emit(RunStoppedPayload())

    def request_pause(self) -> bool:
        """Request a non-terminal pause at the next flux boundary."""
        session = self._run_session
        if session is None or self._active_run_token is None:
            return False
        if not session.request_pause():
            return False
        logger.info("pause requested (next boundary after flux idx %d)", self._cur_idx)
        self._bus.emit(RunPauseRequestedPayload())
        return True

    def continue_run(self) -> int:
        """Continue a paused in-memory RunSession from its original cursor."""
        session = self._run_session
        if session is None or session.status is not RunSessionStatus.PAUSED:
            raise RuntimeError("autofluxdep run is not paused")
        token = self._begin_run_segment(session, continuing=True)
        self._bus.emit(RunContinuedPayload(next_flux_idx=session.next_flux_idx))
        return token

    def _enabled_nodes(self) -> list[PlacedNode]:
        return [node for node in self._state.nodes if node.enabled]

    def _build_providers(self) -> list[PlacedNode]:
        """The execution sequence: the predictor Service prepended to the user's
        providers. The Service is loaded because qubit_freq requires
        ``predict_freq`` — it is not in the user's list, but it runs first each
        point so its ``predict_freq`` is this-point-available downstream."""
        service = PlacedNode(builder=PredictorBuilder())
        return [service, *self._enabled_nodes()]

    def _build_tools(self, providers: list[PlacedNode] | None = None) -> Tools:
        """Build the sweep's run-lived predictor and feedback capabilities.

        ``exp_context.predictor`` holds the raw ``FluxoniumPredictor`` (loaded at
        setup / by PredictorService) or None. A real predictor is wrapped into
        ``FluxoniumPredictorAdapter``; with none loaded we fall back to the
        base-only ``SimplePredictor`` stand-in. Feedback capabilities are built
        from placed-node declarations and generation policy.
        """
        raw = self._state.exp_context.predictor
        predictor = (
            FluxoniumPredictorAdapter(fluxonium=raw)
            if raw is not None
            else SimplePredictor()
        )
        feedback = build_feedback_runtime(
            providers or self._build_providers(),
            md=self._state.exp_context.md,
        )
        return Tools(predictor=predictor, feedback=feedback)

    def _emit_point_done_on_main(self, idx: int) -> None:
        self._cur_idx = idx
        self._bus.emit(PointDonePayload(idx=idx))

    def _emit_node_entered_on_main(self, name: str, idx: int) -> None:
        self._bus.emit(NodeEnteredPayload(name=name, idx=idx))

    def _emit_predictor_changed_on_main(self) -> None:
        self._bus.emit(PredictorChangedPayload())

    def _allocate_results(self, flux: Any) -> dict[str, Any]:
        """Pre-allocate each user provider's sweep Result on the main thread.

        ``flux`` is the full flux axis (filled into each Result up front for the
        Plotter's x). A Service (predictor) has no Result (``make_init_result``
        returns None), so it is absent from the map — the orchestrator curries no
        Result for it and the UI builds no figure, with no ``isinstance``.
        """
        results: dict[str, Any] = {}
        md = self._state.exp_context.md
        for node in self._enabled_nodes():
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
        if self.is_paused:
            raise RuntimeError("autofluxdep run is paused; continue or stop it first")

        self._cur_idx = 0
        self._last_run_info = None
        self._clear_terminal_sample_artifact()
        try:
            session = self._create_run_session(notify)
        except Exception:
            self._clear_run_products()
            raise
        token = self._begin_run_segment(session, continuing=False)
        self._bus.emit(RunStartedPayload())
        return token

    def _create_run_session(self, notify: Notify | None) -> RunSession:
        enabled_nodes, flux_values, project = self._require_start_run_ready()
        logger.info(
            "run start: %d enabled user Node(s) %s over %d flux point(s)",
            len(enabled_nodes),
            [node.name for node in enabled_nodes],
            len(flux_values),
        )

        ctx = self._state.exp_context
        ml = _MlModuleSource(ctx.ml)
        enabled_names = {node.name for node in enabled_nodes}
        if not self._state.run_results or not set(self._state.run_results).issubset(
            enabled_names
        ):
            self.prepare_run_results()
        results = self._state.run_results
        providers = self._build_providers()
        tools = self._build_tools(providers)
        self._state.run_predictor = tools.predictor
        flux_device = self._state.flux_device_name
        cfg_snapshots = self._build_run_cfg_snapshots(enabled_nodes)
        store = RunStore.create(
            project=project,
            flux_values=flux_values,
            flux_device_name=flux_device,
            nodes=enabled_nodes,
            results=results,
            cfg_snapshots={
                name: snapshot.to_wire() for name, snapshot in cfg_snapshots.items()
            },
        )
        return RunSession(
            providers=providers,
            user_nodes=enabled_nodes,
            flux_values=flux_values,
            flux_device=flux_device,
            results=results,
            cfg_snapshots=cfg_snapshots,
            store=store,
            tools=tools,
            ml=ml,
            soc=ctx.soc,
            soccfg=ctx.soccfg,
            md=ctx.md,
            notify=notify,
            event_sink=self._run_events,
            has_loaded_predictor=ctx.predictor is not None,
            progress_label=FLUX_PROGRESS_LABEL,
        )

    def _build_run_cfg_snapshots(
        self, enabled_nodes: list[PlacedNode]
    ) -> dict[str, RunCfgSnapshot]:
        ctx = self._state.exp_context
        snapshots: dict[str, RunCfgSnapshot] = {}
        for node in enabled_nodes:
            base_cfg = node.schema.lower_raw(ctx.ml, ctx.md)
            override_plan = node.builder.override_plan(node.schema)
            knobs = node.schema.lower(ctx.ml, ctx.md)
            validate_override_plan_base_cfg(
                override_plan,
                base_cfg,
                node_name=node.name,
            )
            snapshots[node.name] = RunCfgSnapshot(
                base_cfg=base_cfg,
                override_plan=override_plan,
                knobs=knobs,
            )
        return snapshots

    def _begin_run_segment(self, session: RunSession, *, continuing: bool) -> int:
        if self._active_run_token is not None:
            raise RuntimeError("autofluxdep run is already active")
        session.prepare_segment(continuing=continuing)
        self._run_session = session

        def on_terminal(bg: BgResult, settle: SettleFn) -> None:
            self._on_run_segment_terminal(session, bg, settle)

        spec = OperationSpec(
            exclusion=ExclusionRequest(
                kind=OperationKind.RUN,
                owner_id=RUN_PROGRESS_OWNER_ID,
            ),
            owner_id=RUN_PROGRESS_OWNER_ID,
            wants_progress=True,
            cancel_hook=session.request_stop,
            work=session.start_or_continue,
            run_in_pool=False,
            on_terminal=on_terminal,
        )
        try:
            token = self._runner.begin(spec)
        except Exception as exc:
            if continuing:
                self._run_session = session
                raise
            self._clear_run_products()
            self._run_session = None
            self._finalize_operation_begin_failure(session, exc)
            raise
        self._active_run_token = token
        return token

    def _finalize_operation_begin_failure(
        self, session: RunSession, error: Exception
    ) -> None:
        try:
            session.store.record_run_failed(error, stage="operation_begin")
            session.finalize("failed", error=error)
        except Exception as finalize_exc:
            logger.exception("autofluxdep artifact finalize failed before RUN begin")
            raise RuntimeError(
                f"{error}; artifact finalize failed: {finalize_exc}"
            ) from error

    def _clear_active_run(self, *, release_session: bool) -> None:
        self._active_run_token = None
        if release_session:
            self._run_session = None

    def _prepare_run_segment_terminal(self, *, release_session: bool) -> None:
        self._clear_active_run(release_session=release_session)
        self.persist_all()

    def _settle_run_segment(
        self, settle: SettleFn, outcome: OperationOutcome, payload: BasePayload
    ) -> None:
        settle(outcome)
        self._bus.emit(payload)

    def _finalize_terminal_session(
        self,
        session: RunSession,
        status: str,
        *,
        error: Exception | None = None,
        node: str | None = None,
        flux_idx: int | None = None,
        stage: str | None = None,
    ) -> Exception | None:
        try:
            if status == "failed":
                if error is None:
                    raise RuntimeError("failed run finalization requires an error")
                session.store.record_run_failed(
                    error,
                    flux_idx=flux_idx,
                    node=node,
                    stage=stage,
                )
                session.finalize("failed", error=error)
                return None
            session.finalize(status)
            return None
        except Exception as exc:
            return exc

    def _on_run_segment_terminal(
        self, session: RunSession, bg: BgResult, settle: SettleFn
    ) -> None:
        if bg.ok:
            outcome = bg.result
            if not isinstance(outcome, RunSegmentOutcome):
                self._on_run_failed(
                    session,
                    RuntimeError(f"autofluxdep run returned {type(outcome).__name__}"),
                    settle,
                )
                return
            self._last_run_info = outcome.info
            if outcome.run_error is not None:
                detail = outcome.run_error
                self._on_run_failed(
                    session,
                    detail.error,
                    settle,
                    node=detail.node,
                    flux_idx=detail.flux_idx,
                    stage=detail.stage,
                )
            elif outcome.stopped:
                self._on_run_stopped(session, settle)
            elif outcome.paused:
                self._on_run_paused(session, outcome.next_flux_idx, settle)
            else:
                self._on_run_finished(session, settle)
            return

        assert bg.error is not None
        if session.stop_requested:
            self._on_run_stopped(session, settle)
        else:
            self._on_run_failed(session, bg.error, settle)

    def _on_run_finished(self, session: RunSession, settle: SettleFn) -> None:
        logger.info("run finished: %d flux point(s)", len(session.flux_values))
        self._prepare_run_segment_terminal(release_session=True)
        if (exc := self._finalize_terminal_session(session, "finished")) is not None:
            self._clear_terminal_sample_artifact()
            logger.error(
                "autofluxdep artifact finalize failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            self._settle_run_segment(
                settle,
                OperationOutcome("failed", str(exc)),
                RunFailedPayload(message=str(exc), stage="finalize"),
            )
            return
        self._remember_terminal_sample_artifact(session, "finished")
        self._settle_run_segment(
            settle, OperationOutcome("finished"), RunFinishedPayload()
        )

    def _on_run_paused(
        self, session: RunSession, next_flux_idx: int, settle: SettleFn
    ) -> None:
        logger.info("run paused before flux idx %d", next_flux_idx)
        self._prepare_run_segment_terminal(release_session=False)
        try:
            session.mark_paused()
        except Exception as exc:
            logger.exception("autofluxdep paused artifact flush failed")
            self._on_run_failed(session, exc, settle, stage="pause_flush")
            return
        self._settle_run_segment(
            settle,
            OperationOutcome("cancelled"),
            RunPausedPayload(next_flux_idx=next_flux_idx),
        )

    def _on_run_stopped(self, session: RunSession, settle: SettleFn) -> None:
        logger.info("run stopped at flux idx %d", self._cur_idx)
        self._prepare_run_segment_terminal(release_session=True)
        if (exc := self._finalize_terminal_session(session, "stopped")) is not None:
            self._clear_terminal_sample_artifact()
            logger.error(
                "autofluxdep stopped artifact finalize failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            self._settle_run_segment(
                settle,
                OperationOutcome("failed", str(exc)),
                RunFailedPayload(message=str(exc), stage="stop_finalize"),
            )
            return
        self._remember_terminal_sample_artifact(session, "stopped")
        self._settle_run_segment(
            settle, OperationOutcome("cancelled"), RunStoppedPayload()
        )

    def _on_run_failed(
        self,
        session: RunSession,
        error: Exception,
        settle: SettleFn,
        *,
        node: str | None = None,
        flux_idx: int | None = None,
        stage: str | None = None,
    ) -> None:
        logger.error("run failed at flux idx %d: %s", self._cur_idx, error)
        self._prepare_run_segment_terminal(release_session=True)
        self._clear_terminal_sample_artifact()
        if (
            exc := self._finalize_terminal_session(
                session,
                "failed",
                error=error,
                node=node,
                flux_idx=flux_idx,
                stage=stage,
            )
        ) is not None:
            logger.error(
                "autofluxdep failed artifact finalize failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            error = RuntimeError(f"{error}; artifact finalize failed: {exc}")
        self._settle_run_segment(
            settle,
            OperationOutcome("failed", str(error)),
            RunFailedPayload(
                message=str(error),
                node=node,
                flux_idx=flux_idx,
                stage=stage,
            ),
        )

    def await_operation(self, operation_id: int, timeout: float) -> AwaitResult | None:
        """Testing/support entry: block until an operation settles or wakes."""
        return self._operation_handles.await_outcome(operation_id, timeout)

    def get_operation_progress(self, operation_id: int) -> tuple:
        """Testing/support entry: live progress bars for one operation id."""
        return self._progress_svc.bars_for_operation(operation_id)

    def active_operation_count(self) -> int:
        """How many shared operation handles are live right now."""
        return self._operation_handles.live_count()

    def begin_shutdown(self, on_closed: Callable[[], None]) -> None:
        """Cancel live operations, wait for terminal outcomes, then close the view."""
        if self._shutdown_driver is None:
            from zcu_tools.gui.session.adapters.qt_shutdown_driver import (
                QtShutdownDriver,
            )

            self._shutdown_driver = QtShutdownDriver(self._operation_handles)
        self._shutdown_driver.begin(on_closed)

    def quiesce_background(self, timeout_ms: int = 5000) -> bool:
        """Drain background workers and queued terminal callbacks before teardown."""
        drained = self._background_svc.quiesce(timeout_ms=timeout_ms)
        if not drained:
            logger.warning("autofluxdep background runner did not quiesce before close")
        return drained

    def prepare_run_results(self) -> dict[str, Any]:
        """Allocate + store this run's Results in State (main thread, Run start).

        The UI calls this before starting the worker so the Plotters bind to the
        same Result objects the worker fills. Returns the name→Result map.
        """
        flux = np.asarray(self._current_flux_values_for_run(), dtype=np.float64)
        self._state.run_results = self._allocate_results(flux)
        return self._state.run_results

    # --- dry run (headless: real orchestrator, no Results / notify) ---

    def dry_run(
        self,
        tools: Tools | None = None,
        ml: ModuleSource | None = None,
    ) -> InfoStore:
        """Testing/support entry: exercise the dependency model headless.

        Runs the same orchestrator over the
        same providers (predictor Service prepended), but with no Results and no
        notify (nothing to draw). ``tools`` defaults to a fresh ``Tools`` (a
        SimplePredictor if none bound); ``ml`` is the module-library fallback;
        smoothing is auto-built from declarations. Returns the final InfoStore.
        Providers run in list order (no topo sort)."""
        enabled_nodes = self._enabled_nodes()
        providers = self._build_providers()
        ctx = self._state.exp_context
        run_ml = ml
        if run_ml is None and ctx.ml is not None:
            run_ml = _MlModuleSource(ctx.ml)
        cfg_snapshots = self._build_run_cfg_snapshots(enabled_nodes)
        orch = Orchestrator(
            providers=providers,
            tools=tools or self._build_tools(providers),
            ml=run_ml,
            md=ctx.md,
            cfg_snapshots=cfg_snapshots,
        )
        return orch.run(self._current_flux_values_for_run())
