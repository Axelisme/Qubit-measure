"""Concrete T1 form, captured startup and Qt plotting composition."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import get_ident
from typing import TYPE_CHECKING, Literal

from pydantic import JsonValue
from qick import QickConfig
from qtpy.QtCore import QSignalBlocker, QTimer
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QFileDialog, QWidget
from zcu_tools.datafile import RunSnapshot, SoftwareProvenance
from zcu_tools.device import DeviceInfo
from zcu_tools.experiment.acquisition import capture_software_provenance
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows import (
    Actor,
    EnginePorts,
    FluxdepExportReport,
    InvalidRunState,
    RunIdentity,
    RunPaths,
    RunStatus,
    TunablesSnapshot,
)
from zcu_tools.experiment.workflows.clock import UtcClock
from zcu_tools.gui.app.workflows.controller import WorkflowController
from zcu_tools.gui.app.workflows.runtime import SessionDevices
from zcu_tools.gui.catalog_loader import CatalogReloadError, SourceCatalogLoader
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormModel,
    FormWritePath,
)
from zcu_tools.gui.plotting.container import FigureContainer
from zcu_tools.gui.plotting.explicit import QtPlotHost
from zcu_tools.gui.remote.control_service import ControlOptions
from zcu_tools.gui.session.context_types import ContextSourceUnavailableError
from zcu_tools.gui.session.persistence import PersistenceError
from zcu_tools.gui.session.ports import OwnerScheduler
from zcu_tools.gui.session.services.build import SessionServices
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.ui.alias_dialog import AliasDialog
from zcu_tools.gui.session.ui.setup_dialog import SetupDialog
from zcu_tools.plotting.figures import NamedFigures
from zcu_tools.plotting.plots import Plots
from zcu_tools.progress_bar.backend.tqdm import TQDMProgressBar
from zcu_tools.resources.entry import (
    Origin,
    RoleResolutionError,
    new_run_id,
)
from zcu_tools.resources.entry.save_layout import workflow_roots

from ._form_io import inspect_form_input, prepare_start_form
from .catalog import WorkflowCatalog, build_catalog
from .figures import WorkflowFigureViews
from .forms import WorkflowForms
from .layout import build_workflow_layout
from .persistence import WorkflowInputRestoreReport, WorkflowInputSnapshot
from .remote import FormInspectionReply, T1RemoteControlAdapter
from .tuning_panel import T1TuningPanel

if TYPE_CHECKING:
    from ..calibration import Calibrated
    from .panel import WorkflowFormPanel, WorkflowPlanPanel

logger = logging.getLogger(__name__)


class T1WorkflowWindow(QWidget):
    """Compose one workflow selector, captured Start and tuning on the Qt owner.

    plan_panel/tunables_panel expose selected independent parsed drafts.
    controller stays fixed; its engine is None until native acceptance, and
    tuning_panel presents its unique run binding. SessionServices owns shared
    runner/progress, hardware, exclusion and workers; parent permits embedding.
    RPC calls these same controls. Start uses cached native info; every env.run
    captures live drivers in the worker. Pause retains plan/plots/input capture;
    Resume keeps the same operation, without recapture or a second Engine owner.
    """

    def __init__(
        self,
        state: SessionState,
        services: SessionServices,
        owner: OwnerScheduler,
        *,
        catalog: WorkflowCatalog | None = None,
        loader: SourceCatalogLoader[WorkflowCatalog] | None = None,
        on_close: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Build a Window on the GUI owner with an existing QApplication.

        state belongs to services; owner marshals plot commands. parent=None is
        top-level; otherwise embedded. EMPTY builds partial source-free drafts.
        catalog=None builds T1/overnight, T1 first. loader=None is unmonitored.
        Reload keeps session/Controller and terminal export bindings. Prepare
        captures context/global qubit alias; source errors propagate without I/O.
        Caller retains this Window/parent through shutdown. Close requests Stop,
        waits for delivery, then calls on_close on owner; None skips caller cleanup.
        Callback errors reject close, display diagnostics and retain retry resources.
        """
        super().__init__(parent)
        current = catalog if catalog is not None else build_catalog()
        self._loader = loader
        self._thread = get_ident()
        self._forms_for_run = False
        self._catalog_error: CatalogReloadError | None = None
        self._state = state
        self._services = services
        self._runner = services.runner
        self._owner = owner
        self._progress = services.progress
        self._on_close = on_close
        self.controller: WorkflowController[FormModel, Calibrated] = WorkflowController(
            services.runner, "t1-workflow", on_finished=self._finished
        )
        self.tuning_panel: T1TuningPanel[FormModel, Calibrated] | None = None

        env = (
            self._capture_env(self._services.device.capture_native_device_info())
            if state.session_env.parameters is not None
            else FormEnv()
        )
        self._forms = WorkflowForms(current, env, self)
        self._controls = build_workflow_layout(
            self, self.plan_panel, self.tunables_panel
        )
        self._controls.selector.addItems(list(current.names))
        self._controls.selector.currentTextChanged.connect(self._selection_changed)
        for button, slot in (
            (self._controls.start, self._start_clicked),
            (self._controls.pause, self._pause_clicked),
            (self._controls.resume, self._resume_clicked),
            (self._controls.stop, self._stop_clicked),
            (self._controls.export, self._export_clicked),
            (self._controls.open_trend, self._open_trend_clicked),
            (self._controls.setup, self.open_setup),
            (self._controls.aliases, self._aliases_clicked),
            (self._controls.prepare, self._use_source_clicked),
            (self._controls.reload, self._reload_clicked),
        ):
            button.clicked.connect(slot)
        self._controls.reload.setEnabled(self._loader is not None)
        self._plot_host = QtPlotHost(
            FigureContainer(self._controls.plot_stack, self._controls.plot_placeholder),
            owner,
        )
        self._figure_views = WorkflowFigureViews(self._plot_host)
        self.setup_dialog = SetupDialog(self._services.setup_control, self)
        self.setup_dialog.finished.connect(self._sync_source_status)
        self._sync_source_status()
        self._poll = QTimer(self)
        self._poll.setInterval(50)
        self._poll.timeout.connect(self._refresh_run)

    setup_dialog: SetupDialog

    def open_setup(self) -> None:
        """Show the same retained shared Setup non-modally on the GUI owner.

        Reopening refreshes the dialog's cached inputs, not form captures. No
        entry is opened or hardware connected by showing. Closing Setup updates
        the global qubit alias availability without replacing form drafts.
        """
        self.setup_dialog.show()
        self.setup_dialog.raise_()
        self.setup_dialog.activateWindow()

    def _aliases_clicked(self) -> None:
        try:
            dialog = AliasDialog(self._services.context_control, self)
        except Exception as error:
            logger.exception("T1 workflow alias editor could not open")
            self._controls.message.setText(str(error))
            return
        dialog.finished.connect(self._sync_source_status)
        dialog.finished.connect(dialog.deleteLater)
        dialog.show()

    def _sync_source_status(self, *_args: object) -> None:
        try:
            parameters = self._services.context.read_context()
            source = parameters.resolve_path("$qubit.kind")
        except FailedPreconditionError:
            # EMPTY is expected at fresh app startup; status must not block it.
            self._controls.source_status.setText("Qubit: no context")
        except (RoleResolutionError, ContextSourceUnavailableError) as error:
            self._controls.source_status.setText(f"Qubit unavailable: {error}")
        else:
            self._controls.source_status.setText(f"Qubit: {source}")

    def _use_source_clicked(self) -> None:
        try:
            self.prepare_forms()
        except Exception as error:
            logger.exception("T1 workflow source capture failed")
            self._controls.message.setText(str(error))

    def prepare_forms(self) -> None:
        """Replace both Start drafts and bases with current captured defaults.

        Call on the GUI owner with a readable global qubit alias.
        Capture parameters/library and cached device values once, then replace
        this Window's form attachments. Previous edits and local parse text are
        discarded on success; subsequent source changes do not update the forms.
        No live device reads, submission, storage write or input reload occurs.
        EMPTY raises FailedPreconditionError; missing/unbound qubit raises the
        base RoleResolutionError naming that alias. No component kind is checked.
        Source/editor failures propagate without replacing the existing panels.
        Pending, running or paused operations raise InvalidRunState before capture.
        """
        self._ensure_can_start()
        env = self._capture_env(self._services.device.capture_native_device_info())
        if env.parameters is None:
            raise FailedPreconditionError("No parameter context is installed")
        env.parameters.resolve_path("$qubit.kind")
        old = self.plan_panel, self.tunables_panel
        self._forms.prepare(env, self)
        self._publish_panels(*old)

    @property
    def plan_panel(self) -> WorkflowPlanPanel:
        """Return the selected attachment's current parsed Start plan view."""
        return self._forms.plan

    @property
    def tunables_panel(self) -> WorkflowFormPanel[FormModel]:
        """Return the selected attachment's current parsed Start knobs view."""
        return self._forms.knobs

    @property
    def workflow_name(self) -> str:
        """Return the native name selected by the sole GUI/MCP catalog."""
        return self._forms.attachment.name

    @property
    def default_workflow_name(self) -> str:
        """Return the catalog's default native name for Start without RPC name."""
        return self._forms.catalog.select().name

    def select_workflow(self, name: str) -> None:
        """Select current native/form attachments, retaining independent parsed drafts.

        Use on the GUI owner before Start or after terminal delivery. Unknown
        names raise InvalidInputError; pending/running/paused ownership rejects before
        changes. Existing terminal run/export binding stays with Controller.
        Selection reuses fixed form captures, without device/source reads. Local
        unparsed text is not saved. No run/operation/lease is created or replaced.
        """
        self._ensure_can_start()
        if name == self.workflow_name:
            return
        old = self.plan_panel, self.tunables_panel
        self._forms.select(name, self.capture_persisted_state(), self)
        self._publish_panels(*old)
        self._sync_selector()

    def _sync_selector(self) -> None:
        with QSignalBlocker(self._controls.selector):
            self._controls.selector.clear()
            self._controls.selector.addItems(list(self._forms.catalog.names))
            self._controls.selector.setCurrentText(self.workflow_name)

    def _selection_changed(self, name: str) -> None:
        try:
            self.select_workflow(name)
        except Exception as exc:
            self._sync_selector()
            logger.exception("Workflow selection failed")
            self._controls.message.setText(str(exc))

    def _publish_panels(
        self, old_plan: WorkflowPlanPanel, old_tunables: WorkflowFormPanel[FormModel]
    ) -> None:
        self._controls.layout.replaceWidget(old_plan, self.plan_panel)
        self._controls.layout.replaceWidget(old_tunables, self.tunables_panel)
        for panel in (old_plan, old_tunables):
            panel.close()
            panel.deleteLater()
        self.plan_panel.show()
        self.tunables_panel.show()
        self._forms_for_run = False
        self._close_tuning_panel()

    def _close_tuning_panel(self) -> None:
        if self.tuning_panel is not None:
            self.tuning_panel.close()
            self.tuning_panel.deleteLater()
            self.tuning_panel = None

    def reload_source(self) -> WorkflowInputRestoreReport:
        """Replace current declarations and compatible parsed drafts, on owner thread.

        Pending/running/paused ownership rejects before source preflight/import.
        loader=None rejects unmonitored embedding. Success replaces only form
        panels/catalog, retains a still-declared selection or chooses the new
        default, and reports changed names/dropped paths in the message label.
        services/session/controller/hardware and old run export
        binding remain unchanged. Local unparsed text is not a saved draft.
        Failure retains parsed drafts and old panels but makes Start unavailable
        until successful retry. CatalogReloadError.restart_required forbids retry
        without restarting this app. Imports are bounded, not a rollback of
        arbitrary Python side effects. Replacement errors are translated into
        CatalogReloadError and shown without publishing a partial panel pair.
        """
        if get_ident() != self._thread:
            raise CatalogReloadError(
                "Reload must run on its owner thread", restart_required=True
            )
        self._ensure_idle()
        if self._loader is None:
            raise CatalogReloadError("Source reload is not configured")
        if self._catalog_error is not None and self._catalog_error.restart_required:
            raise self._catalog_error
        snapshot = self.capture_persisted_state()
        old_name = self.workflow_name
        old = self.plan_panel, self.tunables_panel
        try:
            candidate = self._loader.load(self._loader.prepare())
            report = self._forms.reload(candidate, snapshot, self)
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, CatalogReloadError)
                else CatalogReloadError(f"Workflow catalog replacement failed: {exc}")
            )
            self._catalog_error = error
            self._controls.start.setEnabled(False)
            self._controls.reload.setEnabled(not error.restart_required)
            self._controls.message.setText(str(error))
            if error is exc:
                raise
            raise error from exc
        self._publish_panels(*old)
        self._sync_selector()
        self._catalog_error = None
        self._controls.start.setEnabled(True)
        self.show_input_restore_message(report)
        if old_name != self.workflow_name:
            notice = f"Reload could not find {old_name}; selected {self.workflow_name}."
            details = self._controls.message.text()
            self._controls.message.setText(f"{notice} {details}" if details else notice)
        return report

    def _reload_clicked(self) -> None:
        try:
            self.reload_source()
        except Exception as exc:
            logger.exception("Workflow source reload failed")
            self._controls.message.setText(str(exc))

    def capture_persisted_state(self) -> WorkflowInputSnapshot:
        """Capture parsed plan/knobs input on the owner thread, without I/O.

        Expressions stay as entered, not their evaluated values. Plan comes
        from the Start panel. Once a run is attached, knobs come from its unique
        Controller binding, including unsubmitted parsed edits. Renderer-local
        incomplete text, native values and journal/hardware state are excluded.
        """
        binding = self.controller.binding
        knobs = (
            binding.state.input.data
            if binding is not None and self._forms_for_run
            else self.tunables_panel.state.input.data
        )
        return WorkflowInputSnapshot(plan=self.plan_panel.state.input.data, knobs=knobs)

    def restore_persisted_state(
        self, state: WorkflowInputSnapshot
    ) -> WorkflowInputRestoreReport:
        """Overlay saved state on fresh defaults, resetting both panel bases.

        Call on the owner thread after construction, before starting a run.
        Pending/running/paused ownership raises InvalidRunState before edits.
        FormState owns compatibility and returns discarded model-field paths;
        expressions remain input even when their result is invalid. Defaults or
        source failures propagate without publishing either candidate. Successful
        restoration refreshes both editors and clears renderer-local text.
        """
        self._ensure_can_start()
        return self._forms.restore(state)

    def show_input_restore_message(
        self,
        report: WorkflowInputRestoreReport,
        load_error: PersistenceError | None = None,
    ) -> None:
        """Present read failure or discarded fields in the existing owner label.

        report is the completed restore outcome. load_error is the caretaker's
        read/decode failure, if any, which already selected fresh defaults.
        No dialog, retry, state mutation or I/O occurs. Empty successful outcomes
        clear this label. Call only on the GUI owner thread.
        """
        messages = [str(load_error)] if load_error is not None else []
        for form, paths in (
            ("plan", report.plan_dropped),
            ("knobs", report.knobs_dropped),
        ):
            messages.extend(
                f"Discarded {form}: {'.'.join(str(part) for part in path)}"
                for path in paths
            )
        self._controls.message.setText("; ".join(messages))

    def _capture_env(self, devices: Mapping[str, DeviceInfo]) -> FormEnv:
        context = self._services.context
        parameters = context.read_context()
        return FormEnv(
            parameters=parameters, library=parameters.library, devices=devices
        )

    def _ensure_can_start(self) -> None:
        if self._catalog_error is not None:
            raise CatalogReloadError(
                "Workflow source is unavailable; reload successfully or restart",
                restart_required=self._catalog_error.restart_required,
            )
        self._ensure_idle()

    def _ensure_idle(self) -> None:
        """Reject replacement until the native run and owner delivery both end."""
        status = self.controller.status()
        if (
            self.controller.has_pending_segment
            or self.controller.operation_id is not None
        ):
            raise InvalidRunState("start", "running")
        if status is not None and status.lifecycle in {
            "running",
            "pausing",
            "paused",
            "stopping",
        }:
            raise InvalidRunState("start", status.lifecycle)

    def start(
        self,
        run_id: str,
        *,
        plan_batch: Mapping[FormWritePath, JsonValue] | None = None,
        tunables_batch: Mapping[FormWritePath, JsonValue] | None = None,
        name: str | None = None,
    ) -> int | FormErrors:
        """Capture/validate and submit the selected workflow, returning token or errors.

        run_id is a nonempty path-safe segment for exact directories. name=None
        uses GUI selection; a name selects the same catalog first, retaining
        independent parsed drafts. Each None batch reads its panel/local errors;
        explicit batches (including {}) use defaults in this Start's fixed capture
        plus one FormState.write, not GUI draft. Paths are model-field tuple leaves;
        containers are whole leaves and branch switches use shared core rules.
        Invalid input returns FormErrors without applying batches or submitting;
        structural errors use path (). Success resets both panel bases and clears
        local text in that same capture; expressions stay in accepted input.
        Cached native device info is used for forms; no driver live read occurs.
        Missing storage/hardware, source/provenance and native failures propagate,
        without retry/fallback. Pending/active/paused ownership rejects replacement.
        T1 uses validated pi estimate/module scalars; overnight context is None.
        The token is pending submission, not native acceptance.
        """
        self._ensure_can_start()
        if name is not None:
            self.select_workflow(name)
        attachment = self._forms.attachment
        env = self._capture_env(self._services.device.capture_native_device_info())
        if env.parameters is None:
            raise ValueError("Workflow Start requires captured parameters")
        storage = self._services.context.capture_storage(env.parameters)
        prepared_plan = prepare_start_form(
            attachment.plan_model,
            self.plan_panel.state,
            self.plan_panel.editor,
            plan_batch,
            env,
        )
        if isinstance(prepared_plan, FormErrors):
            return prepared_plan
        form, submitted_plan = prepared_plan
        prepared_tunables = prepare_start_form(
            attachment.knobs_model,
            self.tunables_panel.state,
            self.tunables_panel.editor,
            tunables_batch,
            env,
        )
        if isinstance(prepared_tunables, FormErrors):
            return prepared_tunables
        tunables_form, accepted_input = prepared_tunables
        snapshot = RunSnapshot(
            entry_id=storage.entry_id,
            entry_name=storage.entry_name,
            point=storage.point,
            description=storage.description,
            roles=storage.roles,
        )
        prepared = attachment.prepare(
            form, tunables_form, accepted_input, env, snapshot
        )
        if isinstance(prepared, FormErrors):
            return prepared
        session = self._state.session_env
        if session.soc is None or not isinstance(session.soccfg, QickConfig):
            raise ValueError("T1 Start requires connected SoC and native QickConfig")
        roots = workflow_roots(
            result_path=storage.result_path,
            database_path=storage.database_path,
            run_id=run_id,
        )
        plots = Plots(self._plot_host)
        ports, provenance = self._start_ports(
            session.soc, session.soccfg, plots, prepared.context
        )
        identity = RunIdentity(run_id, provenance, str(storage.entry_id), storage.point)
        export_binding = prepared.bind_export(storage, identity, roots)
        # Install presentation before synchronous admission listeners can reenter.
        self._close_tuning_panel()
        self._figure_views.install_native(plots)
        try:
            token = prepared.submit(
                self.controller,
                ports,
                RunPaths(metadata_root=roots.result, data_root=roots.database),
                identity,
                env,
                export_binding,
            )
        except Exception:
            self._figure_views.start_failed()
            raise
        self._figure_views.start_succeeded()
        self._controls.open_trend.setEnabled(False)
        self.plan_panel.state.reset(submitted_plan)
        self.tunables_panel.state.reset(accepted_input)
        for editor in (self.plan_panel.editor, self.tunables_panel.editor):
            editor.refresh(env)
            editor.discard_local_text()
        self.plan_panel.setEnabled(False)
        self.tunables_panel.setEnabled(False)
        self._controls.selector.setEnabled(False)
        self._controls.start.setEnabled(False)
        self._controls.export.setEnabled(False)
        self._controls.message.setText("Startup pending")
        self._forms_for_run = True
        self._poll.start()
        return token

    def _start_ports(
        self,
        soc: object,
        soccfg: QickConfig,
        plots: Plots,
        seed: Calibrated | None,
    ) -> tuple[EnginePorts[Calibrated], SoftwareProvenance]:
        # Native handles were validated before roots/Plots allocation in Start.
        devices = self._services.device.get_connected_devices()
        context = RunContext(soc, soccfg, plots, devices, StopSignal())
        provenance = capture_software_provenance(context=context)
        ports = EnginePorts(
            plots=plots,
            progress=TQDMProgressBar,
            clock=UtcClock(),
            context=seed,
            soc=soc,
            soccfg=soccfg,
            devices=SessionDevices(self._services.device),
        )
        return ports, provenance

    def create_remote_service(
        self, opts: ControlOptions, bus: BaseEventBus
    ) -> T1RemoteControlAdapter:
        """Create an inert RPC adapter of these controls on this Window's owner.

        opts configures shared transport; bus must be this Window's session bus.
        The composition caller owns adapter.start()/stop() and stops the socket
        before destroying this Window. This call does not open a socket, capture
        new inputs or touch hardware. Remote Stop requires explicit run_id;
        stopping the transport does not cancel a native run.
        """
        return T1RemoteControlAdapter(self, opts, bus=bus, owner_scheduler=self._owner)

    def inspect_form(
        self,
        form: Literal["plan", "tunables"],
        *,
        run_id: str | None = None,
        batch: Mapping[FormWritePath, JsonValue] | None = None,
    ) -> FormInspectionReply | FormErrors:
        """Read or preview this Window's form on the GUI owner, without I/O.

        form selects plan or tunables. Before any attached run, run_id must be
        omitted; afterwards require its explicit native identity, with the same
        Controller target checks as other controls. Pending acceptance raises
        RuntimeError. Use binding's fixed env and unique knobs state for a run,
        constructor panel captures otherwise. revision is native, never a guard.

        batch=None reads current input and includes local parse errors. A batch
        (even {}) previews current parsed input plus edits in a call-local
        FormState; local text is not candidate input. Structural/source write
        errors return FormErrors, model errors are in the reply. No input/base,
        accepted input, local text, Engine state, lease or source is changed.
        Schema/source declaration failures propagate. Ordinary value projection
        preserves the existing complex tag and rejects nonfinite values.
        """
        if form not in ("plan", "tunables"):
            raise ValueError("Unknown workflow form")
        panel = self.plan_panel if form == "plan" else self.tunables_panel
        state = panel.state
        editor = panel.editor
        env = panel.env
        revision = None
        binding = self.controller.binding
        if binding is None:
            if run_id is not None:
                self.controller.require_run_id(run_id)
        else:
            if run_id is None:
                raise FailedPreconditionError(
                    "An attached run requires explicit run_id"
                )
            self.controller.require_run_id(run_id)
            if binding.engine is None:
                raise RuntimeError("Form inspection requires native acceptance")
            if self._forms_for_run:
                env = binding.env
            revision = binding.engine.tunables(binding.run_id).revision
            if form == "tunables" and self._forms_for_run:
                state = binding.state
                if self.tuning_panel is None:
                    raise RuntimeError("Accepted run has no tuning panel")
                editor = self.tuning_panel.panel.editor
        local = editor.local_parse_errors() if batch is None else None
        return inspect_form_input(state.input, env, batch, local, revision)

    def figure_names(self, run_id: str) -> tuple[str, ...]:
        """List attached native run_id's figure names on the Qt owner, without rendering.

        Wrong/absent target raises RunMismatch; pending raises RuntimeError;
        released plots raise FailedPreconditionError(no_plots). Accepted empty
        plots return (). Terminal figures remain until Start/close, without effects.
        """
        controller = self.controller
        controller.require_run_id(run_id)
        if controller.engine is None:
            raise RuntimeError("Figures require an accepted run")
        return self._figure_views.native_names()

    def figure_png(self, run_id: str, name: str) -> bytes:
        """Render native run_id's exact name at 640 x 480 on the Qt owner.

        figure_names target/acceptance/lifetime errors apply; unknown name raises
        KeyError. Shared rendering errors propagate, without retry/cache/save,
        canvas replacement or artist-derived records. Offline views are excluded.
        """
        controller = self.controller
        controller.require_run_id(run_id)
        if controller.engine is None:
            raise RuntimeError("Figures require an accepted run")
        return self._figure_views.native_png(name)

    def open_overnight_trend(self, metadata_root: Path) -> NamedFigures:
        """Open metadata_root/journal.jsonl as a read-only overnight named Figure mapping.

        Qt owner and idle admission precede I/O; pending/active/paused runs reject.
        Source/draw errors propagate without replacing the previous offline view;
        draw is not transactional. Native Engine/RPC targets remain unchanged.
        The offline view has no run/operation/RPC binding. Its presentation is
        finished, then released at replacement/Start/close; returned Figures survive.
        """
        if get_ident() != self._thread:
            raise RuntimeError("Trend opening requires the Window owner thread")
        self._ensure_idle()
        return self._figure_views.open(metadata_root)

    def _open_trend_clicked(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select workflow metadata directory"
        )
        if not directory:
            return
        try:
            self.open_overnight_trend(Path(directory))
        except Exception as error:
            logger.exception("Overnight trend opening failed")
            self._controls.message.setText(str(error))

    def apply_tunables(
        self,
        run_id: str,
        batch: Mapping[FormWritePath, JsonValue],
        *,
        actor: Actor,
    ) -> TunablesSnapshot | FormErrors:
        """Submit external edits through this run's existing tuning panel.

        Call on the Qt owner. run_id is the explicit Start identity; Controller
        checks it before draft writes. Wrong/absent targets raise RunMismatch.
        A matching pending run without native acceptance raises RuntimeError
        without changing draft. batch uses model-field tuple leaves; sequences
        and mappings are whole leaves. {} submits complete current parsed input,
        not defaults. actor identifies the user/agent in Engine's numeric journal.
        Parsed GUI draft participates; expressions stay in FormInput and the
        fixed accepted Start capture resolves sources, without live refresh.
        Model-invalid input stays in draft and returns FormErrors. Structural/
        source and native lifecycle/journal errors propagate without retry.
        Success updates accepted input/base and clears renderer-local text.
        Failure does neither; Engine owns revision and atomic publication.
        If a native error ends an idle run, the existing Stop owner delivers its
        actual failed terminal and finishes presentation. The original error is
        reraised unchanged, even if owner cleanup fails; no batch is retried.
        """
        controller = self.controller
        controller.require_run_id(run_id)
        self._refresh_run()
        panel = self.tuning_panel
        if panel is None:
            raise RuntimeError("Applying tunables requires accepted run")
        try:
            return panel.apply_external(batch, actor=actor)
        except Exception:
            status = controller.status()
            if status is not None and status.lifecycle == "failed":
                try:
                    controller.stop(run_id)
                except Exception:
                    logger.exception(
                        "Failed to finish owner cleanup after tunables error"
                    )
            raise

    def export_fluxdep(
        self, run_id: str, *, origin: Origin = "gui"
    ) -> FluxdepExportReport:
        """Explicitly export the targeted terminal run and display all attempts.

        Call on this Window's Qt owner. This is the GUI action and RPC's common
        public entry, using the Controller's fixed Start plan/storage/provenance.
        It never reads current drafts or devices. Target/pending/unsupported and
        nonterminal errors propagate before writes. The returned report contains
        native then Labber path/status/error and a separate ledger_error; a null
        Labber path means allocation failed. Writer/ledger failures are displayed
        without changing the run outcome, canonical records or prior successes.
        """
        report = self.controller.export_fluxdep(run_id, origin=origin)
        lines = [
            f"{out.format}: {out.status}, {out.path if out.path is not None else 'unallocated'}"
            + (f", {out.error}" if out.error else "")
            for out in report.outputs
        ]
        if report.ledger_error is not None:
            lines.append(f"Ledger: {report.ledger_error}")
        self._controls.message.setText("\n".join(lines))
        return report

    def _export_clicked(self) -> None:
        run_id = self.controller.run_id
        if run_id is None:
            return
        try:
            self.export_fluxdep(run_id)
        except Exception as error:
            self._controls.message.setText(str(error))
            logger.exception("Workflow export action failed")

    def pause(self, run_id: str) -> RunStatus:
        """Request Pause of the explicit native target, retaining plan and plots.

        run_id is the identity passed to Start. Controller rejects a wrong or
        absent target with RunMismatch before side effects. A matching pending
        target raises RuntimeError; illegal native states raise InvalidRunState.
        The lease remains held until worker return and owner delivery.
        """
        return self.controller.pause(run_id)

    def resume(self, run_id: str) -> int:
        """Submit the targeted paused run using its retained operation and lease.

        run_id is the Start identity. Controller rejects wrong or absent targets.
        Pending or illegal native states reject Resume. Plan, context, roles,
        expressions and plots stay fixed. Return the same pending run token, not
        proof of worker acceptance. Submission failure retains the paused run,
        its progress and local RUN lease. No device read, retry or hardware
        rollback occurs here; each env.run captures its own native evidence.
        """
        return self.controller.resume(run_id)

    def stop(self, run_id: str) -> RunStatus | None:
        """Request Stop of Start's explicit native target without waiting.

        Controller rejects a wrong or absent target with RunMismatch before side
        effects. Return native status, or None for a matching pending startup.
        """
        return self.controller.stop(run_id)

    def _refresh_run(self) -> None:
        controller = self.controller
        status = controller.status()
        self._controls.open_trend.setEnabled(
            not controller.has_pending_segment
            and controller.operation_id is None
            and (
                status is None
                or status.lifecycle in {"done", "stopped", "failed", "aborted"}
            )
        )
        if status is None:
            self._controls.export.setEnabled(False)
            return
        self._controls.export.setEnabled(
            status.lifecycle in {"done", "stopped", "failed", "aborted"}
        )
        self._controls.message.setText(status.lifecycle)
        engine = controller.engine
        if status.workflow == "overnight" and engine is not None:
            try:
                self._figure_views.refresh_native(engine.records(status.run_id))
            except Exception as error:
                logger.exception("Committed overnight trend rendering failed")
                self._controls.message.setText(str(error))
        self._controls.pause.setEnabled(
            controller.operation_id is not None and status.lifecycle == "running"
        )
        self._controls.resume.setEnabled(
            not controller.has_pending_segment and status.lifecycle == "paused"
        )
        binding = controller.binding
        if (
            self._forms_for_run
            and self.tuning_panel is None
            and binding is not None
            and binding.engine is not None
        ):
            self.tuning_panel = T1TuningPanel(
                controller, self, panel_factory=self._forms.attachment.knobs_panel
            )
            self._controls.layout.addWidget(self.tuning_panel)
        if self.tuning_panel is not None:
            self.tuning_panel.setEnabled(
                status.lifecycle in {"running", "pausing", "paused"}
            )

    def _finished(self, status: RunStatus | None, error: Exception | None) -> None:
        """Display native delivery; terminal admission is already released.

        Paused delivery keeps the same operation, lease, plots and run inputs.
        View errors belong to logging, not the native operation outcome.
        """
        self._refresh_run()
        self._controls.pause.setEnabled(False)
        self._controls.resume.setEnabled(
            error is None and status is not None and status.lifecycle == "paused"
        )
        if error is None and status is not None and status.lifecycle == "paused":
            # Paused delivery retains the operation and run resources.
            self._poll.start()
            return
        self._poll.stop()
        self.plan_panel.setEnabled(True)
        self.tunables_panel.setEnabled(True)
        self._controls.selector.setEnabled(True)
        self._controls.start.setEnabled(True)
        if error is not None:
            self._controls.message.setText(str(error))
        elif status is None:
            self._controls.message.setText("Stopped before startup")
        self._figure_views.finish_native()

    def _pause_clicked(self) -> None:
        run_id = self.controller.run_id
        if run_id is None:
            return
        try:
            self.pause(run_id)
        except Exception as error:
            logger.exception("T1 workflow Pause failed")
            self._controls.message.setText(str(error))

    def _resume_clicked(self) -> None:
        run_id = self.controller.run_id
        if run_id is None:
            return
        try:
            self.resume(run_id)
        except Exception as error:
            logger.exception("T1 workflow Resume failed")
            self._controls.message.setText(str(error))

    def _stop_clicked(self) -> None:
        # Qt clicked(bool) is not a run target; use the Controller's native ID.
        run_id = self.controller.run_id
        if run_id is None:
            return
        try:
            self.stop(run_id)
        except Exception as error:
            logger.exception("T1 workflow Stop failed")
            self._controls.message.setText(str(error))

    def _start_clicked(self) -> None:
        try:
            result = self.start(new_run_id())
        except Exception as error:
            logger.exception("T1 workflow Start failed")
            self._controls.message.setText(str(error))
            return
        if isinstance(result, FormErrors):
            self._controls.message.setText("; ".join(result.messages.values()))

    def _shutdown_on_close(self, event: QCloseEvent | None) -> bool:
        """Isolate composition shutdown failure and reject the Qt close attempt."""
        try:
            if self._on_close is not None:
                self._on_close()
        except Exception as error:
            logger.exception("T1 workflow session shutdown failed")
            self._controls.message.setText(str(error))
            if event is not None:
                event.ignore()
            return False
        return True

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Keep producer/canvas alive during Stop; release only after terminal."""
        run_id = self.controller.run_id
        if self.controller.has_pending_segment:
            if run_id is not None:
                try:
                    self.stop(run_id)
                except Exception as error:
                    logger.exception("T1 workflow Stop on close failed")
                    self._controls.message.setText(str(error))
                    if a0 is not None:
                        a0.ignore()
                    return
            if a0 is not None:
                a0.ignore()
            return
        status = self.controller.status()
        if status is not None and status.lifecycle == "paused" and run_id is not None:
            try:
                self.stop(run_id)
            except Exception as error:
                logger.exception("T1 workflow Stop on close failed")
                self._controls.message.setText(str(error))
                if a0 is not None:
                    a0.ignore()
                return
        if not self._shutdown_on_close(a0):
            return
        self.setup_dialog.close()
        self._poll.stop()
        self._figure_views.release()
        self.plan_panel.editor.detach()
        self.tunables_panel.editor.detach()
        if self.tuning_panel is not None:
            self.tuning_panel.close()
        super().closeEvent(a0)
