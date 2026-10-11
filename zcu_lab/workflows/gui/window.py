"""Concrete T1 form, captured startup and Qt plotting composition."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from functools import partial
from threading import get_ident
from typing import TYPE_CHECKING, Literal

from pydantic import JsonValue
from qick import QickConfig
from qtpy.QtCore import QTimer
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import (
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from zcu_tools.datafile import RunSnapshot, SoftwareProvenance
from zcu_tools.device import DeviceInfo
from zcu_tools.experiment.acquisition import capture_software_provenance
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows import (
    Actor,
    EnginePorts,
    FluxdepExportReport,
    FluxdepExportSnapshot,
    InvalidRunState,
    RunIdentity,
    RunPaths,
    RunStatus,
    TunablesSnapshot,
)
from zcu_tools.experiment.workflows.clock import UtcClock
from zcu_tools.gui.app.workflows.controller import WorkflowController
from zcu_tools.gui.app.workflows.exporting import FluxdepExportBinding
from zcu_tools.gui.app.workflows.runtime import SessionDevices
from zcu_tools.gui.app.workflows.tunables import TunablesHost
from zcu_tools.gui.catalog_loader import CatalogReloadError, SourceCatalogLoader
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormState,
    FormWritePath,
)
from zcu_tools.gui.plotting.container import FigureContainer
from zcu_tools.gui.plotting.explicit import QtPlotHost
from zcu_tools.gui.plotting.figure_export import render_figure_png
from zcu_tools.gui.remote.control_service import ControlOptions
from zcu_tools.gui.session.context_types import ContextSourceUnavailableError
from zcu_tools.gui.session.persistence import PersistenceError
from zcu_tools.gui.session.ports import OwnerScheduler
from zcu_tools.gui.session.services.build import SessionServices
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.ui.alias_dialog import AliasDialog
from zcu_tools.gui.session.ui.setup_dialog import SetupDialog
from zcu_tools.plotting.plots import Plots
from zcu_tools.progress_bar.backend.tqdm import TQDMProgressBar
from zcu_tools.resources.entry import (
    Origin,
    PointStorageBinding,
    RoleResolutionError,
    WorkflowRoots,
    new_run_id,
)
from zcu_tools.resources.entry.save_layout import workflow_roots

from ..calibration import Calibrated
from ._form_io import _inspect_form_input, _prepare_start_form
from .catalog import WorkflowCatalog, build_catalog
from .persistence import WorkflowInputRestoreReport, WorkflowInputSnapshot
from .remote import FormInspectionReply, T1RemoteControlAdapter
from .tuning_panel import T1TuningPanel

if TYPE_CHECKING:
    from ..t1 import T1Plan
    from .panel import T1PlanPanel, T1TunablesPanel
    from .tunables import T1TunablesForm

logger = logging.getLogger(__name__)


def _build_export_binding(
    plan: T1Plan,
    storage: PointStorageBinding,
    run_id: str,
    provenance: SoftwareProvenance,
    roots: WorkflowRoots,
    catalog: WorkflowCatalog,
) -> FluxdepExportBinding | None:
    """Detach the native plan and bind Start's exact entry storage once."""
    if catalog.prepare_export is None:
        return None
    identity = RunIdentity(run_id, provenance, str(storage.entry_id), storage.point)
    paths = RunPaths(metadata_root=roots.result, data_root=roots.database)
    fixed_plan = type(plan).model_validate_json(
        plan.model_dump_json(),
        by_name=True,
        by_alias=False,
    )
    return FluxdepExportBinding(
        FluxdepExportSnapshot(
            identity, paths, fixed_plan.snapshot, datetime.now(UTC), None
        ),
        partial(catalog.prepare_export, plan=fixed_plan, data_root=paths.data_root),
        storage.result_path,
        storage.database_path,
        storage.ledger,
    )


class T1WorkflowWindow(QWidget):
    """Compose captured T1 run controls and accepted-run tuning on the Qt owner.

    plan_panel and tunables_panel expose the existing editable form state.
    controller is fixed for this Window's lifetime; its native engine is None
    until startup acceptance. tuning_panel presents its unique run binding.
    The supplied SessionServices owns shared runner/progress infrastructure,
    hardware, exclusion and workers. parent permits Qt embedding.
    This window does not create a second registry, launch a GUI process or
    refresh YAML. Its RPC adapter calls the same controls. Start uses cached
    native device info for forms; each env.run captures connected devices in
    the worker before acquisition. Resume retains the same operation.
    Pause retains plan, plots and input capture for the same run.
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

        state is the session state owned by services. services supplies the shared
        runner and progress infrastructure for this session. parent embeds the
        Window in a Qt widget; None creates a top-level Window. owner marshals
        plot commands to this thread. EMPTY builds partial source-free drafts.
        prepare_forms explicitly replaces drafts using the readable context
        and its fixed global qubit alias capture. Source errors propagate without
        fallback. Start requires a saved point, native QickConfig and connected
        SoC. setup_dialog is the retained shared Setup instance over this
        session's setup_control. The Aliases button opens the shared AliasDialog
        over this session's context_control; closing it updates source status,
        not captured forms. Only explicit prepare_forms replaces those drafts.
        The caller retains this Window and any Qt parent through shutdown.
        Closing during work requests Stop and keeps it alive until terminal
        delivery. Once native
        work ends, optional on_close releases composition-owned resources on the
        GUI owner before accepting close. If it raises, close is rejected
        and the error is displayed, so resources remain alive for another attempt.
        catalog pairs current science/forms/panels; None builds the default T1
        catalog. loader enables explicit source reload; None leaves embedding
        unmonitored. Reload preserves this session/controller and terminal export
        binding. No instrument is contacted by construction.
        """
        super().__init__(parent)
        self._catalog = catalog if catalog is not None else build_catalog()
        self._loader = loader
        self._thread = get_ident()
        self._forms_for_run = False
        self._state = state
        self._services = services
        self._runner = services.runner
        self._owner = owner
        self._progress = services.progress
        self._on_close = on_close
        self.controller: WorkflowController[T1TunablesForm, Calibrated] = (
            WorkflowController(
                services.runner, "t1-workflow", on_finished=self._finished
            )
        )
        self.tuning_panel: T1TuningPanel | None = None
        self._plots: Plots | None = None

        env = (
            self._capture_env(self._services.device.capture_native_device_info())
            if state.session_env.parameters is not None
            else FormEnv()
        )
        self.plan_panel = self._catalog.plan_panel(env, self)
        self.tunables_panel = self._catalog.knobs_panel(
            FormState(self._catalog.knobs_model.default_input(env)), env, self
        )
        self._message = QLabel(self)
        self._message.setObjectName("workflow.run.message")
        self._message.setWordWrap(True)
        self._start_button = QPushButton("Start", self)
        self._start_button.setObjectName("workflow.run.start")
        self._start_button.clicked.connect(self._start_clicked)
        self._pause_button = QPushButton("Pause", self)
        self._pause_button.setObjectName("workflow.run.pause")
        self._pause_button.setEnabled(False)
        self._pause_button.clicked.connect(self._pause_clicked)
        self._resume_button = QPushButton("Resume", self)
        self._resume_button.setObjectName("workflow.run.resume")
        self._resume_button.setEnabled(False)
        self._resume_button.clicked.connect(self._resume_clicked)
        stop_button = QPushButton("Stop", self)
        stop_button.setObjectName("workflow.run.stop")
        stop_button.clicked.connect(self._stop_clicked)
        self._export_button = QPushButton("Export merged/Labber", self)
        self._export_button.setObjectName("workflow.run.export_fluxdep")
        self._export_button.setEnabled(False)
        self._export_button.clicked.connect(self._export_clicked)

        stack = QStackedWidget(self)
        placeholder = QLabel("Workflow plots", stack)
        stack.addWidget(placeholder)
        self._plot_host = QtPlotHost(FigureContainer(stack, placeholder), owner)
        self._layout = QVBoxLayout(self)
        self._init_source_controls()
        for widget in (
            self.plan_panel,
            self.tunables_panel,
            self._start_button,
            self._pause_button,
            self._resume_button,
            stop_button,
            self._export_button,
            self._message,
            stack,
        ):
            self._layout.addWidget(widget)
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

    def _init_source_controls(self) -> None:
        self.setup_dialog = SetupDialog(self._services.setup_control, self)
        self.setup_dialog.finished.connect(self._sync_source_status)
        setup_button = QPushButton("Setup", self)
        setup_button.setObjectName("workflow.source.setup")
        setup_button.clicked.connect(self.open_setup)
        aliases_button = QPushButton("Aliases", self)
        aliases_button.setObjectName("workflow.source.aliases")
        aliases_button.clicked.connect(self._aliases_clicked)
        self._source_status = QLabel(self)
        self._source_status.setObjectName("workflow.source.status")
        use_button = QPushButton("Prepare forms", self)
        use_button.setObjectName("workflow.source.use")
        use_button.clicked.connect(self._use_source_clicked)
        for widget in (setup_button, aliases_button, self._source_status, use_button):
            self._layout.addWidget(widget)
        self._sync_source_status()

    def _aliases_clicked(self) -> None:
        try:
            dialog = AliasDialog(self._services.context_control, self)
        except Exception as error:
            logger.exception("T1 workflow alias editor could not open")
            self._message.setText(str(error))
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
            self._source_status.setText("Qubit: no context")
        except (RoleResolutionError, ContextSourceUnavailableError) as error:
            self._source_status.setText(f"Qubit unavailable: {error}")
        else:
            self._source_status.setText(f"Qubit: {source}")

    def _use_source_clicked(self) -> None:
        try:
            self.prepare_forms()
        except Exception as error:
            logger.exception("T1 workflow source capture failed")
            self._message.setText(str(error))

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
        plan_panel = self._catalog.plan_panel(env, self)
        try:
            tunables_panel = self._catalog.knobs_panel(
                FormState(self._catalog.knobs_model.default_input(env)), env, self
            )
        except Exception:
            # The first editor is not installed if the second construction fails.
            plan_panel.close()
            plan_panel.deleteLater()
            raise
        self._publish_panels(plan_panel, tunables_panel)

    def _publish_panels(
        self, plan_panel: T1PlanPanel, tunables_panel: T1TunablesPanel
    ) -> None:
        old_plan, old_tunables = self.plan_panel, self.tunables_panel
        self._layout.replaceWidget(old_plan, plan_panel)
        self._layout.replaceWidget(old_tunables, tunables_panel)
        self.plan_panel, self.tunables_panel = plan_panel, tunables_panel
        for panel in (old_plan, old_tunables):
            panel.close()
            panel.deleteLater()
        plan_panel.show()
        tunables_panel.show()
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
        panels/catalog, resets bases and reports dropped paths in the existing
        message label; services/session/controller/hardware and old run export
        binding remain unchanged. Local unparsed text is not a saved draft.
        CatalogReloadError propagates on import/integrity failure; imports are
        bounded, not a rollback of arbitrary Python side effects.
        """
        if get_ident() != self._thread:
            raise CatalogReloadError(
                "Reload must run on its owner thread", restart_required=True
            )
        self._ensure_can_start()
        if self._loader is None:
            raise CatalogReloadError("Source reload is not configured")
        snapshot = self.capture_persisted_state()
        candidate = self._loader.load(self._loader.prepare())
        env = self.plan_panel.env
        plan = candidate.plan_panel(env, self)
        try:
            knobs = candidate.knobs_panel(
                FormState(candidate.knobs_model.default_input(env)), env, self
            )
            report = WorkflowInputRestoreReport(
                plan_dropped=plan.state.write_compatible(snapshot.plan, env),
                knobs_dropped=knobs.state.write_compatible(snapshot.knobs, env),
            )
            plan.state.reset(plan.state.input)
            knobs.state.reset(knobs.state.input)
            plan.editor.refresh(env)
            knobs.editor.refresh(env)
        except Exception:
            plan.close()
            plan.deleteLater()
            raise
        self._publish_panels(plan, knobs)
        self._catalog = candidate
        self.show_input_restore_message(report)
        return report

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
        plan = FormState(self._catalog.plan_model.default_input(self.plan_panel.env))
        knobs = FormState(
            self._catalog.knobs_model.default_input(self.tunables_panel.env)
        )
        report = WorkflowInputRestoreReport(
            plan_dropped=plan.write_compatible(state.plan, self.plan_panel.env),
            knobs_dropped=knobs.write_compatible(state.knobs, self.tunables_panel.env),
        )
        self.plan_panel.state.reset(plan.input)
        self.tunables_panel.state.reset(knobs.input)
        for panel in (self.plan_panel, self.tunables_panel):
            panel.editor.refresh(panel.env)
            panel.editor.discard_local_text()
        return report

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
        self._message.setText("; ".join(messages))

    def _capture_env(self, devices: Mapping[str, DeviceInfo]) -> FormEnv:
        context = self._services.context
        parameters = context.read_context()
        return FormEnv(
            parameters=parameters, library=parameters.library, devices=devices
        )

    def _ensure_can_start(self) -> None:
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
    ) -> int | FormErrors:
        """Capture/validate and submit T1, returning a token or directed errors.

        run_id is a nonempty path-safe segment for this run's exact directories.
        A None batch reads that GUI panel and its local parse errors. An explicit
        batch, including {}, starts from that form's default_input in this Start's
        fixed capture, then uses FormState.write once. Paths are model-field tuple
        leaves; lists/mappings are whole leaves and branch switches use core rules.
        Invalid batch/model input returns FormErrors without touching either panel,
        submission or files. Structural errors without a directed path are model-wide.
        Success resets both panel bases to the submitted input and clears local text.
        Both editors display ordinary values in this same fixed Start capture,
        not their old constructor capture; they do not poll later source changes.
        Draft/unbound storage, missing hardware, source/provenance and native
        failures raise without fallback/retry. Forms use detached native device
        info from the session cache; Start does not read connected drivers.
        The returned operation token is
        pending, not proof of accepted startup. Calibrated uses the validated
        form's required pi_length_us and selected native module scalars.
        Expressions remain in the accepted FormInput used for later tuning.
        """
        self._ensure_can_start()
        env = self._capture_env(self._services.device.capture_native_device_info())
        if env.parameters is None:
            raise ValueError("T1 Start requires captured parameters")
        storage = self._services.context.capture_storage(env.parameters)
        prepared_plan = _prepare_start_form(
            self._catalog.plan_model,
            self.plan_panel.state,
            self.plan_panel.editor,
            plan_batch,
            env,
        )
        if isinstance(prepared_plan, FormErrors):
            return prepared_plan
        form, submitted_plan = prepared_plan
        prepared_tunables = _prepare_start_form(
            self._catalog.knobs_model,
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
        plan = form.to_cfg(env, snapshot)
        if isinstance(plan, FormErrors):
            return plan
        seed = self._catalog.seed(plan, form.pi_length_us)
        session = self._state.session_env
        if session.soc is None or not isinstance(session.soccfg, QickConfig):
            raise ValueError("T1 Start requires connected SoC and native QickConfig")
        roots = workflow_roots(
            result_path=storage.result_path,
            database_path=storage.database_path,
            run_id=run_id,
        )
        plots = Plots(self._plot_host)
        devices = self._services.device.get_connected_devices()
        context = RunContext(session.soc, session.soccfg, plots, devices, StopSignal())
        provenance = capture_software_provenance(context=context)
        ports = EnginePorts(
            plots=plots,
            progress=TQDMProgressBar,
            clock=UtcClock(),
            context=seed,
            soc=session.soc,
            soccfg=session.soccfg,
            devices=SessionDevices(self._services.device),
        )
        binding: TunablesHost[T1TunablesForm, Calibrated] = TunablesHost(
            FormState(accepted_input), env, run_id
        )
        if self._plots is not None:
            self._plots.release()
        # Install presentation before synchronous admission listeners can reenter.
        self._close_tuning_panel()
        self._plots = plots
        try:
            token = self.controller.start(
                self._catalog.step,
                plan=plan,
                tunables=tunables_form.to_cfg(),
                paths=RunPaths(metadata_root=roots.result, data_root=roots.database),
                ports=ports,
                binding=binding,
                identity=RunIdentity(
                    run_id, provenance, str(storage.entry_id), storage.point
                ),
                export_binding=_build_export_binding(
                    plan, storage, run_id, provenance, roots, self._catalog
                ),
            )
        except Exception:
            plots.finish(present=False)
            plots.release()
            self._plots = None
            raise
        self.plan_panel.state.reset(submitted_plan)
        self.tunables_panel.state.reset(accepted_input)
        for editor in (self.plan_panel.editor, self.tunables_panel.editor):
            editor.refresh(env)
            editor.discard_local_text()
        self.plan_panel.setEnabled(False)
        self.tunables_panel.setEnabled(False)
        self._start_button.setEnabled(False)
        self._export_button.setEnabled(False)
        self._message.setText("Startup pending")
        self._forms_for_run = True
        self._poll.start()
        return token

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
        return _inspect_form_input(state.input, env, batch, local, revision)

    def figure_names(self, run_id: str) -> tuple[str, ...]:
        """List names from this run's Plots without rendering; call on Qt owner.

        run_id must be the attached native identity. RunMismatch rejects absent
        or stale targets first. Pending acceptance raises RuntimeError. Released
        plots raise FailedPreconditionError(no_plots), never an empty success.
        An accepted run without registered figures returns (). Terminal figures
        stay readable until a new Start or close. No control/input/save effects.
        """
        controller = self.controller
        controller.require_run_id(run_id)
        if controller.engine is None:
            raise RuntimeError("Figures require an accepted run")
        plots = self._plots
        if plots is None:
            raise FailedPreconditionError(
                "Run plots have been released", reason_code="no_plots"
            )
        return tuple(plots)

    def figure_png(self, run_id: str, name: str) -> bytes:
        """Render the named current Figure as PNG at 640 x 480 on the Qt owner.

        run_id follows figure_names target/acceptance/lifetime requirements.
        name must be an exact registered name; unknown names raise KeyError.
        Use the existing shared renderer; no saving, cache, canvas replacement or
        artist-derived records. Renderer failures propagate without retry.
        """
        controller = self.controller
        controller.require_run_id(run_id)
        if controller.engine is None:
            raise RuntimeError("Figures require an accepted run")
        plots = self._plots
        if plots is None:
            raise FailedPreconditionError(
                "Run plots have been released", reason_code="no_plots"
            )
        return render_figure_png(plots[name])

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
        self._message.setText("\n".join(lines))
        return report

    def _export_clicked(self) -> None:
        run_id = self.controller.run_id
        if run_id is None:
            return
        try:
            self.export_fluxdep(run_id)
        except Exception as error:
            self._message.setText(str(error))
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
        if status is None:
            self._export_button.setEnabled(False)
            return
        self._export_button.setEnabled(
            status.lifecycle in {"done", "stopped", "failed", "aborted"}
        )
        self._message.setText(status.lifecycle)
        self._pause_button.setEnabled(
            controller.operation_id is not None and status.lifecycle == "running"
        )
        self._resume_button.setEnabled(
            not controller.has_pending_segment and status.lifecycle == "paused"
        )
        binding = controller.binding
        if (
            self.tuning_panel is None
            and binding is not None
            and binding.engine is not None
        ):
            self.tuning_panel = T1TuningPanel(controller, self)
            self._layout.addWidget(self.tuning_panel)
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
        self._pause_button.setEnabled(False)
        self._resume_button.setEnabled(
            error is None and status is not None and status.lifecycle == "paused"
        )
        if error is None and status is not None and status.lifecycle == "paused":
            # Paused delivery retains the operation and run resources.
            self._poll.start()
            return
        self._poll.stop()
        self.plan_panel.setEnabled(True)
        self.tunables_panel.setEnabled(True)
        self._start_button.setEnabled(True)
        if error is not None:
            self._message.setText(str(error))
        elif status is None:
            self._message.setText("Stopped before startup")
        if self._plots is not None:
            self._plots.finish()

    def _pause_clicked(self) -> None:
        run_id = self.controller.run_id
        if run_id is None:
            return
        try:
            self.pause(run_id)
        except Exception as error:
            logger.exception("T1 workflow Pause failed")
            self._message.setText(str(error))

    def _resume_clicked(self) -> None:
        run_id = self.controller.run_id
        if run_id is None:
            return
        try:
            self.resume(run_id)
        except Exception as error:
            logger.exception("T1 workflow Resume failed")
            self._message.setText(str(error))

    def _stop_clicked(self) -> None:
        # Qt clicked(bool) is not a run target; use the Controller's native ID.
        run_id = self.controller.run_id
        if run_id is None:
            return
        try:
            self.stop(run_id)
        except Exception as error:
            logger.exception("T1 workflow Stop failed")
            self._message.setText(str(error))

    def _start_clicked(self) -> None:
        try:
            result = self.start(new_run_id())
        except Exception as error:
            logger.exception("T1 workflow Start failed")
            self._message.setText(str(error))
            return
        if isinstance(result, FormErrors):
            self._message.setText("; ".join(result.messages.values()))

    def _shutdown_on_close(self, event: QCloseEvent | None) -> bool:
        """Isolate composition shutdown failure and reject the Qt close attempt."""
        try:
            if self._on_close is not None:
                self._on_close()
        except Exception as error:
            logger.exception("T1 workflow session shutdown failed")
            self._message.setText(str(error))
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
                    self._message.setText(str(error))
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
                self._message.setText(str(error))
                if a0 is not None:
                    a0.ignore()
                return
        if not self._shutdown_on_close(a0):
            return
        self.setup_dialog.close()
        self._poll.stop()
        if self._plots is not None:
            self._plots.release()
            self._plots = None
        self.plan_panel.editor.detach()
        self.tunables_panel.editor.detach()
        if self.tuning_panel is not None:
            self.tuning_panel.close()
        super().closeEvent(a0)
