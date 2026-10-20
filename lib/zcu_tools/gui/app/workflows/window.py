"""Generic workflow form, captured startup and Qt plotting composition."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from threading import get_ident
from typing import TYPE_CHECKING, Literal

from pydantic import JsonValue
from qick import QickConfig
from qtpy.QtCore import QSignalBlocker
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QWidget

from zcu_tools.datafile import RunSnapshot, SoftwareProvenance
from zcu_tools.device import DeviceInfo
from zcu_tools.experiment.acquisition import capture_software_provenance
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows import (
    Actor,
    EnginePorts,
    InvalidRunState,
    RunIdentity,
    RunPaths,
    RunStatus,
    TunablesSnapshot,
)
from zcu_tools.experiment.workflows.clock import UtcClock
from zcu_tools.gui.app.workflows._form_io import FormInspection, prepare_start_form
from zcu_tools.gui.app.workflows.controller import WorkflowController
from zcu_tools.gui.app.workflows.figure_area import WorkflowFigureArea
from zcu_tools.gui.app.workflows.layout import build_workflow_layout
from zcu_tools.gui.app.workflows.persistence import (
    WorkflowInputRestoreReport,
    WorkflowInputSnapshot,
)
from zcu_tools.gui.app.workflows.remote import (
    FormDescriptionInspectionReply,
    FormInspectionReply,
    WorkflowRemoteControlAdapter,
)
from zcu_tools.gui.app.workflows.runtime import SessionDevices
from zcu_tools.gui.app.workflows.session_strip import WorkflowSessionStrip
from zcu_tools.gui.app.workflows.tuning_panel import WorkflowTuningPanel
from zcu_tools.gui.catalog_loader import CatalogReloadError, SourceCatalogLoader
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormModel,
    FormWritePath,
)
from zcu_tools.gui.remote.control_service import ControlOptions
from zcu_tools.gui.session.alias_presets import AliasPresets
from zcu_tools.gui.session.persistence import PersistenceError
from zcu_tools.gui.session.ports import OwnerScheduler
from zcu_tools.gui.session.services.build import SessionServices
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.ui.arb_waveform_dialog import ArbWaveformDialog
from zcu_tools.gui.session.ui.device_dialog import DeviceDialog
from zcu_tools.gui.session.ui.inspect_dialog import InspectDialog
from zcu_tools.gui.session.ui.setup_dialog import SetupDialog
from zcu_tools.gui.session.ui.soc_dialog import SocDialog
from zcu_tools.gui.widgets.dialog_lifecycle import DialogRefStore
from zcu_tools.gui.widgets.dialog_presenter import QtDialogPresenter
from zcu_tools.plotting.plots import Plots
from zcu_tools.progress_bar.backend.tqdm import TQDMProgressBar
from zcu_tools.resources.entry import new_run_id
from zcu_tools.resources.entry.save_layout import workflow_roots

from .attachment import Attachment
from .catalog import WorkflowCatalog
from .figures import WorkflowFigureViews
from .forms import WorkflowDraft, WorkflowForms
from .input_restore_message import WorkflowInputRestoreMessage
from .run_view import WorkflowRunView

if TYPE_CHECKING:
    from zcu_tools.cfg_model import TemplateDef
    from zcu_tools.gui.app.workflows.panel import WorkflowFormPanel, WorkflowPlanPanel


logger = logging.getLogger(__name__)


class WorkflowWindow[C](QWidget):
    """Compose one workflow selector, captured Start and tuning on the Qt owner.

    plan_panel/tunables_panel expose selected independent parsed drafts.
    controller stays fixed; its engine is None until native acceptance, and
    tuning_panel presents its unique run binding in the Tunables tab; a status
    bar names the run above the diagnostics line (WorkflowRunView). Shared
    SessionServices owns runner/progress, hardware, exclusion and workers;
    parent permits embedding.
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
        registrations: Sequence[Attachment[C]],
        templates: Mapping[str, TemplateDef],
        alias_presets: AliasPresets | None,
        loader: SourceCatalogLoader[WorkflowCatalog[C]] | None = None,
        on_close: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Build a Window on the GUI owner with an existing QApplication.

        state belongs to services; owner marshals plot commands. parent=None is
        top-level; otherwise embedded. EMPTY builds partial source-free drafts.
        registrations is nonempty, ordered and has unique declaration names;
        its first item is the default. Invalid catalogs raise ValueError.
        The event bus comes from services. templates supplies the
        persistent module/waveform catalog shared Inspect offers under New; an
        empty catalog is valid. alias_presets is the caller's complete named
        catalog for Inspect, or None for no presets. No files are read for either.
        loader supplies current WorkflowCatalog[C] instances on source reload;
        caller owns its bounded namespaces and fresh-registration builder.
        loader=None is unmonitored and disables Reload.
        Reload keeps session/Controller and the retained run binding. Prepare
        captures context and its alias table; source errors propagate without I/O.
        Caller retains this Window/parent through shutdown. Close requests Stop,
        waits for delivery, then calls on_close on owner; None skips caller cleanup.
        Callback errors reject close, display diagnostics and retain retry resources.
        """
        super().__init__(parent)
        current = WorkflowCatalog(tuple(registrations))
        self._loader = loader
        self._thread = get_ident()
        self._catalog_error: CatalogReloadError | None = None
        self._state = state
        self._services = services
        self._templates = templates
        self._alias_presets = alias_presets
        self._dialog_presenter = QtDialogPresenter()
        self._runner = services.runner
        self._owner = owner
        self._progress = services.progress
        self._on_close = on_close
        self.controller: WorkflowController[FormModel, C] = WorkflowController(
            services.runner, "workflow", on_finished=self._finished
        )

        env = (
            self._capture_env(self._services.device.capture_native_device_info())
            if state.session_env.parameters is not None
            else FormEnv()
        )
        self._forms = WorkflowForms(current, env, self)
        self._session = WorkflowSessionStrip(services, self)
        self._figure_area = WorkflowFigureArea(owner, self)
        self._controls = build_workflow_layout(
            self,
            self._session,
            self._figure_area,
            self.plan_panel,
            self.tunables_panel,
        )
        self._input_restore_message = WorkflowInputRestoreMessage(
            self._controls.message
        )
        self._dialogs = DialogRefStore()
        self._controls.selector.addItems(list(current.names))
        self._controls.selector.currentTextChanged.connect(self._selection_changed)
        for button, slot in (
            (self._controls.start, self._start_clicked),
            (self._controls.pause, self._pause_clicked),
            (self._controls.resume, self._resume_clicked),
            (self._controls.stop, self._stop_clicked),
            (self._controls.setup, self.open_setup),
            (self._controls.soc, self.open_soc),
            (self._controls.devices, self.open_devices),
            (self._controls.inspect, self.open_inspect),
            (self._controls.prepare, self._use_source_clicked),
            (self._controls.reload, self._reload_clicked),
        ):
            button.clicked.connect(slot)
        self._run_view = WorkflowRunView(
            self._controls, self.controller, self._forms, self
        )
        self._figure_views = WorkflowFigureViews(self._figure_area)
        self.setup_dialog = SetupDialog(self._services.setup_control, self)
        self.setup_dialog.finished.connect(self._session.refresh)
        self.soc_dialog = SocDialog(self._services.soc_control, self)
        self._sync_run_locked_controls()

    setup_dialog: SetupDialog
    soc_dialog: SocDialog

    def open_setup(self) -> None:
        """Show the same retained shared Setup non-modally on the GUI owner.

        Reopening refreshes the dialog's cached inputs, not form captures. No
        entry is opened or hardware connected by showing. Closing Setup refreshes
        the session chips without replacing form drafts.
        """
        self.setup_dialog.show()
        self.setup_dialog.raise_()
        self.setup_dialog.activateWindow()

    def open_soc(self) -> None:
        """Show the retained non-modal SoC connection/configuration dialog.

        Reopening prefills cached connection inputs. The host keeps submissions
        read-only while a run holds the session; configuration browsing and
        Close remain available.
        """
        self.soc_dialog.show()
        self.soc_dialog.raise_()
        self.soc_dialog.activateWindow()

    def open_devices(self) -> None:
        """Open the shared Device dialog window-modally on the GUI owner.

        It lists and sets up this session's devices through device_control and
        evaluates expressions against the current parameter context. A dialog
        already open is raised instead of opening a second one; closing deletes
        it. From Start until terminal delivery the hardware gate rejects device
        changes, so the dialog is read-only: it still lists, refreshes and
        polls the selected device, but cannot add, drop, apply or stop. Opening
        reads no hardware and changes no form capture.
        """
        dialog = self._dialogs.get("devices")
        if dialog is not None:
            dialog.raise_()
            dialog.activateWindow()
            return
        dialog = DeviceDialog(
            self._services.device_control,
            parameter_provider=self._services.context_control.read_context,
            parent=self,
        )
        dialog.set_read_only(read_only=self._run_view.start_locked)
        self._dialogs.open_named("devices", dialog)

    def open_inspect(self) -> None:
        """Open the shared inspector on this session, on the Qt owner thread.

        Raise an already open instance. Close deletes it; the next opening
        reads fresh aliases as well as the current parameters and library.
        From Start until terminal delivery only browsing is allowed. Writes
        outside that interval go through the same context control as Setup.
        """
        dialog = self._dialogs.get("inspect")
        if dialog is not None:
            dialog.raise_()
            dialog.activateWindow()
            return
        dialog = InspectDialog(
            self._services.context_control,
            self._services.setup_control.get_bus(),
            templates=self._templates,
            arb_waveforms=self._services.arb_waveform,
            open_arb_waveforms=self._open_arb_waveforms,
            dialog_presenter=self._dialog_presenter,
            alias_presets=self._alias_presets,
            parent=self,
        )
        dialog.set_read_only(read_only=self._run_view.start_locked)
        self._dialogs.open_named("inspect", dialog)

    def _open_arb_waveforms(self) -> None:
        dialog = self._dialogs.get("arb")
        if dialog is not None:
            dialog.raise_()
            dialog.activateWindow()
            return
        dialog = ArbWaveformDialog(self._services.arb_waveform, parent=self)
        dialog.set_read_only(read_only=self._run_view.start_locked)
        self._dialogs.open_named("arb", dialog)

    def _sync_run_locked_controls(self) -> None:
        """Apply the run's Start lock to source controls and shared dialogs.

        From Start until terminal delivery (pending, running, pausing, paused,
        stopping) the host rejects Prepare forms and Reload source, so both are
        disabled. The hardware gate rejects device changes; the host also locks
        source edits through shared dialogs. Setup, SoC, Device, Inspect and ARB
        dialogs are read-only, with their open buttons still enabled. Reload also
        needs a loader and no restart-required reload failure.
        """
        locked = self._run_view.start_locked
        error = self._catalog_error
        restart_required = error is not None and error.restart_required
        self._controls.prepare.setEnabled(not locked)
        self._controls.reload.setEnabled(
            not locked and self._loader is not None and not restart_required
        )
        self.setup_dialog.set_read_only(read_only=locked)
        self.soc_dialog.set_read_only(read_only=locked)
        for key in ("devices", "inspect", "arb"):
            dialog = self._dialogs.get(key)
            if isinstance(dialog, (DeviceDialog, InspectDialog, ArbWaveformDialog)):
                dialog.set_read_only(read_only=locked)

    def _use_source_clicked(self) -> None:
        try:
            self.prepare_forms()
        except Exception as error:
            logger.exception("Workflow source capture failed")
            self._controls.message.setText(str(error))

    def prepare_forms(self) -> None:
        """Replace both Start drafts and bases with current captured defaults.

        Call on the GUI owner with a readable parameter context.
        Capture parameters/library and cached device values once, then replace
        this Window's form attachments. Previous edits and local parse text are
        discarded on success; subsequent source changes do not update the forms.
        No live device reads, submission, storage write or input reload occurs.
        EMPTY raises FailedPreconditionError. Forms resolve their own alias
        dependencies; missing/unbound aliases raise RoleResolutionError naming
        the alias when consumed by form preparation. No component kind is checked.
        Source/editor failures propagate without replacing the existing panels.
        Pending, running or paused operations raise InvalidRunState before capture.
        Success also refreshes the session chips from current services.
        """
        self._ensure_can_start()
        env = self._capture_env(self._services.device.capture_native_device_info())
        if env.parameters is None:
            raise FailedPreconditionError("No parameter context is installed")
        old = self.plan_panel, self.tunables_panel
        self._forms.prepare(env, self)
        self._publish_panels(*old)
        self._session.refresh()

    @property
    def plan_panel(self) -> WorkflowPlanPanel:
        """Return the selected attachment's current parsed Start plan view."""
        return self._forms.plan

    @property
    def tunables_panel(self) -> WorkflowFormPanel[FormModel]:
        """Return the selected attachment's current parsed Start knobs view."""
        return self._forms.knobs

    @property
    def tuning_panel(self) -> WorkflowTuningPanel[FormModel, C] | None:
        """Return the attached run's tuning panel, None before native acceptance.

        It is the Tunables tab's content while the run is live and stays,
        hidden, after terminal delivery until Start, panel replacement or close.
        """
        return self._run_view.tuning_panel

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
        changes. Existing run binding stays with Controller.
        Selection reuses fixed form captures, without device/source reads. Local
        unparsed text is not saved. No run/operation/lease is created or replaced.
        """
        self._ensure_can_start()
        if name == self.workflow_name:
            return
        old = self.plan_panel, self.tunables_panel
        report = self._forms.select(name, self._capture_draft(), self)
        self._publish_panels(*old)
        self._sync_selector()
        self.show_input_restore_message(report)

    def _sync_selector(self) -> None:
        with QSignalBlocker(self._controls.selector):
            self._controls.selector.clear()
            self._controls.selector.addItems(list(self._forms.catalog.names))
            self._controls.selector.setCurrentText(self.workflow_name)
        self._run_view.refresh()

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
        self._controls.plan_page.replaceWidget(old_plan, self.plan_panel)
        self._controls.tunables_page.replaceWidget(old_tunables, self.tunables_panel)
        for panel in (old_plan, old_tunables):
            panel.close()
            panel.deleteLater()
        self.plan_panel.show()
        self.tunables_panel.show()
        self._run_view.forms_replaced()

    def reload_source(self) -> WorkflowInputRestoreReport:
        """Replace current declarations and compatible parsed drafts, on owner thread.

        Pending/running/paused ownership rejects before source preflight/import.
        loader=None rejects unmonitored embedding. Success replaces only form
        panels/catalog, retains a still-declared selection or chooses the new
        default, and reports changed names/dropped paths in the message label.
        services/session/controller/hardware and the old run binding
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
        snapshot = self._capture_draft()
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
            self._sync_run_locked_controls()
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
        from the Start panel. From Start until terminal delivery, knobs come
        from the run's unique Controller binding, including unsubmitted parsed
        edits; otherwise, including after terminal delivery, from the editable
        Start knobs panel. Renderer-local incomplete text, native values and
        journal/hardware state are excluded.
        """
        draft = self._capture_draft()
        return WorkflowInputSnapshot.capture(draft.plan, draft.knobs)

    def _capture_draft(self) -> WorkflowDraft:
        binding = self.controller.binding
        # start_locked, not knobs_from_run: save the knobs the user edits and
        # would Start next. After terminal delivery those are the Start knobs,
        # already reset to the run's accepted input, while remote reads of the
        # run_id still use the run's binding.
        knobs = (
            binding.state.input
            if binding is not None and self._run_view.start_locked
            else self.tunables_panel.state.input
        )
        return WorkflowDraft(self.plan_panel.state.input, knobs)

    def restore_persisted_state(
        self, state: WorkflowInputSnapshot
    ) -> WorkflowInputRestoreReport:
        """Restore complete saved drafts, resetting both panel bases atomically.

        Call on the owner thread after construction, before starting a run.
        Pending/running/paused ownership raises InvalidRunState before edits.
        Core restores both drafts recursively against current defaults without
        reading sources or executing converters. Format/provider failures leave
        both panels untouched. Missing refs and ordinary invalid values remain
        draft input for general validation. Return dropped paths for display.
        Success refreshes both editors and clears renderer-local text.
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
        No dialog, retry, form mutation or I/O occurs. Empty successful outcomes
        clear only a previous restore message, never unrelated run diagnostics.
        Call only on the GUI owner thread.
        """
        self._input_restore_message.show(report, load_error)

    def _capture_env(self, devices: Mapping[str, DeviceInfo]) -> FormEnv:
        context = self._services.context
        parameters = context.read_context()
        return FormEnv(
            parameters=parameters,
            library=parameters.library,
            devices=devices,
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
        The selected registration supplies its opaque workflow context seed.
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
            raise ValueError(
                "Workflow Start requires connected SoC and native QickConfig"
            )
        roots = workflow_roots(
            result_path=storage.result_path,
            database_path=storage.database_path,
            workflow=attachment.name,
            run_id=run_id,
        )
        plots = Plots(self._figure_area)
        ports, provenance = self._start_ports(
            session.soc, session.soccfg, plots, prepared.context
        )
        identity = RunIdentity(run_id, provenance, str(storage.entry_id), storage.point)
        # Install presentation before synchronous admission listeners can reenter.
        self._run_view.release_tuning()
        self._figure_views.install_native(plots)
        try:
            token = prepared.submit(
                self.controller,
                ports,
                RunPaths(metadata_root=roots.result, data_root=roots.database),
                identity,
                env,
            )
        except Exception:
            self._figure_views.start_failed()
            raise
        self.plan_panel.state.reset(submitted_plan)
        self.tunables_panel.state.reset(accepted_input)
        for editor in (self.plan_panel.editor, self.tunables_panel.editor):
            editor.refresh(env)
            editor.discard_local_text()
        self._run_view.submitted(attachment.name)
        self._sync_run_locked_controls()
        return token

    def _start_ports(
        self,
        soc: object,
        soccfg: QickConfig,
        plots: Plots,
        seed: C | None,
    ) -> tuple[EnginePorts[C], SoftwareProvenance]:
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
    ) -> WorkflowRemoteControlAdapter[C]:
        """Create an inert RPC adapter of these controls on this Window's owner.

        opts configures shared transport; bus must be this Window's session bus.
        The composition caller owns adapter.start()/stop() and stops the socket
        before destroying this Window. This call does not open a socket, capture
        new inputs or touch hardware. Remote Stop requires explicit run_id;
        stopping the transport does not cancel a native run.
        """
        return WorkflowRemoteControlAdapter(
            self, opts, bus=bus, owner_scheduler=self._owner
        )

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
        Source/declaration failures propagate. Ordinary value projection
        preserves the existing complex tag and rejects nonfinite values.
        """
        return self._form_inspection(form, run_id).read(batch)

    def describe_form(
        self,
        form: Literal["plan", "tunables"],
        path: FormWritePath,
        *,
        run_id: str | None = None,
        batch: Mapping[FormWritePath, JsonValue] | None = None,
    ) -> FormDescriptionInspectionReply | FormErrors:
        """Observe and describe one form candidate in the same GUI owner turn.

        form, run_id and batch follow inspect_form's target/preview contract.
        path is a tuple of model-field names; () describes root. Return the same
        lean observation plus metadata for path and its direct children, using
        that exact candidate and fixed env. Model-invalid candidates remain
        readable. Structural writes return FormErrors; unknown/inactive/whole-
        leaf descendant paths raise FormPathError. Pending/stale target and source/
        declaration failures follow inspect_form. No input, base, accepted input,
        Engine, capture, local text or source is changed and no I/O occurs.
        """
        return self._form_inspection(form, run_id).describe(path, batch)

    def _form_inspection(
        self,
        form: Literal["plan", "tunables"],
        run_id: str | None,
    ) -> FormInspection:
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
            # knobs_from_run, not start_locked: run_id names the attached run
            # until fresh Start panels replace its forms, so reads and previews
            # use that run's fixed env and knobs state, also after terminal.
            if self._run_view.knobs_from_run:
                env = binding.env
            revision = binding.engine.tunables(binding.run_id).revision
            if form == "tunables" and self._run_view.knobs_from_run:
                state = binding.state
                if self.tuning_panel is None:
                    raise RuntimeError("Accepted run has no tuning panel")
                editor = self.tuning_panel.panel.editor
        return FormInspection(state.input, env, editor.local_parse_errors(), revision)

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
        self._run_view.refresh()
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

    def _finished(self, status: RunStatus | None, error: Exception | None) -> None:
        """Display native delivery; terminal admission is already released.

        Paused delivery keeps the same operation, lease, plots and run inputs.
        View errors belong to logging, not the native operation outcome.
        """
        if error is None and status is not None and status.lifecycle == "paused":
            self._run_view.paused()
            return
        self._services.device.refresh_connected_device_info()
        self._run_view.ended(status, error)
        self._sync_run_locked_controls()
        self._figure_views.finish_native()

    def _pause_clicked(self) -> None:
        run_id = self.controller.run_id
        if run_id is None:
            return
        try:
            self.pause(run_id)
        except Exception as error:
            logger.exception("Workflow Pause failed")
            self._controls.message.setText(str(error))
            return
        self._run_view.refresh()

    def _resume_clicked(self) -> None:
        run_id = self.controller.run_id
        if run_id is None:
            return
        try:
            self.resume(run_id)
        except Exception as error:
            logger.exception("Workflow Resume failed")
            self._controls.message.setText(str(error))
            return
        self._run_view.refresh()

    def _stop_clicked(self) -> None:
        # Qt clicked(bool) is not a run target; use the Controller's native ID.
        run_id = self.controller.run_id
        if run_id is None:
            return
        try:
            self.stop(run_id)
        except Exception as error:
            logger.exception("Workflow Stop failed")
            self._controls.message.setText(str(error))
            return
        self._run_view.refresh()

    def _start_clicked(self) -> None:
        try:
            result = self.start(new_run_id())
        except Exception as error:
            logger.exception("Workflow Start failed")
            self._controls.message.setText(str(error))
            return
        if isinstance(result, FormErrors):
            self._controls.message.setText("; ".join(result.messages.values()))

    def _shutdown_on_close(self, event: QCloseEvent | None) -> bool:
        """Resolve Inspect drafts before shutdown; reject cancellation or failure."""
        inspect = self._dialogs.get("inspect")
        if inspect is not None and not inspect.close():
            # Cancel must leave the session services alive for continued editing.
            if event is not None:
                event.ignore()
            return False
        try:
            if self._on_close is not None:
                self._on_close()
        except Exception as error:
            logger.exception("Workflow session shutdown failed")
            self._controls.message.setText(str(error))
            if event is not None:
                event.ignore()
            return False
        return True

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Handle Qt close on the owner thread, retaining an active producer/canvas.

        a0 is Qt's close event, or None for binding-compatible direct dispatch.
        Pending work requests Stop and ignores the event until owner delivery;
        paused runs stop before shutdown. Control/cleanup failures are logged and
        shown and reject close, without releasing retry resources. After terminal
        delivery and successful on_close, close Setup/forms and release plots.
        Borrowed services remain caller-owned; this never disconnects hardware
        independently of on_close. No event-loop wait or retry occurs here.
        """
        run_id = self.controller.run_id
        if self.controller.has_pending_segment:
            if run_id is not None:
                try:
                    self.stop(run_id)
                except Exception as error:
                    logger.exception("Workflow Stop on close failed")
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
                logger.exception("Workflow Stop on close failed")
                self._controls.message.setText(str(error))
                if a0 is not None:
                    a0.ignore()
                return
        if not self._shutdown_on_close(a0):
            return
        self.setup_dialog.close()
        self.soc_dialog.close()
        for key in ("devices", "arb"):
            dialog = self._dialogs.get(key)
            if dialog is not None:
                dialog.reject()
        self._session.detach()
        self._run_view.close()
        self._figure_views.release()
        self.plan_panel.editor.detach()
        self.tunables_panel.editor.detach()
        super().closeEvent(a0)
