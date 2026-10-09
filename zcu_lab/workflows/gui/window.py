"""Concrete T1 form, captured startup and Qt plotting composition."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from math import isfinite
from typing import Literal
from uuid import uuid4

from pydantic import JsonValue
from pydantic_core import PydanticCustomError
from qick import QickConfig
from qtpy.QtCore import QTimer
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget
from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.acquisition import capture_software_provenance
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows import (
    Actor,
    DeviceSnapshot,
    EnginePorts,
    InvalidRunState,
    RunIdentity,
    RunPaths,
    RunStatus,
    TunablesSnapshot,
)
from zcu_tools.experiment.workflows.clock import UtcClock
from zcu_tools.gui.app.workflows.controller import WorkflowController
from zcu_tools.gui.app.workflows.runtime import SessionDevices
from zcu_tools.gui.app.workflows.tunables import TunablesHost
from zcu_tools.gui.cfg.codec import encode_complex
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormState,
    FormValue,
    FormWritePath,
)
from zcu_tools.gui.plotting.container import FigureContainer
from zcu_tools.gui.plotting.explicit import QtPlotHost
from zcu_tools.gui.plotting.figure_export import render_figure_png
from zcu_tools.gui.remote.control_service import ControlOptions
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.ports import OwnerScheduler
from zcu_tools.gui.session.services.build import SessionServices
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.ui.setup_dialog import SetupDialog
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.plotting.plots import Plots
from zcu_tools.progress_bar.backend.tqdm import TQDMProgressBar
from zcu_tools.resources.entry import RoleSpec
from zcu_tools.resources.entry.save_layout import workflow_roots

from ..calibration import Calibrated
from ..t1 import T1Plan, t1_fluxdep
from .panel import T1PlanPanel, T1TunablesPanel
from .plan import T1PlanForm
from .remote import FormInspectionReply, T1RemoteControlAdapter
from .tunables import T1TunablesForm
from .tuning_panel import T1TuningPanel

logger = logging.getLogger(__name__)


def _prepare_start_form[F: FormModel](
    model: type[F],
    state: FormState[F],
    editor: FormEditor[F],
    batch: Mapping[FormWritePath, JsonValue] | None,
    env: FormEnv,
) -> tuple[F, FormInput[F]] | FormErrors:
    """Select draft or captured defaults, returning validated values and input.

    Explicit batches never read the editor or draft. Native directed default/write
    errors retain their field path; undirected structural errors use ().
    """
    if batch is None:
        errors = editor.local_parse_errors()
        if errors is not None:
            return errors
    else:
        try:
            state = FormState(model.default_input(env))
            state.write(batch, env)
        except PydanticCustomError as exc:
            # field_error owns the directed path; other native errors propagate.
            if exc.type != "form_field" or exc.context is None:
                raise
            return FormErrors({exc.context["path"]: exc.context["message"]})
        except ValueError as exc:
            # Batch structure errors have no directed path in the core contract.
            return FormErrors({(): str(exc)})
    form = state.result(env)
    if isinstance(form, FormErrors):
        return form
    return form, state.input


def _encode_form_value(value: FormValue) -> JsonValue:
    """Project ordinary display leaves; scalar complex format belongs to codec."""
    if isinstance(value, complex):
        encoded: dict[str, JsonValue] = {
            key: list[JsonValue](parts) for key, parts in encode_complex(value).items()
        }
        return encoded
    if isinstance(value, (list, tuple)):
        return [_encode_form_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode_form_value(item) for key, item in value.items()}
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("Nonfinite form values cannot be encoded as JSON")
    return value


def _inspect_form_input(
    input: FormInput[FormModel],
    env: FormEnv,
    batch: Mapping[FormWritePath, JsonValue] | None,
    local_errors: FormErrors | None,
    revision: int | None,
) -> FormInspectionReply | FormErrors:
    """Project input/candidate and partial ordinary values without publication.

    env/revision are fixed owner observations, not sources to refresh. A batch
    uses the core's write contract in this call-local state. local_errors are
    supplied only for current GUI reads. Translate directed/structural write
    errors as Start does; declaration and unexpected failures propagate.
    """
    state = FormState(input)
    if batch is not None:
        try:
            state.write(batch, env)
        except PydanticCustomError as exc:
            if exc.type != "form_field" or exc.context is None:
                raise
            return FormErrors({exc.context["path"]: exc.context["message"]})
        except ValueError as exc:
            return FormErrors({(): str(exc)})
    preview = state.preview(env)
    errors = dict(preview.errors.messages) if preview.errors is not None else {}
    if local_errors is not None:
        errors.update(local_errors.messages)
    return {
        "schema": state.input_schema(env),
        "input": state.input.data,
        "values": [
            {"path": list(path), "value": _encode_form_value(value)}
            for path, value in preview.values.items()
        ],
        "errors": [
            {"path": list(path), "message": message} for path, message in errors.items()
        ],
        "revision": revision,
    }


def _calibrated_seed(plan: T1Plan, pi_length_us: float) -> Calibrated:
    frequency = plan.qub_pulse.freq
    readout = plan.readout.pulse_cfg
    if (
        not isinstance(frequency, float)
        or not isinstance(readout.freq, float)
        or not isinstance(readout.gain, float)
    ):
        raise ValueError("T1 initial calibration requires scalar module frequency/gain")
    return Calibrated(frequency, pi_length_us, readout.freq, readout.gain)


class T1WorkflowWindow(QWidget):
    """Compose captured T1 run controls and accepted-run tuning on the Qt owner.

    plan_panel and tunables_panel expose the existing editable form state.
    controller is fixed for this Window's lifetime; its native engine is None
    until startup acceptance. tuning_panel presents its unique run binding.
    The supplied session/runner own shared hardware, exclusion and workers.
    This window does not create a second registry, launch a GUI process or
    refresh YAML. Its RPC adapter calls the same controls. Start and each Resume
    synchronously read all connected devices for native provenance; any read
    failure names the device and rejects that request without cached fallback.
    Pause retains plan, plots and input capture for the same run.
    """

    def __init__(
        self,
        state: SessionState,
        services: SessionServices,
        runner: OperationRunner,
        owner: OwnerScheduler,
        progress: ProgressService,
        parent: QWidget | None = None,
        *,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        """Build form/plot attachments on the GUI owner with a QApplication.

        state/services/runner/progress must be the same owning session. The
        progress service is borrowed directly from its composition root, not
        obtained through a wider control facet. owner marshals
        plot commands to this thread. EMPTY or unselected focus builds partial
        source-free drafts. A selected focus captures defaults; source errors
        propagate without fallback. Start requires a saved point, native
        QickConfig and connected SoC. setup_dialog is the retained shared Setup
        instance over this session's setup_control. Selecting a qubit changes
        Context focus only; Use source calls prepare_forms explicitly.
        parent controls Qt lifetime; closing during work requests Stop and keeps
        the window alive until terminal delivery. Once native work ends, on_close
        releases composition-owned resources. If it raises, close is rejected
        and the error is displayed, so resources remain alive for another attempt.
        No instrument is contacted by construction.
        """
        super().__init__(parent)
        self._state = state
        self._services = services
        self._runner = runner
        self._owner = owner
        self._progress = progress
        self.controller: WorkflowController[T1TunablesForm, Calibrated] = (
            WorkflowController(runner, "t1-workflow", on_finished=self._finished)
        )
        self.tuning_panel: T1TuningPanel | None = None
        self._plots: Plots | None = None

        env = self._capture_env(
            DeviceSnapshot(self._services.device.capture_native_device_info())
        )
        self.plan_panel = T1PlanPanel(env, self)
        self.tunables_panel = T1TunablesPanel(
            FormState(T1TunablesForm.default_input(env)), env, self
        )
        self._message = QLabel(self)
        self._message.setObjectName("workflow.run.message")
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

        stack = QStackedWidget(self)
        placeholder = QLabel("Workflow plots", stack)
        stack.addWidget(placeholder)
        self._plot_host = QtPlotHost(FigureContainer(stack, placeholder), owner)
        self._layout = QVBoxLayout(self)
        for widget in (
            self.plan_panel,
            self.tunables_panel,
            self._start_button,
            self._pause_button,
            self._resume_button,
            stop_button,
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
        qubit choices from ContextControlPort without replacing form drafts.
        """
        raise NotImplementedError

    def prepare_forms(self) -> None:
        """Replace both Start drafts and bases with current captured defaults.

        Call on the GUI owner after selecting a qubit through Context focus.
        Capture parameters/library and cached device values once, then replace
        this Window's form attachments. Previous edits and local parse text are
        discarded on success; subsequent source changes do not update the forms.
        No live device reads, submission, storage write or input reload occurs.
        Source/editor failures propagate without replacing the existing panels.
        Pending, running or paused operations raise InvalidRunState before capture.
        """
        raise NotImplementedError

    def _capture_env(self, devices: DeviceSnapshot) -> FormEnv:
        context = self._services.context
        roles = context.capture_roles(
            {"qubit": RoleSpec("qubit/*")},
            focus=context.get_component_focus().name,
            explicit={},
        )
        parameters = context.read_context(roles=roles)
        return FormEnv(
            parameters=parameters, library=parameters.library, devices=devices.items
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
        failures raise without fallback/retry. Every connected device is read
        synchronously before submission, and read failure names that device.
        The returned operation token is
        pending, not proof of accepted startup. Calibrated uses the validated
        form's required pi_length_us and selected native module scalars.
        Expressions remain in the accepted FormInput used for later tuning.
        """
        self._ensure_can_start()
        env = self._capture_env(
            DeviceSnapshot(self._services.device.capture_live_native_device_info())
        )
        if env.parameters is None:
            raise ValueError("T1 Start requires captured parameters")
        storage = self._services.context.capture_storage(env.parameters)
        prepared_plan = _prepare_start_form(
            T1PlanForm, self.plan_panel.state, self.plan_panel.editor, plan_batch, env
        )
        if isinstance(prepared_plan, FormErrors):
            return prepared_plan
        form, submitted_plan = prepared_plan
        prepared_tunables = _prepare_start_form(
            T1TunablesForm,
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
        seed = _calibrated_seed(plan, form.pi_length_us)
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
        if self.tuning_panel is not None:
            self.tuning_panel.close()
            self.tuning_panel.deleteLater()
            self.tuning_panel = None
        self._plots = plots
        try:
            token = self.controller.start(
                t1_fluxdep,
                plan=plan,
                tunables=tunables_form.to_cfg(),
                paths=RunPaths(metadata_root=roots.result, data_root=roots.database),
                ports=ports,
                binding=binding,
                identity=RunIdentity(
                    run_id,
                    DeviceSnapshot(env.devices),
                    provenance,
                    str(storage.entry_id),
                    storage.point,
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
        self._message.setText("Startup pending")
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
            env = binding.env
            revision = binding.engine.tunables(binding.run_id).revision
            if form == "tunables":
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

    def pause(self, run_id: str) -> RunStatus:
        """Request Pause of the explicit native target, retaining plan and plots.

        run_id is the identity passed to Start. Controller rejects a wrong or
        absent target with RunMismatch before side effects. A matching pending
        target raises RuntimeError; illegal native states raise InvalidRunState.
        The lease remains held until worker return and owner delivery.
        """
        return self.controller.pause(run_id)

    def resume(self, run_id: str) -> int:
        """Submit the targeted paused run after reading all connected devices.

        run_id is the Start identity. Controller rejects wrong or absent targets
        before device reads. Pending or illegal native states reject Resume.
        Read every connected driver synchronously through DeviceService. Any
        read failure names the device and rejects this Resume without falling
        back to cache. Record read-back values as native device provenance.
        Plan, context, roles, expressions and plots stay fixed. Return the same
        pending run token, not proof of worker acceptance. Healthy read or
        submission failure retains the paused run, its progress and local lease.
        No retry, ramp-interleaving guard or hardware rollback is provided.
        """
        return self.controller.resume(
            run_id,
            devices=lambda: DeviceSnapshot(
                self._services.device.capture_live_native_device_info()
            ),
        )

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
            return
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
            result = self.start(uuid4().hex)
        except Exception as error:
            logger.exception("T1 workflow Start failed")
            self._message.setText(str(error))
            return
        if isinstance(result, FormErrors):
            self._message.setText("; ".join(result.messages.values()))

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
        self._poll.stop()
        if self._plots is not None:
            self._plots.release()
            self._plots = None
        self.plan_panel.editor.detach()
        self.tunables_panel.editor.detach()
        if self.tuning_panel is not None:
            self.tuning_panel.close()
        super().closeEvent(a0)
