"""Concrete T1 form, captured startup and Qt plotting composition."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from uuid import uuid4

from pydantic import JsonValue
from qick import QickConfig
from qtpy.QtCore import QTimer
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget
from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.acquisition import capture_software_provenance
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows import (
    DeviceSnapshot,
    EnginePorts,
    InvalidRunState,
    RunIdentity,
    RunMismatch,
    RunPaths,
    RunStatus,
)
from zcu_tools.experiment.workflows.clock import UtcClock
from zcu_tools.gui.app.workflows.controller import WorkflowController
from zcu_tools.gui.app.workflows.runtime import SessionDevices
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormWritePath
from zcu_tools.gui.plotting.container import FigureContainer
from zcu_tools.gui.plotting.explicit import QtPlotHost
from zcu_tools.gui.remote.control_service import ControlOptions
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.ports import OwnerScheduler
from zcu_tools.gui.session.services.build import SessionServices
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.plotting.plots import Plots
from zcu_tools.progress_bar.backend.tqdm import TQDMProgressBar
from zcu_tools.resources.entry import RoleSpec
from zcu_tools.resources.entry.save_layout import workflow_roots

from ..calibration import Calibrated
from ..t1 import T1Plan, t1_fluxdep
from .panel import T1PlanPanel, T1TunablesPanel
from .remote import T1RemoteControlAdapter
from .tunables import T1TunablesForm
from .tuning_panel import T1TuningPanel

logger = logging.getLogger(__name__)


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
    controller is None before the first submitted Start; its native engine is
    None until startup acceptance. tuning_panel attaches only after acceptance.
    The supplied session/runner own shared hardware, exclusion and workers.
    This window does not create a second registry, launch a GUI process or
    refresh YAML. Its RPC adapter calls the same controls. Pause retains plan, plots and input capture; Resume only
    captures current cached devices after reacquiring the lease. Each Start captures current
    published parameters and cached device data in one owner sequence.
    """

    def __init__(
        self,
        state: SessionState,
        services: SessionServices,
        runner: OperationRunner,
        owner: OwnerScheduler,
        progress: ProgressService,
        parent: QWidget | None = None,
    ) -> None:
        """Build form/plot attachments on the GUI owner with a QApplication.

        state/services/runner/progress must be the same owning session. The
        progress service is borrowed directly from its composition root, not
        obtained through a wider control facet. owner marshals
        plot commands to this thread. A readable context and resolvable qubit
        focus are required to build form defaults. Startup also requires a saved
        point, native QickConfig and connected SoC. Source errors propagate.
        parent controls Qt lifetime; closing during work requests Stop and keeps
        the window alive until terminal delivery. No instrument is contacted.
        """
        super().__init__(parent)
        self._state = state
        self._services = services
        self._runner = runner
        self._owner = owner
        self._progress = progress
        self.controller: WorkflowController[Calibrated] | None = None
        self.tuning_panel: T1TuningPanel | None = None
        self._plots: Plots | None = None
        self._run_env: FormEnv | None = None
        self._accepted_input: FormInput[T1TunablesForm] | None = None

        env = self._capture_env()
        self.plan_panel = T1PlanPanel(env, self)
        self.tunables_panel = T1TunablesPanel(env, self)
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

    def _capture_env(self) -> FormEnv:
        context = self._services.context
        roles = context.capture_roles(
            {"qubit": RoleSpec("qubit/*")},
            focus=context.get_component_focus().name,
            explicit={},
        )
        parameters = context.read_context(roles=roles)
        devices = self._services.device.capture_native_device_info()
        return FormEnv(
            parameters=parameters, library=parameters.library, devices=devices
        )

    def _ensure_can_start(self) -> None:
        """Reject replacement until the native run and owner delivery both end."""
        if self.controller is None:
            return
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
        Draft/unbound storage, missing hardware, source/provenance and native
        failures raise without fallback/retry. The returned operation token is
        pending, not proof of accepted startup. Calibrated uses the validated
        form's required pi_length_us and selected native module scalars.
        Expressions remain in the accepted FormInput used for later tuning.
        """
        self._ensure_can_start()
        if plan_batch is not None or tunables_batch is not None:
            raise NotImplementedError("partial-input Start preparation")
        env = self._capture_env()
        parameters = env.parameters
        if parameters is None:
            raise ValueError("T1 Start requires captured parameters")
        storage = self._services.context.capture_storage(parameters)
        for editor in (self.plan_panel.editor, self.tunables_panel.editor):
            errors = editor.local_parse_errors()
            if errors is not None:
                return errors
        form = self.plan_panel.state.result(env)
        if isinstance(form, FormErrors):
            return form
        tunables_form = self.tunables_panel.state.result(env)
        if isinstance(tunables_form, FormErrors):
            return tunables_form
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
        candidate = WorkflowController(
            self._runner,
            ports,
            "t1-workflow",
            self._progress,
            on_finished=self._finished,
        )
        if self._plots is not None:
            self._plots.release()
        accepted_input = self.tunables_panel.state.input
        # Runner progress listeners can synchronously request Stop or close.
        self.controller = candidate
        try:
            token = candidate.start(
                t1_fluxdep,
                plan=plan,
                tunables=tunables_form.to_cfg(),
                paths=RunPaths(metadata_root=roots.result, data_root=roots.database),
                identity=RunIdentity(
                    run_id,
                    DeviceSnapshot(env.devices),
                    provenance,
                    str(storage.entry_id),
                    storage.point,
                ),
            )
        except Exception:
            self.controller = None
            raise
        if self.tuning_panel is not None:
            self.tuning_panel.close()
            self.tuning_panel.deleteLater()
            self.tuning_panel = None
        self._plots = plots
        self._run_env = env
        self._accepted_input = accepted_input
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

    def pause(self, run_id: str) -> RunStatus:
        """Request Pause of the explicit native target, retaining plan and plots.

        run_id is the identity passed to Start. Controller rejects a wrong or
        absent target with RunMismatch before side effects. A matching pending
        target raises RuntimeError; illegal native states raise InvalidRunState.
        The lease remains held until worker return and owner delivery.
        """
        if self.controller is None:
            raise RunMismatch(run_id, None)
        return self.controller.pause(run_id)

    def resume(self, run_id: str) -> int:
        """Reacquire the RUN lease and submit the explicitly targeted paused run.

        run_id is the identity passed to Start. Controller rejects a wrong or
        absent target with RunMismatch before lease admission or cached device
        capture. A matching pending target raises RuntimeError; illegal native
        states raise InvalidRunState. Plan, context, roles, expressions and plots
        stay fixed. Return the pending segment token, not native acceptance.
        Capture/submission errors propagate without retry or live hardware reads.
        """
        if self.controller is None:
            raise RunMismatch(run_id, None)
        return self.controller.resume(
            run_id,
            devices=lambda: DeviceSnapshot(
                self._services.device.capture_native_device_info()
            ),
        )

    def stop(self, run_id: str) -> RunStatus | None:
        """Request Stop of Start's explicit native target without waiting.

        Controller rejects a wrong or absent target with RunMismatch before side
        effects. Return native status, or None for a matching pending startup.
        """
        if self.controller is None:
            raise RunMismatch(run_id, None)
        return self.controller.stop(run_id)

    def _refresh_run(self) -> None:
        controller = self.controller
        if controller is None:
            return
        status = controller.status()
        if status is None:
            return
        self._message.setText(status.lifecycle)
        self._pause_button.setEnabled(
            controller.operation_id is not None and status.lifecycle == "running"
        )
        self._resume_button.setEnabled(
            controller.operation_id is None and status.lifecycle == "paused"
        )
        engine = controller.engine
        if (
            self.tuning_panel is None
            and engine is not None
            and self._run_env is not None
            and self._accepted_input is not None
        ):
            self.tuning_panel = T1TuningPanel(
                engine, status.run_id, self._run_env, self._accepted_input, self
            )
            self._layout.addWidget(self.tuning_panel)
        if self.tuning_panel is not None:
            self.tuning_panel.setEnabled(
                status.lifecycle in {"running", "pausing", "paused"}
            )

    def _finished(self, status: RunStatus | None, error: Exception | None) -> None:
        self._refresh_run()
        self._pause_button.setEnabled(False)
        self._resume_button.setEnabled(
            error is None and status is not None and status.lifecycle == "paused"
        )
        if error is None and status is not None and status.lifecycle == "paused":
            # Keep polling across segments; admission still guards owner delivery.
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
        run_id = None if self.controller is None else self.controller.run_id
        if run_id is None:
            return
        try:
            self.pause(run_id)
        except Exception as error:
            logger.exception("T1 workflow Pause failed")
            self._message.setText(str(error))

    def _resume_clicked(self) -> None:
        run_id = None if self.controller is None else self.controller.run_id
        if run_id is None:
            return
        try:
            self.resume(run_id)
        except Exception as error:
            logger.exception("T1 workflow Resume failed")
            self._message.setText(str(error))

    def _stop_clicked(self) -> None:
        # Qt clicked(bool) is not a run target; use the Controller's native ID.
        run_id = None if self.controller is None else self.controller.run_id
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
        run_id = None if self.controller is None else self.controller.run_id
        if self.controller is not None and (
            self.controller.has_pending_segment
            or self.controller.operation_id is not None
        ):
            if run_id is not None:
                self.stop(run_id)
            if a0 is not None:
                a0.ignore()
            return
        if self.controller is not None:
            status = self.controller.status()
            if (
                status is not None
                and status.lifecycle == "paused"
                and run_id is not None
            ):
                self.stop(run_id)
        self._poll.stop()
        if self._plots is not None:
            self._plots.release()
        self.plan_panel.editor.detach()
        self.tunables_panel.editor.detach()
        if self.tuning_panel is not None:
            self.tuning_panel.close()
        super().closeEvent(a0)
