"""Concrete T1 GUI composition behind the shared process runtime."""

from typing import ClassVar

from zcu_tools.device.manager import DeviceManager
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.remote.control_service import ControlOptions
from zcu_tools.gui.runtime import GuiAssembly, GuiRuntimeBehavior, GuiRuntimeSpec
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.hardware_gate import RunBlocksHardwareGate
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.services.build import SessionServices, build_session_services
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from zcu_lab import ALIAS_PRESET

from .window import T1WorkflowWindow


class WorkflowsGuiBehavior(GuiRuntimeBehavior):
    """Compose a fresh EMPTY session with one Window and native Controller.

    state and services are the shared session owners created by assemble.
    window is its sole T1WorkflowWindow, also used by the optional RPC adapter.
    Keep this behavior alive through Qt shutdown. No preferences/input files
    are restored and construction does not connect instruments.
    """

    spec: ClassVar[GuiRuntimeSpec] = GuiRuntimeSpec(
        app_name="workflows",
        app_slug="workflows",
        default_control_port=0,
        logging_extra_namespaces=("zcu_lab",),
    )
    state: SessionState
    services: SessionServices
    window: T1WorkflowWindow

    def __init__(self, *, project_root: str = "") -> None:
        """Anchor entry result/Database roots at project_root; no I/O yet.

        An empty root means the launch cwd. Component definitions must already
        be explicitly registered by the launcher before Setup opens an entry.
        Requires an existing QApplication before assemble, not before this call.
        """
        self._project_root = project_root

    def assemble(self, control: ControlOptions | None) -> GuiAssembly:
        """Build one session/Qt infrastructure and return its native owners.

        Run on the QApplication's owner thread once per behavior. control=None
        omits RPC; otherwise create an inert adapter via window.create_remote_service.
        Assembly.controller is window.controller, not a wrapper controller.
        Window borrows runner/progress from services and has no Qt parent.
        Bind entry waveform assets using ArbWaveformDatabase.info_at. No socket,
        entry or hardware is opened. Source/Qt construction failures propagate.
        Window close invokes shutdown before accepting close.
        """
        self.state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
        self._bus = BaseEventBus()
        self._gate = RunBlocksHardwareGate(run_kind="run", bus=self._bus)
        self._handles = OperationHandles()
        self._background = BackgroundRunner()
        self._owner = QtOwnerScheduler()
        self._progress_transport = QtProgressTransport()
        self._progress = ProgressService(self._progress_transport)
        self._runner = OperationRunner(
            self._gate, self._handles, self._progress, self._background, self._bus
        )
        self._devices = DeviceManager()
        self.services = build_session_services(
            state=self.state,
            bus=self._bus,
            gate=self._gate,
            handles=self._handles,
            background=self._background,
            progress=self._progress,
            runner=self._runner,
            bind_assets=ArbWaveformDatabase.info_at,
            project_root=self._project_root,
            device_registry=self._devices,
            alias_rows=ALIAS_PRESET,
        )
        self.window = T1WorkflowWindow(
            self.state,
            self.services,
            self._owner,
            on_close=self.shutdown,
        )
        self._adapter = (
            self.window.create_remote_service(control, self._bus)
            if control is not None
            else None
        )
        return GuiAssembly(
            controller=self.window.controller,
            window=self.window,
            control_adapter=self._adapter,
        )

    def after_show(self, assembly: GuiAssembly) -> None:
        """Open this assembly's same shared Setup instance non-modally.

        Called by shared runtime after showing the Window and starting control.
        Reject another Window with ValueError. No hardware/entry is auto-selected.
        """
        if assembly.window is not self.window:
            raise ValueError("Setup requires this behavior's Window")
        self.window.open_setup()

    def shutdown(self) -> None:
        """Close executor admission, cancel handles, drain and release resources.

        Call on the owner thread after assemble and native terminal delivery.
        Permanently close submission admission, cancel every pending operation,
        then drain the same executor's workers and queued terminal delivery before
        closing this registry's drivers and stopping RPC. Drain timeout/driver
        close failures raise; Window rejects close and displays the error.
        Admission stays closed after failure; retry drain/close, not new work.
        Repeated successful calls are harmless. This
        neither changes hardware output values nor reconnects anything.
        """
        self._background.stop_accepting()
        self._handles.cancel_all()
        if not self._background.quiesce():
            raise RuntimeError("Session background drain did not complete")
        self._devices.close_all_devices()
        if self._adapter is not None:
            self._adapter.stop()
