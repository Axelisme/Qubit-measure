"""Concrete T1 GUI composition behind the shared process runtime."""

from typing import ClassVar

from zcu_tools.gui.remote.control_service import ControlOptions
from zcu_tools.gui.runtime import GuiAssembly, GuiRuntimeBehavior, GuiRuntimeSpec
from zcu_tools.gui.session.services.build import SessionServices
from zcu_tools.gui.session.state import SessionState

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
        Bind entry waveform assets using ArbWaveformDatabase.info_at. No socket,
        entry or hardware is opened. Source/Qt construction failures propagate.
        Window close invokes shutdown before accepting close.
        """
        raise NotImplementedError

    def after_show(self, assembly: GuiAssembly) -> None:
        """Open this assembly's same shared Setup instance non-modally.

        Called by shared runtime after showing the Window and starting control.
        Reject another Window with ValueError. No hardware/entry is auto-selected.
        """
        raise NotImplementedError

    def shutdown(self) -> None:
        """Cancel session handles, drain workers and release owned resources.

        Call on the owner thread after assemble and native terminal delivery.
        Cancel every pending session operation, then quiesce the same background
        executor before closing the registry's drivers and stopping RPC.
        Drain timeout/driver close failures raise; Window rejects close without
        destroying the session. Repeated successful calls are harmless. This
        neither changes hardware output values nor reconnects anything.
        """
        raise NotImplementedError
