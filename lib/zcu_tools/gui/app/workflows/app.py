"""Generic workflow GUI composition behind the shared process runtime."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar

from zcu_tools.cfg_model import TemplateDef
from zcu_tools.device.manager import DeviceManager
from zcu_tools.gui.app.workflows.persistence import (
    WorkflowInputCodec,
    WorkflowInputRestoreReport,
    WorkflowInputSnapshot,
)
from zcu_tools.gui.catalog_loader import SourceCatalogLoader
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.remote.control_service import ControlOptions
from zcu_tools.gui.runtime import GuiAssembly, GuiRuntimeBehavior, GuiRuntimeSpec
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.alias_presets import AliasPresets, validate_alias_presets
from zcu_tools.gui.session.hardware_gate import RunBlocksHardwareGate
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.persistence import SingleFileCaretaker
from zcu_tools.gui.session.services.build import SessionServices, build_session_services
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from .attachment import Attachment
from .catalog import WorkflowCatalog
from .window import WorkflowWindow


class WorkflowsGuiBehavior[C](GuiRuntimeBehavior):
    """Compose a fresh EMPTY session with one Window and native Controller.

    state and services are the shared session owners created by assemble.
    window is its sole WorkflowWindow, also used by the optional RPC adapter.
    Keep this behavior alive through Qt shutdown. Parsed plan/knobs input is
    restored before display and flushed on close. Construction never connects
    instruments or restores native runs, session sources or hardware.
    """

    spec: ClassVar[GuiRuntimeSpec] = GuiRuntimeSpec(
        app_name="workflows",
        app_slug="workflows",
        default_control_port=0,
    )
    state: SessionState
    services: SessionServices
    window: WorkflowWindow[C]

    def __init__(
        self,
        *,
        registrations: Sequence[Attachment[C]],
        templates: Mapping[str, TemplateDef],
        loader: SourceCatalogLoader[WorkflowCatalog[C]] | None = None,
        alias_presets: AliasPresets | None = None,
        project_root: str = "",
        clean: bool = False,
        cache_dir: Path | None = None,
    ) -> None:
        """Configure roots and input restoration without I/O or Qt construction.

        registrations is a nonempty ordered collection of coherent workflow/form
        attachments sharing opaque context C. The first is default; empty or
        duplicate names raise ValueError before assembly. loader optionally loads
        replacement WorkflowCatalog[C] instances through the same registry;
        None disables source reload. Caller configures bounded/fixed namespaces
        and a builder that imports current registrations on every load.
        templates is the caller's persistent module/waveform template catalog,
        used by shared Inspect's New action; an empty mapping is valid. It is
        passed directly to the Window, without importing a user package.
        alias_presets maps names to ordered authored rows. Omitted or None means
        no catalog and an empty initial table. An explicit mapping must contain
        "default" (possibly empty), or raise ValueError. Invalid table syntax,
        duplicates or forward links raise InvalidInputError. The public
        alias_presets field retains the complete catalog for dialog composition;
        no file location or reload is retained. Assembly uses default rows for
        entries without saved aliases; saved empty and nonempty tables win.
        Presets never declare workflow roles or choose connected devices.

        Empty project_root means the launch cwd. clean skips reading the previous
        input file, but successful close still flushes current input. cache_dir
        overrides the shared user-cache directory, for embedding or isolated tests.
        Component definitions must be registered before Setup opens an entry.
        An existing QApplication is required for assemble, not for this call.
        """
        validate_alias_presets(alias_presets)
        self._registrations = tuple(registrations)
        WorkflowCatalog(self._registrations)
        self._loader = loader
        self._templates = templates
        self.alias_presets: AliasPresets | None = alias_presets
        self._project_root = project_root
        self._clean = clean
        self._cache_dir = cache_dir
        self._caretaker: (
            SingleFileCaretaker[WorkflowInputSnapshot, WorkflowInputRestoreReport]
            | None
        ) = None

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
            alias_rows=()
            if self.alias_presets is None
            else self.alias_presets["default"],
        )
        self.window = WorkflowWindow(
            self.state,
            self.services,
            self._owner,
            registrations=self._registrations,
            templates=self._templates,
            alias_presets=self.alias_presets,
            loader=self._loader,
            on_close=self._close,
        )
        self._caretaker = SingleFileCaretaker(
            self.window,
            codec=WorkflowInputCodec(),
            filename="t1_workflow_state_v1.json",
            cache_dir=self._cache_dir,
        )
        outcome = self._caretaker.restore_all(load=not self._clean)
        self.window.show_input_restore_message(outcome.report, outcome.load_error)
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

    def _close(self) -> None:
        if self._caretaker is None:
            raise RuntimeError("Workflow input persistence is not assembled")
        self._caretaker.flush()
        self.shutdown()

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
