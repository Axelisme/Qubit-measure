"""Session ↔ app seam — ports the session services depend on.

The session services (connection / device / context / ...) never reach for a
concrete app collaborator; they depend on the narrow Protocols here and the app
injects its concrete implementation. This keeps the session core app-agnostic and
free of a back-edge to any ``gui.app.*`` package.

This module holds the **exclusion seam**: the session-core operation-kind
vocabulary (``OperationKind``), the conflict error, and the ``ExclusionGate``
port. Each app keeps its own concrete ``OperationGate`` (the conflict *policy*)
and adds its own app-specific kinds (measure: ``run``; autofluxdep: a sweep kind);
the port is keyed by the kind's wire string so a session service can name a
session kind without the gate's full vocabulary leaking here (ADR-0019, decision
3 of the session-core extraction).

More driven-adapter ports (driver factory / project IO / progress transport) join
this module as the session services move in.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from zcu_tools.gui.plotting import FigureContainer
    from zcu_tools.gui.session.services.device import DeviceProtocol
    from zcu_tools.gui.session.types import ExpContext
    from zcu_tools.meta_tool import ModuleLibrary
    from zcu_tools.progress_bar.base import ProgressTotal, ProgressValue


class OperationKind(str, Enum):
    """Session-core hardware operation kinds (the ones session services emit).

    A ``str``-valued enum: a member IS its wire string, so it passes straight
    through the str-keyed :class:`ExclusionGate`. App-specific kinds (measure's
    ``run``, a future sweep kind) live in the app and are added to the app's gate
    policy — they are deliberately absent here.
    """

    SOC_CONNECT = "soc_connect"
    DEVICE_CONNECT = "device_connect"
    DEVICE_DISCONNECT = "device_disconnect"
    DEVICE_SETUP = "device_setup"


class OperationConflictError(RuntimeError):
    """Raised when a hardware operation conflicts with an active operation."""


class ExclusionGate(Protocol):
    """The hardware mutual-exclusion seam a session service depends on.

    ``kind`` is the operation kind's wire string (an ``OperationKind`` member
    passes directly, being a ``str``). The concrete gate (app-owned) holds the
    conflict policy across both session kinds and the app's own kinds; a session
    service only ever names session kinds through this port.
    """

    def ensure_can_start(self, kind: str) -> None:
        """Fail-fast: raise ``OperationConflictError`` if an active lease
        conflicts with ``kind`` (called before the handle is opened)."""
        ...

    def register(
        self,
        token: int,
        kind: str,
        *,
        owner_id: str,
        resource_id: Optional[str] = None,
    ) -> None:
        """Add an active exclusion lease under ``token`` (after ensure_can_start)."""
        ...

    def release(self, token: int) -> None:
        """Free the lease held by ``token`` on the terminal path."""
        ...

    def is_device_mutating(self, name: str) -> bool:
        """True while a device-mutation lease (connect/disconnect/setup) for
        ``name`` is active — guards a snapshot read against a racing mutation."""
        ...


@dataclass(frozen=True)
class OffMainScopes:
    """The opt-in ambient scopes a thunk runs inside (ADR-0019). Each is
    independently ``None``-able; only non-``None`` ones are entered, on the
    worker thread.

    - ``figure_container``: GUI matplotlib routing — sets the routing ContextVar
      *and* installs ``QtLivePlotBackend`` together (one facet: both direct
      ``plt.subplots`` and liveplot calls land in this container on the main
      thread; the liveplot backend requires the routing container, so they are
      co-dependent and driven by this single field).
    - ``pbar_factory``: progress — the per-operation pbar factory (the Progress
      facet's injection point; the owner mints it bound to the operation token).
    - ``stop_event``: cancel — installs ``ActiveTask`` so the work can self-
      interrupt cooperatively (the Cancel facet's off-main realisation).

    The value lives here (session seam) so the ``BackgroundExecutor`` port and the
    session services can name it; the *entering* logic (which pulls in the Qt
    liveplot backend) stays in the app's concrete ``BackgroundService``.
    """

    figure_container: Optional["FigureContainer"] = None
    pbar_factory: Optional[Callable[..., Any]] = None
    stop_event: Optional[threading.Event] = None


class BackgroundExecutor(Protocol):
    """Off-main execution seam a session service depends on. The app injects its
    concrete ``BackgroundService``; the session service never constructs one.

    Mirrors ``BackgroundService.submit``: run ``work`` off-main inside ``scopes``,
    delivering its result to ``on_done`` or its exception to ``on_error`` on the
    main thread. ``run_in_pool`` picks the shared pool vs a dedicated thread.
    """

    def submit(
        self,
        work: Callable[[], Any],
        scopes: Optional[OffMainScopes] = None,
        *,
        run_in_pool: bool,
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
    ) -> None: ...


class ProgressHub(Protocol):
    """Per-operation progress seam a session service depends on. The app injects
    its concrete ``ProgressService``.

    ``make_factory`` mints the per-operation pbar factory (placed into
    ``OffMainScopes.pbar_factory``); ``discard_operation`` drops the operation's
    progress container at the terminal path.
    """

    def make_factory(self, operation_id: int, owner_id: str) -> Callable[..., Any]: ...
    def discard_operation(self, operation_id: int) -> None: ...


class ProgressEventKind(Enum):
    """The three things a worker progress bar tells the main thread."""

    CREATE = "create"  # a new bar appears (main thread stamps its start_time)
    UPDATE = "update"  # an existing bar advances / relabels
    CLOSE = "close"  # a bar leaves (leave=False bars only)


@dataclass(frozen=True)
class ProgressEvent:
    """One cross-thread progress notification.

    ``operation_id`` selects the owning progress container; ``handle_id`` selects
    a bar within it. The worker never picks a container — it emits these and the
    consumer (main thread) routes by these ids, so a bar created under one
    operation can never update under another.
    """

    operation_id: int
    handle_id: int
    kind: ProgressEventKind
    label: str = ""
    total: "ProgressTotal" = None  # carried on CREATE and UPDATE (total may change)
    n: "ProgressValue" = 0  # meaningful on UPDATE


@runtime_checkable
class ProgressTransport(Protocol):
    """Carries progress events from worker threads to a main-thread consumer.

    The whole point of this port is to own the *cross-thread marshal* so the
    consumer (``ProgressService``) stays Qt-free and lock-free. Contract:

    - ``emit`` is thread-safe (called from any worker QThread, fire-and-forget).
    - the receiver registered via ``set_receiver`` is invoked on the *consumer's*
      thread (the main thread) — never on the worker thread.

    The Qt implementation (``QtProgressTransport``) realises the marshal with a
    queued signal connection; a synchronous in-memory implementation suffices
    for single-threaded tests.
    """

    def emit(self, event: ProgressEvent) -> None: ...

    def set_receiver(self, receiver: Callable[[ProgressEvent], None]) -> None: ...


@runtime_checkable
class DriverFactoryPort(Protocol):
    """Constructs a live hardware driver from (type_name, address).

    The driven adapter that touches pyvisa / instrument classes. ``DeviceService``
    depends on this port (injected as ``driver_factory``); the app's default
    implementation is the concrete adapter. Tests inject a fake factory to avoid
    real hardware.
    """

    def __call__(self, type_name: str, address: str) -> "DeviceProtocol": ...


@dataclass(frozen=True)
class DeviceMemoryInfo:
    """A remembered (memory-only) device's identity — the element type of
    ``RememberedDevicePort.register_remembered_devices``. Lives in the seam so both
    the device service and startup depend on it here, not on each other's module.
    """

    type_name: str
    name: str
    address: str


@runtime_checkable
class RememberedDevicePort(Protocol):
    """Remembered-device registration as used by ``StartupService.restore_devices``.

    The one device command startup issues; depends on the port, not the concrete
    ``DeviceService``.
    """

    def register_remembered_devices(
        self, entries: "list[DeviceMemoryInfo]"
    ) -> None: ...


@runtime_checkable
class ProjectIOPort(Protocol):
    """Experiment-project file I/O as used by ``ContextService``.

    Implemented by ``IOManager`` (which wraps ``ExperimentManager``). This is the
    file-backed project / flux-context store; the service never touches
    ``ExperimentManager`` directly.
    """

    @property
    def has_project(self) -> bool: ...
    def setup(self, result_dir: str) -> None: ...
    def list_contexts(self) -> list[str]: ...
    def get_active_label(self) -> Optional[str]: ...
    def use_context(self, label: str, base_ctx: "ExpContext") -> "ExpContext": ...
    def new_context(
        self,
        base_ctx: "ExpContext",
        value: Optional[float] = None,
        unit: str = "none",
        clone_from: Optional[str] = None,
    ) -> "ExpContext": ...


@runtime_checkable
class ContextReadPort(Protocol):
    """Read-only view of the active context's ModuleLibrary, as a CfgEditor needs.

    A ``CfgEditorSession`` reads the current ml to seed a session opened
    ``from_name`` (load an existing entry's shape). Reading only — all ml/md
    *content writes* go through the app's ``ContextWritePort`` (ADR-0006:
    ContextService is the single write authority). Symmetric name with
    ``ContextWritePort``.
    """

    def get_current_ml(self) -> "ModuleLibrary": ...


@runtime_checkable
class StartupContextPort(Protocol):
    """Context bootstrap commands as used by ``StartupService``.

    ``StartupService`` orchestrates project startup (one-way command into the
    context); it depends on this port, not the concrete ``ContextService``.
    """

    def set_startup_context(
        self,
        md: object,
        ml: object,
        chip_name: str,
        qub_name: str,
        res_name: str,
        result_dir: str,
        database_path: str,
    ) -> None: ...
    def setup_project(self, result_dir: str) -> None: ...
