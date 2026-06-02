"""Ports — interfaces that application services depend on instead of concrete
infrastructure (driven/secondary adapters).

DDD/Hexagonal: an application service must not depend on a concrete external
system (persistence, project file I/O, hardware driver); it depends on a *port*
(an interface) which a driven adapter implements. See ``docs/adr/0008`` §
"Driven Adapter" and the M1 milestone.

These ports are ``Protocol``s (structural), so the existing concrete services
(``StartupPersistenceService`` / ``SessionPersistenceService`` / ``IOManager``)
satisfy them without any inheritance change — M1 only narrows what each consumer
*sees* and lets tests inject in-memory fakes. Each port declares exactly the
methods its consumer calls (interface segregation), so a consumer cannot reach
infrastructure capability it has no business using.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Protocol,
    runtime_checkable,
)

from zcu_tools.progress_bar.base import ProgressTotal, ProgressValue

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from zcu_tools.gui.adapter import (
        AdapterCapabilities,
        CfgSchema,
        ExpContext,
        SavePaths,
        WritebackItem,
    )
    from zcu_tools.gui.state import TabInteractionState
    from zcu_tools.meta_tool import ModuleLibrary

    from .device import DeviceProtocol
    from .persistence_types import AppPersistedState


@dataclass(frozen=True)
class TabSnapshot:
    """Immutable full-state snapshot of one tab (contract-layer DTO).

    A single type for two consumers (formerly ``TabViewSnapshot`` + the on-disk
    ``PersistedTab``):

    - **render** (``TabService.get_snapshot``): every field populated, handed to
      the View to draw one tab.
    - **restore** (``TabService.new_tab(from_dict=...)``): only the serializable
      head fields carry meaning; the live fields below are ``None`` / empty.

    ``cfg_schema`` is always the *live* ``CfgSchema`` (resolved EvalValue), which
    the render path uses directly. The disk codec (``SessionPersistenceService``)
    converts ``cfg_schema`` ↔ raw at the file boundary, so the persisted form
    never leaks into the snapshot. Lives in ``ports`` (the contract layer) so an
    application service can pass it around without importing a sibling
    application-service module (ADR-0008).
    """

    adapter_name: str
    cfg_schema: "CfgSchema"
    # The user's explicit override only (None = follow the adapter suggestion).
    # This is the serializable save-path state — persist/restore round-trip it so
    # a reload never pins an adapter-derived path.
    save_paths_override: Optional["SavePaths"]
    # Live render-only fields; None / empty on the persist + restore paths.
    tab_id: Optional[str] = None
    interaction: Optional["TabInteractionState"] = None
    capabilities: Optional["AdapterCapabilities"] = None
    analyze_params: object | None = None
    writeback_items: "tuple[WritebackItem, ...]" = ()
    figure: Optional["Figure"] = None
    # Render-computed effective paths (override, else adapter suggestion from
    # ctx). The View shows this; it is *not* persisted (derivable on restore).
    save_paths: Optional["SavePaths"] = None


@dataclass(frozen=True)
class RestoreIssue:
    """One rejected tab during session restore (adapter missing / cfg invalid)."""

    subject: str
    message: str


@dataclass(frozen=True)
class RestoreReport:
    """Outcome of applying a persisted session: how many tabs restored, and the
    per-tab rejections to surface to the user."""

    restored_tabs: int
    rejected_tabs: tuple[RestoreIssue, ...]


@runtime_checkable
class PersistOriginatorPort(Protocol):
    """The Memento Originator surface the ``PersistenceCaretaker`` depends on.

    The Caretaker (a Driven Adapter doing only disk I/O) never touches State,
    services, or cfg — it only asks the originator (the Controller) for one
    immutable snapshot to write, and hands one back to restore. Two narrow
    methods keep the Caretaker decoupled from the whole Controller interface.
    """

    def capture_persisted_state(self) -> "AppPersistedState": ...
    def restore_persisted_state(
        self, state: "AppPersistedState"
    ) -> "RestoreReport": ...


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
class DriverFactoryPort(Protocol):
    """Constructs a live hardware driver from (type_name, address).

    The driven adapter that touches pyvisa / instrument classes. ``DeviceService``
    depends on this port (already injected as ``driver_factory``); the default
    implementation ``_default_driver_factory`` is the concrete adapter. Tests
    inject a fake factory to avoid real hardware.

    Note (M1 scope): the *live-driver registry* ``GlobalDeviceManager`` is a
    module-level singleton still accessed directly inside ``DeviceService``'s
    worker-thread static helpers (``_connect``/``_disconnect``/``_set_value``).
    That is the hardware-I/O boundary already isolated as static methods and
    controlled in tests via fixture cleanup; it is intentionally left un-ported
    in M1 to avoid churning the worker paths for marginal test value. Revisit if
    a fake registry becomes needed.
    """

    def __call__(self, type_name: str, address: str) -> "DeviceProtocol": ...


@runtime_checkable
class ContextReadPort(Protocol):
    """Read-only view of the active context's ModuleLibrary, as a CfgEditor needs.

    A ``CfgEditorSession`` reads the current ml to seed a session opened
    ``from_name`` (load an existing entry's shape). Reading only — all ml/md
    *content writes* go through ``ContextWritePort`` (ADR-0011: ContextService is
    the single write authority). Symmetric name with ``ContextWritePort``.
    """

    def get_current_ml(self) -> "ModuleLibrary": ...


@runtime_checkable
class ContextWritePort(Protocol):
    """The single authority for ml/md content writes (ADR-0011).

    Sources holding an un-lowered ``CfgSchema`` (editor commit, writeback apply,
    inspect save, create_from_role) write through this port; ContextService
    lowers (``schema.to_raw_dict`` with the live md, so callers can never forget md)
    + registers + bumps the ``context`` version + emits ML/MD_CHANGED. The only
    implementer is ContextService.

    ``apply_writes`` is the batch entry: a single apply (writeback) of md attrs +
    multiple ml entries lands as **one** version bump and **at most one**
    ML_CHANGED + one MD_CHANGED (the per-write methods each bump/emit on their
    own; batching avoids N redundant full-refreshes).
    """

    def set_ml_module_from_schema(self, name: str, schema: "CfgSchema") -> None: ...
    def set_ml_waveform_from_schema(self, name: str, schema: "CfgSchema") -> None: ...
    def set_md_attr(self, key: str, value: Any) -> None: ...
    def apply_writes(self, writes: "ContextWrites") -> None: ...


@dataclass(frozen=True)
class ContextWrites:
    """A batch of ml/md content writes applied atomically (one bump + one emit
    per kind). ``md`` maps attr name → value; ``ml_modules`` / ``ml_waveforms``
    map entry name → its un-lowered ``CfgSchema``. Insertion order preserved."""

    md: "dict[str, Any]"
    ml_modules: "dict[str, CfgSchema]"
    ml_waveforms: "dict[str, CfgSchema]"


@runtime_checkable
class WritebackQueryPort(Protocol):
    """The writeback-items query as used by the tab read model.

    ``TabService.get_snapshot`` is a read-model assembler; it composes a tab's
    writeback proposals into the snapshot but must not depend on the concrete
    ``WritebackService`` (ADR-0008 violation 2 — no app-service→app-service
    coupling). It depends on this narrow query port instead, which prevents a
    back-edge from ever forming. ``WritebackService`` implements it.
    """

    def get_tab_writeback_items(self, tab_id: str) -> list["WritebackItem"]: ...


@runtime_checkable
class TabLifecyclePort(Protocol):
    """Tab create/restore/close + cfg as commanded by ``WorkspaceService``.

    ``WorkspaceService`` orchestrates the tab lifecycle (one-way command); it
    depends on this port, not the concrete ``TabService`` (ADR-0008 violation 2).
    """

    def new_tab(
        self, adapter_name: str, from_dict: Optional["TabSnapshot"] = None
    ) -> str: ...
    def close_tab(self, tab_id: str) -> None: ...
    def make_default_cfg(self, adapter_name: str) -> "CfgSchema": ...


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


@dataclass(frozen=True)
class DeviceMemoryInfo:
    """A remembered (memory-only) device's identity — the element type of
    ``RememberedDevicePort.register_remembered_devices``. Lives in the contract
    layer (ports) so both the device service and startup depend on it here, not
    on each other's module.
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

    def register_remembered_devices(self, entries: list[DeviceMemoryInfo]) -> None: ...


class ProgressEventKind(Enum):
    """The three things a worker progress bar tells the main thread."""

    CREATE = "create"  # a new bar appears (main thread stamps its start_time)
    UPDATE = "update"  # an existing bar advances / relabels
    CLOSE = "close"  # a bar leaves (leave=False bars only)


@dataclass(frozen=True)
class ProgressEvent:
    """One cross-thread progress notification.

    ``operation_id`` selects the owning :class:`ProgressContainer`; ``handle_id``
    selects a bar within it. The worker never picks a container — it emits these
    and the consumer (main thread) routes by these ids, so a bar created under
    one operation can never update under another.
    """

    operation_id: int
    handle_id: int
    kind: ProgressEventKind
    label: str = ""
    total: ProgressTotal = None  # meaningful on CREATE
    n: ProgressValue = 0  # meaningful on UPDATE


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
