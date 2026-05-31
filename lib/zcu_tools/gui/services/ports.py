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
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import CfgSchema, ExpContext, WritebackItem
    from zcu_tools.meta_tool import ModuleLibrary

    from .device import DeviceProtocol
    from .session_persistence import PersistedSession
    from .startup_persistence import PersistedDeviceEntry, PersistedStartup


@runtime_checkable
class StartupStorePort(Protocol):
    """Startup-preference persistence as used by ``StartupService``.

    Implemented by ``StartupPersistenceService``. Covers the remembered
    project/connection/device/left-panel settings store.
    """

    def load(self) -> Optional["PersistedStartup"]: ...
    def get_current(self) -> "PersistedStartup": ...
    def update_project(
        self,
        *,
        chip_name: str,
        qub_name: str,
        res_name: str,
        result_dir: str,
        database_path: str,
    ) -> None: ...
    def update_connection(self, *, ip: str, port: int) -> None: ...
    def replace_devices(self, entries: list["PersistedDeviceEntry"]) -> None: ...
    def update_left_panel_width(self, width: int) -> None: ...


@runtime_checkable
class SessionStorePort(Protocol):
    """Tab-session persistence + cfg codec as used by ``WorkspaceService``.

    Implemented by ``SessionPersistenceService``. The codec methods
    (``schema_to_raw`` / ``raw_to_schema``) are pure transforms bundled on the
    same concrete service; the port surfaces them alongside the session I/O
    because ``WorkspaceService`` needs both to round-trip a session.
    """

    def save_session(self, session: "PersistedSession") -> None: ...
    def load_session(self) -> Optional["PersistedSession"]: ...
    def schema_to_raw(self, schema: "CfgSchema", *, ml: Any) -> dict[str, object]: ...
    def raw_to_schema(
        self, base_schema: "CfgSchema", raw_cfg: dict[str, object]
    ) -> "CfgSchema": ...


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
        unit: str = "A",
        clone_from_current: bool = False,
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
    lowers (``schema_to_dict`` with the live md, so callers can never forget md)
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

    ``TabViewService`` is a read-model assembler; it composes a tab's writeback
    proposals into the snapshot but must not depend on the concrete
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

    def new_tab(self, adapter_name: str) -> str: ...
    def restore_tab(self, adapter_name: str) -> str: ...
    def close_tab(self, tab_id: str) -> None: ...
    def get_tab_default_cfg(self, tab_id: str) -> "CfgSchema": ...
    def update_tab_cfg(self, tab_id: str, schema: "CfgSchema") -> None: ...


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
