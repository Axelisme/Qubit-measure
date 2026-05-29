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

from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import CfgSchema, ExpContext
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
class ModuleLibraryWritePort(Protocol):
    """ModuleLibrary read + raw-entry registration as used by a CfgEditor commit.

    A ``CfgEditorSession`` (aggregate root) lowers its draft against the current
    ModuleLibrary and registers the concrete entry through this port — it no
    longer borrows the whole Controller (the old leaky ``_EditorCtrl``). The
    Controller (or any owner) implements it. Reading the current ml is needed
    both to lower ``EvalValue`` to concrete numbers and to seed a session opened
    ``from_name``.
    """

    def get_current_ml(self) -> "ModuleLibrary": ...
    def set_ml_module_from_raw(self, name: str, raw_dict: dict) -> None: ...
    def set_ml_waveform_from_raw(self, name: str, raw_dict: dict) -> None: ...
