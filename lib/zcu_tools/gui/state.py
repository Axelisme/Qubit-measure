from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from typing_extensions import Generic, TypeVar

from .adapter import (
    AbsExpAdapter,
    AnalyzeResultWithFigure,
    CfgSchema,
    ExpContext,
    SavePaths,
    T_AnalyzeParams,
    T_Cfg,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    # Imported only for typing: zcu_tools.device.base pulls in matplotlib.pyplot
    # at import time, which would break the GUI's "configure backend before
    # pyplot import" invariant if loaded eagerly here. ``from __future__ import
    # annotations`` keeps all annotations as strings, so a TYPE_CHECKING import
    # is sufficient.
    from zcu_tools.device.base import BaseDeviceInfo

T_Result = TypeVar("T_Result")
T_AnalyzeResult = TypeVar("T_AnalyzeResult", bound=AnalyzeResultWithFigure)


class DeviceStatus(Enum):
    """Lifecycle status of a device entry held in State.

    ``MEMORY_ONLY`` is a remembered-but-not-live entry (no driver in
    GlobalDeviceManager). All other values are *live* statuses: the device has
    a driver registered and is either idle (``CONNECTED``) or in a transient
    operation. The cross-object invariant is: a device is in GlobalDeviceManager
    iff its State status is a live status.
    """

    MEMORY_ONLY = "memory_only"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    SETTING_UP = "setting_up"
    SETTING_VALUE = "setting_value"


@dataclass(frozen=True)
class DeviceState:
    """Serializable device state — the SSOT for one device.

    State owns this; DeviceService holds only the live driver (in
    GlobalDeviceManager), the worker threads and the progress model. ``info`` is
    a ``BaseDeviceInfo`` value snapshot (not a live driver) and so lives here.
    ``remember`` is the persistent flag that drives the startup persistence
    projection — it is no longer a transient connect-request attribute.

    There is deliberately no ``progress`` field: setup progress is live
    (``DeviceSetupProgressModel`` in DeviceService) and is spliced in only at the
    ``DeviceSnapshot`` projection boundary.
    """

    name: str
    type_name: str
    address: str
    status: DeviceStatus
    remember: bool
    info: BaseDeviceInfo | None = None
    error: str | None = None

    # -- status predicates (the entity answers questions about itself) -----

    def is_memory_only(self) -> bool:
        return self.status is DeviceStatus.MEMORY_ONLY

    def is_connected(self) -> bool:
        """Idle and live (a driver is registered, no operation in flight)."""
        return self.status is DeviceStatus.CONNECTED

    def is_live(self) -> bool:
        """A driver is registered (connected or in a transient operation) — the
        cross-object invariant's 'in GlobalDeviceManager' side."""
        return self.status is not DeviceStatus.MEMORY_ONLY


@dataclass
class TabState(Generic[T_Cfg, T_Result, T_AnalyzeResult, T_AnalyzeParams]):
    adapter_name: str
    adapter: AbsExpAdapter[T_Cfg, T_Result, T_AnalyzeResult, T_AnalyzeParams]
    # Committed cfg SSOT for this tab. The tab's CfgFormWidget LiveModel is the
    # runtime draft; it auto-commits here through Controller.update_tab_cfg on
    # every change. Run / Save / Session persistence read this field, never
    # the live form.
    cfg_schema: CfgSchema
    run_result: Optional[T_Result] = None
    analyze_result: Optional[T_AnalyzeResult] = None
    figure: Optional["Figure"] = None
    analyze_param_instance: Optional[T_AnalyzeParams] = None
    save_path_overrides: Optional[SavePaths] = None
    applied_writeback_keys: set[str] = field(default_factory=set)
    is_running: bool = False
    is_analyzing: bool = False
    is_saving_data: bool = False

    # -- predicates (the entity answers questions about itself) ------------

    def is_busy(self) -> bool:
        """Any per-tab operation in flight (run / analyze / save)."""
        return self.is_running or self.is_analyzing or self.is_saving_data

    def has_run_result(self) -> bool:
        return self.run_result is not None

    def has_analyze_result(self) -> bool:
        return self.analyze_result is not None

    def has_figure(self) -> bool:
        return self.figure is not None


class VersionTable:
    """Monotonic per-resource version counters (optimistic-concurrency guard).

    A passive container: each resource key maps to an integer that only ever
    increases by one per mutation. Callers (the resource-owning service, on the
    Qt main thread) ``bump`` a key when they actually write that resource's
    state. The guard compares an op's declared ``expected_versions`` against the
    current table atomically inside the main-thread dispatch sequence.

    Resource keys are mid-grained: ``context``, ``soc``, ``device:<name>``,
    ``tab:<id>:cfg`` / ``:result`` / ``:save_path``, ``tab:<id>`` (existence)
    and ``editor:<id>``. A key absent from the table means version 0 (never
    bumped, or its resource was dropped — both read as "gone" by the guard).
    """

    def __init__(self) -> None:
        self._versions: dict[str, int] = {}

    def bump(self, key: str) -> int:
        new = self._versions.get(key, 0) + 1
        self._versions[key] = new
        logger.debug("version bump: %s -> %d", key, new)
        return new

    def get(self, key: str) -> int:
        """Current version of ``key`` (0 if never bumped / dropped)."""
        return self._versions.get(key, 0)

    def snapshot(self) -> dict[str, int]:
        """Full table copy (the ``resources.versions`` RPC payload)."""
        return dict(self._versions)

    def drop_prefix(self, prefix: str) -> None:
        """Forget every key starting with ``prefix`` (e.g. a closed tab).

        A dependency on a dropped key reads as version 0, which the guard
        treats as stale (the resource the caller depended on is gone).
        """
        doomed = [k for k in self._versions if k.startswith(prefix)]
        for k in doomed:
            del self._versions[k]
        if doomed:
            logger.debug("version drop_prefix: %s -> dropped %s", prefix, doomed)


@dataclass(frozen=True)
class TabInteractionState:
    global_run_active: bool
    is_running: bool
    is_analyzing: bool
    is_saving_data: bool
    has_context: bool
    has_active_context: bool
    has_soc: bool
    has_run_result: bool
    has_analyze_result: bool
    has_figure: bool


class State:
    """Passive GUI state container shared by Controller and domain services."""

    def __init__(self, ctx: ExpContext) -> None:
        self.exp_context: ExpContext = ctx
        self.tabs: dict[str, TabState[Any, Any, Any, Any]] = {}
        self.active_tab_id: Optional[str] = None
        self.running_tab_id: Optional[str] = None
        # Device state SSOT. DeviceService writes here (on the Qt main thread,
        # at its terminal slots) and holds only the live driver / worker / progress.
        self.devices: dict[str, DeviceState] = {}
        # Optimistic-concurrency version counters. Owned by State but bumped by
        # whichever main-thread writer actually mutates a resource (State's own
        # mutators below for context/tab/cfg/save_path/device; connection/run
        # services for soc/result at their terminal slots).
        self.version = VersionTable()

    def set_context(self, ctx: ExpContext) -> None:
        self.exp_context = ctx
        self.version.bump("context")

    # ------------------------------------------------------------------
    # Device state (DeviceService writes these on the Qt main thread).
    # Every *semantic write* bumps device:<name>; cache refresh does not
    # (see refresh_device_info_cache).
    # ------------------------------------------------------------------

    def put_device(self, dev: DeviceState) -> None:
        """Insert or replace a device entry (create / status transition)."""
        logger.debug("put_device: name=%r status=%s", dev.name, dev.status.value)
        self.devices[dev.name] = dev
        self.version.bump(f"device:{dev.name}")

    def set_device_status(
        self, name: str, status: DeviceStatus, *, error: Optional[str] = None
    ) -> None:
        logger.debug("set_device_status: name=%r status=%s", name, status.value)
        self.devices[name] = replace(self.devices[name], status=status, error=error)
        self.version.bump(f"device:{name}")

    def set_device_info(self, name: str, info: BaseDeviceInfo | None) -> None:
        """Semantic info update (after set-value / setup); bumps version."""
        logger.debug("set_device_info: name=%r info=%s", name, info is not None)
        self.devices[name] = replace(self.devices[name], info=info)
        self.version.bump(f"device:{name}")

    def set_device_remember(self, name: str, remember: bool) -> None:
        logger.debug("set_device_remember: name=%r remember=%s", name, remember)
        self.devices[name] = replace(self.devices[name], remember=remember)
        self.version.bump(f"device:{name}")

    def refresh_device_info_cache(self, name: str, info: BaseDeviceInfo) -> bool:
        """Refresh the cached driver info on a *read*; return whether it changed.

        A live read (``get_device_info``) caches the freshest driver info back
        into State. The principle is "bump == state actually changed", not "bump
        == a client wrote":

        - **info unchanged** → pure cache sync; do NOT bump (a redundant refresh
          must not spuriously invalidate other clients' ``expected_versions``).
        - **info changed** → the device's real value moved underneath us (e.g.
          external hardware change). That *is* a genuine state change, so bump
          ``device:<name>`` and return True so the caller can emit DEVICE_CHANGED.
          Compared by full ``BaseDeviceInfo`` value equality (pydantic ``==``).
        """
        current = self.devices[name]
        changed = current.info != info
        self.devices[name] = replace(current, info=info, error=None)
        if changed:
            self.version.bump(f"device:{name}")
        return changed

    def remove_device(self, name: str) -> None:
        logger.debug("remove_device: name=%r", name)
        del self.devices[name]
        # A stale dependency on a removed device now reads as version 0 (gone).
        self.version.drop_prefix(f"device:{name}")

    def get_device(self, name: str) -> Optional[DeviceState]:
        return self.devices.get(name)

    def has_device(self, name: str) -> bool:
        return name in self.devices

    def list_devices(self) -> tuple[DeviceState, ...]:
        return tuple(self.devices[name] for name in sorted(self.devices))

    def add_tab(
        self,
        tab_id: str,
        tab: TabState[Any, Any, Any, Any],
    ) -> None:
        if tab_id in self.tabs:
            raise ValueError(f"tab_id {tab_id!r} already exists")
        logger.debug(
            "add_tab: tab_id=%r adapter=%s",
            tab_id,
            type(tab.adapter).__name__,
        )
        self.tabs[tab_id] = tab
        self.version.bump(f"tab:{tab_id}")

    def remove_tab(self, tab_id: str) -> None:
        logger.debug("remove_tab: tab_id=%r", tab_id)
        if self.is_tab_busy(tab_id):
            raise RuntimeError(f"Cannot close busy tab {tab_id!r}")
        del self.tabs[tab_id]
        # Forget every version entry for this tab; a stale dependency on a
        # closed tab now reads as version 0 (gone) and the guard blocks.
        self.version.drop_prefix(f"tab:{tab_id}")
        if self.active_tab_id == tab_id:
            self.active_tab_id = None
        if self.running_tab_id == tab_id:
            self.running_tab_id = None

    def get_tab(self, tab_id: str) -> TabState:
        return self.tabs[tab_id]

    def set_active_tab(self, tab_id: str) -> None:
        if tab_id not in self.tabs:
            raise KeyError(f"tab_id {tab_id!r} not found")
        logger.debug("set_active_tab: tab_id=%r", tab_id)
        self.active_tab_id = tab_id

    def update_tab_result(self, tab_id: str, result: object) -> None:
        logger.debug(
            "update_tab_result: tab_id=%r result_type=%s", tab_id, type(result).__name__
        )
        tab = self.tabs[tab_id]
        tab.run_result = result
        tab.analyze_param_instance = None
        # invalidate stale analyze results and figure from the previous run
        tab.analyze_result = None
        tab.figure = None
        tab.applied_writeback_keys.clear()
        self.version.bump(f"tab:{tab_id}:result")

    def update_tab_analyze(
        self, tab_id: str, analyze_result: object, figure: Optional["Figure"]
    ) -> None:
        logger.debug(
            "update_tab_analyze: tab_id=%r figure=%s",
            tab_id,
            "yes" if figure is not None else "none",
        )
        tab = self.tabs[tab_id]
        tab.analyze_result = analyze_result
        tab.figure = figure
        tab.applied_writeback_keys.clear()

    def update_tab_cfg_schema(self, tab_id: str, schema: CfgSchema) -> None:
        logger.debug("update_tab_cfg_schema: tab_id=%r", tab_id)
        self.tabs[tab_id].cfg_schema = schema
        self.version.bump(f"tab:{tab_id}:cfg")

    def update_tab_analyze_params(self, tab_id: str, instance: object) -> None:
        logger.debug(
            "update_tab_analyze_params: tab_id=%r instance_type=%s",
            tab_id,
            type(instance).__name__,
        )
        self.tabs[tab_id].analyze_param_instance = instance

    def update_tab_analyze_param_instance(self, tab_id: str, instance: object) -> None:
        logger.debug(
            "update_tab_analyze_param_instance: tab_id=%r instance_type=%s",
            tab_id,
            type(instance).__name__,
        )
        self.tabs[tab_id].analyze_param_instance = instance

    def update_tab_save_path_overrides(
        self,
        tab_id: str,
        paths: Optional[SavePaths],
    ) -> None:
        logger.debug("update_tab_save_path_overrides: tab_id=%r", tab_id)
        self.tabs[tab_id].save_path_overrides = paths
        self.version.bump(f"tab:{tab_id}:save_path")

    def clear_tab_save_path_overrides(
        self,
        tab_id: str,
    ) -> None:
        logger.debug("clear_tab_save_path_overrides: tab_id=%r", tab_id)
        self.tabs[tab_id].save_path_overrides = None
        self.version.bump(f"tab:{tab_id}:save_path")

    def get_effective_save_paths(self, tab_id: str) -> Optional[SavePaths]:
        return self.tabs[tab_id].save_path_overrides

    def set_tab_running(self, tab_id: str, running: bool) -> None:
        logger.debug("set_tab_running: tab_id=%r running=%s", tab_id, running)
        tab = self.tabs[tab_id]
        if (
            running
            and self.running_tab_id is not None
            and self.running_tab_id != tab_id
        ):
            raise RuntimeError(
                f"Cannot mark tab {tab_id!r} running while "
                f"{self.running_tab_id!r} is already running"
            )
        tab.is_running = running
        if running:
            self.running_tab_id = tab_id
        elif self.running_tab_id == tab_id:
            self.running_tab_id = None
        # Run-lock transition affects whether a run.start may proceed; the tab's
        # own existence/run-state resource version moves with it.
        self.version.bump(f"tab:{tab_id}")

    def set_tab_analyzing(self, tab_id: str, analyzing: bool) -> None:
        logger.debug("set_tab_analyzing: tab_id=%r analyzing=%s", tab_id, analyzing)
        self.tabs[tab_id].is_analyzing = analyzing

    def set_tab_saving_data(self, tab_id: str, saving_data: bool) -> None:
        logger.debug(
            "set_tab_saving_data: tab_id=%r saving_data=%s", tab_id, saving_data
        )
        self.tabs[tab_id].is_saving_data = saving_data

    def is_run_active(self) -> bool:
        return self.running_tab_id is not None

    def is_tab_running(self, tab_id: str) -> bool:
        return self.tabs[tab_id].is_running

    def is_tab_analyzing(self, tab_id: str) -> bool:
        return self.tabs[tab_id].is_analyzing

    def is_tab_saving_data(self, tab_id: str) -> bool:
        return self.tabs[tab_id].is_saving_data

    def is_tab_busy(self, tab_id: str) -> bool:
        return self.tabs[tab_id].is_busy()
