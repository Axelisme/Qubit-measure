from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from typing_extensions import Generic, TypeVar

from .adapter import (
    AnalyzeResultWithFigure,
    CfgSchema,
    ExpAdapterProtocol,
    ExpContext,
    SavePaths,
    T_AnalyzeParams,
    T_Cfg,
)

logger = logging.getLogger(__name__)

# VersionTable is the shared optimistic-concurrency mechanism (app-agnostic);
# re-exported so ``state.VersionTable`` stays resolvable. The domain key set +
# bump↔drop contract are documented below beside the *_VERSION_KEY constants.
from zcu_tools.gui.version_table import (
    VersionTable as VersionTable,  # noqa: E402  (re-export)
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    # Imported only for typing: zcu_tools.device.base pulls in matplotlib.pyplot
    # at import time, which would break the GUI's "configure backend before
    # pyplot import" invariant if loaded eagerly here. ``from __future__ import
    # annotations`` keeps all annotations as strings, so a TYPE_CHECKING import
    # is sufficient.
    from zcu_tools.device.base import BaseDeviceInfo
    from zcu_tools.gui.app.main.adapter import WritebackItem

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


@dataclass(frozen=True)
class DeviceState:
    """Serializable device state — the SSOT for one device.

    State owns this; DeviceService holds only the live driver (in
    GlobalDeviceManager), the worker threads and the progress model. ``info`` is
    a ``BaseDeviceInfo`` value snapshot (not a live driver) and so lives here.
    ``remember`` is the persistent flag that drives the startup persistence
    projection — it is no longer a transient connect-request attribute.

    There is deliberately no ``progress`` field: setup progress is live
    telemetry owned by ``ProgressService`` (keyed by the operation token) and is
    polled separately via ``operation.progress`` (by operation_id) — never
    spliced into state.
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
class Session(Generic[T_Cfg, T_Result, T_AnalyzeResult, T_AnalyzeParams]):
    adapter_name: str
    adapter: ExpAdapterProtocol
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
    # Persistent writeback draft (ADR-0008): computed once when analyze finishes,
    # read/edited in place by UI + agent, applied as-is. Module/waveform items
    # carry a gc=False CfgEditorService model (editor_id); cleared + torn down on
    # rerun / reanalyze.
    writeback_items: list["WritebackItem"] = field(default_factory=list)
    applied_session_ids: set[str] = field(default_factory=set)
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

    def effective_save_paths(self, ctx: "ExpContext") -> Optional[SavePaths]:
        """Resolve the tab's save paths: user override, else adapter suggestion
        derived from ``ctx`` (None until the context can suggest a path). Pure —
        a tab answering about its own save destination."""
        if self.save_path_overrides is not None:
            return self.save_path_overrides
        if not ctx.database_path or not ctx.result_dir or not ctx.active_label:
            return None
        return self.adapter.make_save_paths(ctx)


# Set-cardinality version key for the whole device collection. Bumped only when
# the device *set* gains or loses a member (not on status/info edits of an
# existing member — those move that member's own ``device:<name>`` key). A
# guarded op depending on the whole set (run.start) declares this key so a
# concurrently-added device is detected, which a per-member glob cannot reveal.
DEVICE_SET_VERSION_KEY = "devices:__set__"


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


DEFAULT_LEFT_PANEL_WIDTH = 500


@dataclass
class StartupPrefs:
    """Remembered startup preferences — the *prefill* values, distinct from the
    active ``ExpContext``.

    These are what the setup dialog prefills and what persistence projects to
    disk; they are NOT the live connection/active-project state. Because the
    instrument never auto-connects on launch, there is no need to distinguish
    "currently connected to" from "remembered" — apply/connect just update these
    prefill values at write-time, and restore writes them back without applying
    a context. Mutable (State holds live mutable objects); a value-only block, so
    not versioned (no guarded op depends on it).
    """

    chip_name: str = ""
    qub_name: str = ""
    res_name: str = ""
    result_dir: str = ""
    database_path: str = ""
    ip: str = "192.168.10.1"
    port: int = 8887
    left_panel_width: int = DEFAULT_LEFT_PANEL_WIDTH


class State:
    """Passive GUI state container shared by Controller and domain services."""

    def __init__(self, ctx: ExpContext) -> None:
        self.exp_context: ExpContext = ctx
        self.tabs: dict[str, Session[Any, Any, Any, Any]] = {}
        self.active_tab_id: Optional[str] = None
        self.running_tab_id: Optional[str] = None
        # Remembered startup prefs (prefill values), distinct from exp_context.
        # StartupService writes at apply/connect; PersistenceCaretaker projects.
        self.startup_prefs: StartupPrefs = StartupPrefs()
        # Device state SSOT. DeviceService writes here (on the Qt main thread,
        # at its terminal slots) and holds only the live driver / worker / progress.
        self.devices: dict[str, DeviceState] = {}
        # Optimistic-concurrency version counters. Owned by State but bumped by
        # whichever main-thread writer actually mutates a resource (State's own
        # mutators below for context/tab/cfg/save_path/device; connection/run
        # services for soc/result at their terminal slots).
        self.version = VersionTable()

    def set_context(self, ctx: ExpContext) -> None:
        """Replace the whole ExpContext. Pure field swap — does NOT bump the
        ``context`` resource version, because the same setter is used to swap
        non-md/ml fields (soc/soccfg via connect, predictor via load/clear).

        The ``context`` version represents md/ml *content*: only callers that
        actually change md/ml (setup_project / use_context / new_context) bump
        it explicitly. soc has its own ``soc`` version key; predictor is not a
        guarded resource. This keeps soc-connect / predictor-load from spuriously
        marking md/ml-dependent ops (run / editor.commit / writeback) stale.

        The full set of "writes md/ml → bump context" paths is enumerated at the
        canonical anchor on ``ContextService.set_md_attr``.
        """
        self.exp_context = ctx

    # ------------------------------------------------------------------
    # Device state (DeviceService writes these on the Qt main thread).
    # Every *semantic write* bumps device:<name>; cache refresh does not
    # (see refresh_device_info_cache).
    # ------------------------------------------------------------------

    def put_device(self, dev: DeviceState) -> None:
        """Insert or replace a device entry (create / status transition)."""
        logger.debug("put_device: name=%r status=%s", dev.name, dev.status.value)
        is_new = dev.name not in self.devices
        self.devices[dev.name] = dev
        self.version.bump(f"device:{dev.name}")
        # Set-membership guard: a per-device version cannot reveal a *new* member
        # to an opt-in concurrency check (the agent never declared a key for a
        # device that did not exist when it read versions). The set-cardinality
        # key advances whenever the device set grows or shrinks, so an op that
        # depends on the whole set (run.start) detects a concurrently-added
        # device. Status transitions reuse an existing entry → set unchanged →
        # no bump here (the per-device key already moves).
        if is_new:
            self.version.bump(DEVICE_SET_VERSION_KEY)

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
        # The device set shrank — advance the set-cardinality key (symmetric to
        # the grow case in put_device) so a whole-set dependant detects it.
        self.version.bump(DEVICE_SET_VERSION_KEY)

    def get_device(self, name: str) -> Optional[DeviceState]:
        return self.devices.get(name)

    def has_device(self, name: str) -> bool:
        return name in self.devices

    def list_devices(self) -> tuple[DeviceState, ...]:
        return tuple(self.devices[name] for name in sorted(self.devices))

    def add_tab(
        self,
        tab_id: str,
        tab: Session[Any, Any, Any, Any],
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

    def get_tab(self, tab_id: str) -> Session:
        return self.tabs[tab_id]

    def has_tab(self, tab_id: str) -> bool:
        """Existence query — callers ask the aggregate, not the raw dict."""
        return tab_id in self.tabs

    def list_tab_ids(self) -> list[str]:
        """Tab ids in insertion order — callers ask the aggregate, not the dict."""
        return list(self.tabs.keys())

    def set_active_tab(self, tab_id: str) -> None:
        if tab_id not in self.tabs:
            raise KeyError(f"tab_id {tab_id!r} not found")
        logger.debug("set_active_tab: tab_id=%r", tab_id)
        self.active_tab_id = tab_id

    def clear_tab_results(self, tab_id: str) -> None:
        """Drop this tab's run/analyze/figure/writeback results back to empty.

        Called at the *start* of a run so that while the run is in flight (and
        after a failed/cancelled run) the tab honestly has no result — analyze /
        save then fail-fast with ``no_run_result`` (the true reason) instead of
        being blocked behind a stale previous result. The per-item writeback
        editor models must already be torn down by the caller (WritebackService),
        mirroring ``update_tab_result``.
        """
        logger.debug("clear_tab_results: tab_id=%r", tab_id)
        tab = self.tabs[tab_id]
        tab.run_result = None
        tab.analyze_result = None
        tab.figure = None
        tab.analyze_param_instance = None
        tab.writeback_items = []
        tab.applied_session_ids.clear()
        self.version.bump(f"tab:{tab_id}:result")
        self.version.bump(f"tab:{tab_id}:analyze")

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
        # New run → the previous run's writeback draft is stale. Callers must
        # teardown the per-item editor models (WritebackService) before this.
        tab.writeback_items = []
        tab.applied_session_ids.clear()
        self.version.bump(f"tab:{tab_id}:result")

    def update_tab_analyze(
        self,
        tab_id: str,
        analyze_result: object,
        figure: Optional["Figure"],
        writeback_items: Optional[list["WritebackItem"]] = None,
    ) -> None:
        logger.debug(
            "update_tab_analyze: tab_id=%r figure=%s",
            tab_id,
            "yes" if figure is not None else "none",
        )
        tab = self.tabs[tab_id]
        tab.analyze_result = analyze_result
        tab.figure = figure
        # Fresh analyze → store the freshly computed persistent writeback draft
        # (the sink computes it via WritebackService). Per-item models from a
        # previous analyze must already have been torn down by the caller.
        tab.writeback_items = list(writeback_items or [])
        tab.applied_session_ids.clear()
        # Analyze result is a guarded resource (writeback depends on it), mirroring
        # update_tab_result's tab:<id>:result bump.
        self.version.bump(f"tab:{tab_id}:analyze")

    def update_tab_cfg_schema(self, tab_id: str, schema: CfgSchema) -> None:
        logger.debug("update_tab_cfg_schema: tab_id=%r", tab_id)
        self.tabs[tab_id].cfg_schema = schema
        self.version.bump(f"tab:{tab_id}:cfg")

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
