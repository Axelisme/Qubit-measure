from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.session.state import (
    DEFAULT_LEFT_PANEL_WIDTH as DEFAULT_LEFT_PANEL_WIDTH,  # noqa: F401  (re-export)
)
from zcu_tools.gui.session.state import (
    DEVICE_SET_VERSION_KEY as DEVICE_SET_VERSION_KEY,  # noqa: F401  (re-export)
)
from zcu_tools.gui.session.state import (
    DeviceState as DeviceState,  # noqa: F401  (re-export)
)
from zcu_tools.gui.session.state import (
    DeviceStatus as DeviceStatus,  # noqa: F401  (re-export)
)
from zcu_tools.gui.session.state import (
    SessionState,
)
from zcu_tools.gui.session.state import (
    StartupPrefs as StartupPrefs,  # noqa: F401  (re-export)
)
from zcu_tools.gui.session.types import ExpContext

from .adapter import (
    AnalyzeResultWithFigure,
    ExpAdapterProtocol,
    SavePaths,
    T_AnalyzeParams,
    T_Cfg,
)

logger = logging.getLogger(__name__)

# VersionTable is the shared optimistic-concurrency mechanism (app-agnostic);
# re-exported so ``state.VersionTable`` stays resolvable. The session-core keys +
# bump↔drop contract live on SessionState; tab keys are bumped by State below.
from zcu_tools.gui.version_table import (
    VersionTable as VersionTable,  # noqa: E402  (re-export)
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from zcu_tools.gui.app.main.adapter import WritebackItem

T_Result = TypeVar("T_Result")
T_AnalyzeResult = TypeVar("T_AnalyzeResult", bound=AnalyzeResultWithFigure)


@dataclass
class Session(Generic[T_Cfg, T_Result, T_AnalyzeResult, T_AnalyzeParams]):
    adapter_name: str
    adapter: ExpAdapterProtocol
    # Committed cfg SSOT for this tab. The tab's CfgFormWidget LiveModel is the
    # runtime draft; it auto-commits here through Controller.update_tab_cfg on
    # every change. Run / Save / Session persistence read this field, never
    # the live form.
    cfg_schema: CfgSchema
    run_result: T_Result | None = None
    result_source_path: str | None = None
    analyze_result: T_AnalyzeResult | None = None
    figure: Figure | None = None
    analyze_param_instance: T_AnalyzeParams | None = None
    # Post-analysis layer (方案 A): parallel ``post_*`` fields that depend on the
    # primary ``analyze_result``. Invalidated (cleared) whenever the primary
    # analyze result changes or the run re-runs — a post result computed from a
    # stale primary fit must never linger. ``post_analyze_param_instance`` holds
    # the user's chosen post params (mirrors ``analyze_param_instance``).
    post_analyze_result: Any = None
    post_figure: Figure | None = None
    post_analyze_param_instance: Any = None
    save_path_overrides: SavePaths | None = None
    # Persistent writeback draft (ADR-0008): computed once when analyze finishes,
    # read/edited in place by UI + agent, applied as-is. Module/waveform items
    # carry a gc=False CfgEditorService model (editor_id); cleared + torn down on
    # rerun / reanalyze.
    writeback_items: list[WritebackItem] = field(default_factory=list)
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

    def has_post_analyze_result(self) -> bool:
        return self.post_analyze_result is not None

    def has_figure(self) -> bool:
        return self.figure is not None

    def effective_save_paths(self, ctx: ExpContext) -> SavePaths | None:
        """Resolve the tab's save paths: user override, else adapter suggestion
        derived from ``ctx`` (None until the context can suggest a path). Pure —
        a tab answering about its own save destination."""
        if self.save_path_overrides is not None:
            return self.save_path_overrides
        if not ctx.database_path or not ctx.result_dir or not ctx.active_label:
            return None
        return self.adapter.make_save_paths(ctx)


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
    # Post-analysis (second layer) facts — gate the Post sub-tab. The post form
    # is enabled once a primary analyze result exists; the post figure/summary
    # render once a post result exists.
    has_post_analyze_result: bool = False


class State(SessionState):
    """Passive GUI state container shared by Controller and domain services.

    Extends ``SessionState`` (active context + device set + startup prefs + the
    shared version table) with measure-gui's experiment surface: the tabs and
    their run/analyze/save lifecycle. Tab version keys (``tab:<id>...``) bump the
    same shared table as the inherited session keys (decision 6).
    """

    def __init__(self, ctx: ExpContext) -> None:
        super().__init__(ctx)
        self.tabs: dict[str, Session[Any, Any, Any, Any]] = {}
        self.active_tab_id: str | None = None
        self.running_tab_id: str | None = None

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
        """Tab ids in current display order — callers ask the aggregate, not the dict."""
        return list(self.tabs.keys())

    def reorder_tabs(self, tab_ids: Sequence[str]) -> None:
        """Replace the tab display order without replacing Session objects."""
        new_order = list(tab_ids)
        if len(new_order) != len(set(new_order)):
            raise ValueError(f"duplicate tab_id in reorder: {new_order!r}")
        if set(new_order) != set(self.tabs):
            raise ValueError(
                "reorder_tabs must contain exactly the current tabs: "
                f"got {new_order!r}, expected {list(self.tabs)!r}"
            )
        logger.debug("reorder_tabs: tab_ids=%r", new_order)
        self.tabs = {tab_id: self.tabs[tab_id] for tab_id in new_order}

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
        tab.result_source_path = None
        tab.analyze_result = None
        tab.figure = None
        tab.analyze_param_instance = None
        tab.writeback_items = []
        tab.applied_session_ids.clear()
        self._invalidate_post_analyze(tab)
        self.version.bump(f"tab:{tab_id}:result")
        self.version.bump(f"tab:{tab_id}:analyze")
        self.version.bump(f"tab:{tab_id}:post_analyze")

    def update_tab_result(self, tab_id: str, result: object) -> None:
        logger.debug(
            "update_tab_result: tab_id=%r result_type=%s", tab_id, type(result).__name__
        )
        tab = self.tabs[tab_id]
        tab.run_result = result
        tab.result_source_path = None
        tab.analyze_param_instance = None
        # invalidate stale analyze results and figure from the previous run
        tab.analyze_result = None
        tab.figure = None
        # New run → the previous run's writeback draft is stale. Callers must
        # teardown the per-item editor models (WritebackService) before this.
        tab.writeback_items = []
        tab.applied_session_ids.clear()
        # New run → any post-analysis built on the previous analyze result is also
        # stale (post depends on the primary analyze, which is cleared above).
        self._invalidate_post_analyze(tab)
        self.version.bump(f"tab:{tab_id}:result")
        self.version.bump(f"tab:{tab_id}:post_analyze")

    def update_tab_loaded_result(
        self, tab_id: str, result: object, source_path: str
    ) -> None:
        logger.debug(
            "update_tab_loaded_result: tab_id=%r source_path=%r result_type=%s",
            tab_id,
            source_path,
            type(result).__name__,
        )
        tab = self.tabs[tab_id]
        tab.run_result = result
        tab.result_source_path = source_path
        tab.analyze_param_instance = None
        tab.analyze_result = None
        tab.figure = None
        tab.writeback_items = []
        tab.applied_session_ids.clear()
        self._invalidate_post_analyze(tab)
        self.version.bump(f"tab:{tab_id}:result")
        self.version.bump(f"tab:{tab_id}:analyze")
        self.version.bump(f"tab:{tab_id}:post_analyze")

    def update_tab_analyze(
        self,
        tab_id: str,
        analyze_result: object,
        figure: Figure | None,
        writeback_items: list[WritebackItem] | None = None,
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
        # A re-analyze replaces the primary result the post-analysis depends on,
        # so any existing post result is now stale (方案 A invalidation).
        self._invalidate_post_analyze(tab)
        # Analyze result is a guarded resource (writeback depends on it), mirroring
        # update_tab_result's tab:<id>:result bump.
        self.version.bump(f"tab:{tab_id}:analyze")
        self.version.bump(f"tab:{tab_id}:post_analyze")

    @staticmethod
    def _invalidate_post_analyze(tab: Session[Any, Any, Any, Any]) -> None:
        """Drop a tab's post-analysis fields back to empty (no version bump — the
        caller bumps ``tab:<id>:post_analyze`` alongside its own keys). Post-
        analysis depends on the primary analyze result, so it is cleared at every
        out-edge that changes/clears that result (run start, re-run, re-analyze)."""
        tab.post_analyze_result = None
        tab.post_figure = None
        tab.post_analyze_param_instance = None

    def update_tab_post_analyze(
        self,
        tab_id: str,
        post_analyze_result: object,
        figure: Figure | None,
    ) -> None:
        """Record a freshly computed post-analysis result + figure (mirrors
        ``update_tab_analyze``). Fast-fails if the tab has no primary analyze
        result — post-analysis requires the primary fit it builds on."""
        tab = self.tabs[tab_id]
        if not tab.has_analyze_result():
            raise RuntimeError(
                f"Cannot record post-analysis for tab {tab_id!r}: no primary "
                "analyze result"
            )
        logger.debug(
            "update_tab_post_analyze: tab_id=%r figure=%s",
            tab_id,
            "yes" if figure is not None else "none",
        )
        tab.post_analyze_result = post_analyze_result
        tab.post_figure = figure
        self.version.bump(f"tab:{tab_id}:post_analyze")

    def update_tab_post_analyze_param_instance(
        self, tab_id: str, instance: object
    ) -> None:
        logger.debug(
            "update_tab_post_analyze_param_instance: tab_id=%r instance_type=%s",
            tab_id,
            type(instance).__name__,
        )
        self.tabs[tab_id].post_analyze_param_instance = instance

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
        paths: SavePaths | None,
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
        # Run-lock transition affects whether a tab.run_start may proceed; the tab's
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
