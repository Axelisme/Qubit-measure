from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

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


# ``Session`` is the aggregate root, but its result-bearing resources are owned by
# fixed panes.  The pane objects are deliberately small value carriers: workers
# prepare a complete replacement, and State swaps the carrier on the owner thread.
# WritebackService remains the owner of the opaque draft itself.
@dataclass
class RunPaneState(Generic[T_Result]):
    result: T_Result | None = None
    source_path: str | None = None


@dataclass
class AnalysisPaneState(Generic[T_AnalyzeResult, T_AnalyzeParams]):
    params: T_AnalyzeParams | None = None
    result: T_AnalyzeResult | None = None
    figure: Figure | None = None
    writeback_draft: object | None = None
    image_path_override: str | None = None


@dataclass
class PostAnalysisPaneState(Generic[T_AnalyzeResult, T_AnalyzeParams]):
    params: T_AnalyzeParams | None = None
    result: T_AnalyzeResult | None = None
    figure: Figure | None = None
    writeback_draft: object | None = None
    image_path_override: str | None = None


@dataclass
class SavePaneState:
    """Save owns only the data-path override; image paths belong to image panes."""

    data_path_override: str | None = None


@dataclass(frozen=True, slots=True)
class RetiredRunResource:
    result: object | None = None
    source_path: str | None = None


@dataclass(frozen=True, slots=True)
class RetiredAnalysisResource:
    params: object | None = None
    result: object | None = None
    figure: Figure | None = None
    writeback_draft: object | None = None


@dataclass(frozen=True, slots=True)
class RetiredPaneResources:
    """Resources detached by one owner-thread State transition.

    The transition returns all detached drafts before any cleanup is attempted.
    This lets a service tear them down after the new pane is committed and makes
    cleanup failures non-transactional: State never needs to roll back a pane.
    """

    run: RetiredRunResource = field(default_factory=RetiredRunResource)
    analysis: RetiredAnalysisResource = field(default_factory=RetiredAnalysisResource)
    post_analysis: RetiredAnalysisResource = field(
        default_factory=RetiredAnalysisResource
    )

    @property
    def writeback_drafts(self) -> tuple[object, ...]:
        """All detached opaque drafts, de-duplicated by identity."""
        drafts: list[object] = []
        for candidate in (
            self.analysis.writeback_draft,
            self.post_analysis.writeback_draft,
        ):
            if candidate is not None and all(candidate is not old for old in drafts):
                drafts.append(candidate)
        return tuple(drafts)


_UNSET: object = object()


@dataclass
class Session(Generic[T_Cfg, T_Result, T_AnalyzeResult, T_AnalyzeParams]):
    adapter_name: str
    adapter: ExpAdapterProtocol
    # Committed cfg SSOT for this tab. The tab's CfgFormWidget LiveModel is the
    # runtime draft; it auto-commits here through Controller.update_tab_cfg on
    # every change. Run / Save / Session persistence read this field, never
    # the live form.
    cfg_schema: CfgSchema

    # Canonical pane-owned resources.
    run: RunPaneState[T_Result] = field(
        default_factory=lambda: cast(RunPaneState[T_Result], RunPaneState())
    )
    analysis: AnalysisPaneState[T_AnalyzeResult, T_AnalyzeParams] = field(
        default_factory=lambda: cast(
            AnalysisPaneState[T_AnalyzeResult, T_AnalyzeParams], AnalysisPaneState()
        )
    )
    post_analysis: PostAnalysisPaneState[Any, Any] = field(
        default_factory=PostAnalysisPaneState
    )
    save: SavePaneState = field(default_factory=SavePaneState)

    # State flags are tab interaction resources, not result ownership.
    is_analyzing: bool = False
    is_saving_data: bool = False

    # -- predicates (the entity answers questions about itself) ------------

    def has_run_result(self) -> bool:
        return self.run.result is not None

    def has_analyze_result(self) -> bool:
        return self.analysis.result is not None

    def has_post_analyze_result(self) -> bool:
        return self.post_analysis.result is not None

    def has_figure(self) -> bool:
        return self.analysis.figure is not None

    @staticmethod
    def _with_suffix(path: str, suffix: str) -> str:
        root, extension = os.path.splitext(path)
        return f"{root}{suffix}{extension}"

    def _adapter_save_paths(self, ctx: ExpContext) -> SavePaths | None:
        if not ctx.database_path or not ctx.result_dir or not ctx.active_label:
            return None
        return self.adapter.make_save_paths(ctx)

    def effective_data_path(self, ctx: ExpContext) -> str | None:
        if self.save.data_path_override is not None:
            return self.save.data_path_override
        paths = self._adapter_save_paths(ctx)
        return None if paths is None else paths.data_path

    def effective_analysis_image_path(self, ctx: ExpContext) -> str | None:
        if self.analysis.image_path_override is not None:
            return self.analysis.image_path_override
        paths = self._adapter_save_paths(ctx)
        return (
            None if paths is None else self._with_suffix(paths.image_path, "_analysis")
        )

    def effective_post_analysis_image_path(self, ctx: ExpContext) -> str | None:
        if self.post_analysis.image_path_override is not None:
            return self.post_analysis.image_path_override
        paths = self._adapter_save_paths(ctx)
        return (
            None
            if paths is None
            else self._with_suffix(paths.image_path, "_post_analysis")
        )


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
        self._assert_owner()
        if tab_id in self.tabs:
            raise ValueError(f"tab_id {tab_id!r} already exists")
        logger.debug(
            "add_tab: tab_id=%r adapter=%s",
            tab_id,
            type(tab.adapter).__name__,
        )
        self.tabs[tab_id] = tab
        self.version.bump(f"tab:{tab_id}")

    def remove_tab(self, tab_id: str) -> RetiredPaneResources:
        self._assert_owner()
        logger.debug("remove_tab: tab_id=%r", tab_id)
        if self.is_tab_busy(tab_id):
            raise RuntimeError(f"Cannot close busy tab {tab_id!r}")
        retired = self._retired_all(self.tabs[tab_id])
        del self.tabs[tab_id]
        # Forget every version entry for this tab; a stale dependency on a
        # closed tab now reads as version 0 (gone) and the guard blocks.
        self.version.drop_prefix(f"tab:{tab_id}")
        if self.active_tab_id == tab_id:
            self.active_tab_id = None
        if self.running_tab_id == tab_id:
            self.running_tab_id = None
        return retired

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
        self._assert_owner()
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
        self._assert_owner()
        if tab_id not in self.tabs:
            raise KeyError(f"tab_id {tab_id!r} not found")
        logger.debug("set_active_tab: tab_id=%r", tab_id)
        self.active_tab_id = tab_id

    @staticmethod
    def _retired_run(tab: Session[Any, Any, Any, Any]) -> RetiredRunResource:
        return RetiredRunResource(
            result=tab.run.result,
            source_path=tab.run.source_path,
        )

    @staticmethod
    def _retired_analysis(
        pane: AnalysisPaneState[Any, Any] | PostAnalysisPaneState[Any, Any],
    ) -> RetiredAnalysisResource:
        return RetiredAnalysisResource(
            params=pane.params,
            result=pane.result,
            figure=pane.figure,
            writeback_draft=pane.writeback_draft,
        )

    @staticmethod
    def _retired_all(tab: Session[Any, Any, Any, Any]) -> RetiredPaneResources:
        return RetiredPaneResources(
            run=State._retired_run(tab),
            analysis=State._retired_analysis(tab.analysis),
            post_analysis=State._retired_analysis(tab.post_analysis),
        )

    @staticmethod
    def _empty_analysis_like(
        pane: AnalysisPaneState[Any, Any] | PostAnalysisPaneState[Any, Any],
    ) -> AnalysisPaneState[Any, Any]:
        return AnalysisPaneState(image_path_override=pane.image_path_override)

    @staticmethod
    def _empty_post_analysis_like(
        pane: PostAnalysisPaneState[Any, Any] | AnalysisPaneState[Any, Any],
    ) -> PostAnalysisPaneState[Any, Any]:
        return PostAnalysisPaneState(image_path_override=pane.image_path_override)

    def _replace_run_pane(
        self,
        tab_id: str,
        pane: RunPaneState[Any],
    ) -> RetiredPaneResources:
        """Commit one complete run replacement and invalidate its dependents.

        No cleanup, adapter call, or validation occurs here. All potentially
        fallible work belongs before this owner-thread swap; the returned object
        is the complete detached-resource list for post-commit teardown.
        """
        self._assert_owner()
        tab = self.tabs[tab_id]
        retired = self._retired_all(tab)
        tab.run = pane
        tab.analysis = self._empty_analysis_like(tab.analysis)
        tab.post_analysis = self._empty_post_analysis_like(tab.post_analysis)
        self.version.bump(f"tab:{tab_id}:result")
        self.version.bump(f"tab:{tab_id}:analyze")
        self.version.bump(f"tab:{tab_id}:post_analyze")
        return retired

    def replace_run_pane(
        self,
        tab_id: str,
        pane: RunPaneState[Any],
    ) -> RetiredPaneResources:
        """Atomically replace Run and clear Analysis/Post resources."""
        return self._replace_run_pane(tab_id, pane)

    def swap_run_pane(
        self,
        tab_id: str,
        pane: RunPaneState[Any],
    ) -> RetiredPaneResources:
        """Alias for :meth:`replace_run_pane` used by lifecycle services."""
        return self.replace_run_pane(tab_id, pane)

    def clear_tab_results(self, tab_id: str) -> RetiredPaneResources:
        """Invalidate Run, Analysis and Post, returning all retired resources.

        Run start intentionally invalidates the old canonical result. The caller
        must perform any opaque-draft teardown *after* this swap; a failed/cancelled
        run therefore keeps the honest empty run state instead of restoring stale
        analysis content.
        """
        logger.debug("clear_tab_results: tab_id=%r", tab_id)
        return self._replace_run_pane(tab_id, RunPaneState())

    def update_tab_result(self, tab_id: str, result: object) -> RetiredPaneResources:
        self._assert_owner()
        logger.debug(
            "update_tab_result: tab_id=%r result_type=%s", tab_id, type(result).__name__
        )
        return self._replace_run_pane(tab_id, RunPaneState(result=result))

    def update_tab_loaded_result(
        self, tab_id: str, result: object, source_path: str
    ) -> RetiredPaneResources:
        self._assert_owner()
        logger.debug(
            "update_tab_loaded_result: tab_id=%r source_path=%r result_type=%s",
            tab_id,
            source_path,
            type(result).__name__,
        )
        return self._replace_run_pane(
            tab_id, RunPaneState(result=result, source_path=source_path)
        )

    def swap_analysis_pane(
        self,
        tab_id: str,
        pane: AnalysisPaneState[Any, Any],
    ) -> RetiredPaneResources:
        """Atomically commit a complete Analysis pane.

        The primary swap invalidates Post because Post consumes this exact
        analysis result. The previous Analysis and Post carriers are both
        returned so a service can tear down every retired draft after commit.
        """
        self._assert_owner()
        tab = self.tabs[tab_id]
        retired = RetiredPaneResources(
            analysis=State._retired_analysis(tab.analysis),
            post_analysis=State._retired_analysis(tab.post_analysis),
        )
        if pane.image_path_override is None:
            pane.image_path_override = tab.analysis.image_path_override
        tab.analysis = pane
        tab.post_analysis = self._empty_post_analysis_like(tab.post_analysis)
        self.version.bump(f"tab:{tab_id}:analyze")
        self.version.bump(f"tab:{tab_id}:post_analyze")
        return retired

    def replace_analysis_pane(
        self,
        tab_id: str,
        *,
        result: object,
        figure: Figure | None,
        params: object | None = None,
        writeback_draft: object | None = None,
        image_path_override: str | None = None,
    ) -> RetiredPaneResources:
        """Build and commit an Analysis carrier in one State transition."""
        retired = self.swap_analysis_pane(
            tab_id,
            AnalysisPaneState(
                params=params,
                result=result,
                figure=figure,
                writeback_draft=writeback_draft,
                image_path_override=image_path_override,
            ),
        )
        return retired

    def update_tab_analyze(
        self,
        tab_id: str,
        analyze_result: object,
        figure: Figure | None,
        writeback_draft: object | None = None,
        analyze_params_instance: object = _UNSET,
    ) -> RetiredPaneResources:
        self._assert_owner()
        tab = self.tabs[tab_id]
        params = (
            tab.analysis.params
            if analyze_params_instance is _UNSET
            else analyze_params_instance
        )
        logger.debug(
            "update_tab_analyze: tab_id=%r figure=%s",
            tab_id,
            "yes" if figure is not None else "none",
        )
        return self.replace_analysis_pane(
            tab_id,
            result=analyze_result,
            figure=figure,
            params=params,
            writeback_draft=writeback_draft,
        )

    @staticmethod
    def _reset_tab_derived(tab: Session[Any, Any, Any, Any]) -> None:
        """Clear state derived from a tab's current run result."""
        tab.analysis = State._empty_analysis_like(tab.analysis)
        tab.post_analysis = State._empty_post_analysis_like(tab.post_analysis)

    @staticmethod
    def _invalidate_post_analyze(tab: Session[Any, Any, Any, Any]) -> None:
        """Drop Post resources while preserving its independent image path."""
        tab.post_analysis = State._empty_post_analysis_like(tab.post_analysis)

    def swap_post_analysis_pane(
        self,
        tab_id: str,
        pane: PostAnalysisPaneState[Any, Any],
    ) -> RetiredPaneResources:
        """Atomically commit a complete Post pane without touching Analysis."""
        self._assert_owner()
        tab = self.tabs[tab_id]
        if not tab.has_analyze_result():
            raise RuntimeError(
                f"Cannot record post-analysis for tab {tab_id!r}: no primary "
                "analyze result"
            )
        retired = RetiredPaneResources(
            post_analysis=State._retired_analysis(tab.post_analysis),
        )
        if pane.image_path_override is None:
            pane.image_path_override = tab.post_analysis.image_path_override
        tab.post_analysis = pane
        self.version.bump(f"tab:{tab_id}:post_analyze")
        return retired

    def replace_post_analysis_pane(
        self,
        tab_id: str,
        *,
        result: object,
        figure: Figure | None,
        params: object | None = None,
        writeback_draft: object | None = None,
        image_path_override: str | None = None,
    ) -> RetiredPaneResources:
        """Build and commit a Post carrier in one State transition."""
        retired = self.swap_post_analysis_pane(
            tab_id,
            PostAnalysisPaneState(
                params=params,
                result=result,
                figure=figure,
                writeback_draft=writeback_draft,
                image_path_override=image_path_override,
            ),
        )
        return retired

    def update_tab_post_analyze(
        self,
        tab_id: str,
        post_analyze_result: object,
        figure: Figure | None,
        *,
        post_analyze_params_instance: object = _UNSET,
        writeback_draft: object | None = None,
    ) -> RetiredPaneResources:
        """Record a Post result while retaining the independent Analysis pane."""
        self._assert_owner()
        tab = self.tabs[tab_id]
        params = (
            tab.post_analysis.params
            if post_analyze_params_instance is _UNSET
            else post_analyze_params_instance
        )
        logger.debug(
            "update_tab_post_analyze: tab_id=%r figure=%s",
            tab_id,
            "yes" if figure is not None else "none",
        )
        retired = self.replace_post_analysis_pane(
            tab_id,
            result=post_analyze_result,
            figure=figure,
            params=params,
            writeback_draft=writeback_draft,
        )
        return retired

    def update_tab_post_analyze_param_instance(
        self, tab_id: str, instance: object
    ) -> None:
        self._assert_owner()
        logger.debug(
            "update_tab_post_analyze_param_instance: tab_id=%r instance_type=%s",
            tab_id,
            type(instance).__name__,
        )
        self.tabs[tab_id].post_analysis.params = instance

    def update_tab_cfg_schema(self, tab_id: str, schema: CfgSchema) -> None:
        self._assert_owner()
        logger.debug("update_tab_cfg_schema: tab_id=%r", tab_id)
        self.tabs[tab_id].cfg_schema = schema
        self.version.bump(f"tab:{tab_id}:cfg")

    def update_tab_analyze_param_instance(self, tab_id: str, instance: object) -> None:
        self._assert_owner()
        logger.debug(
            "update_tab_analyze_param_instance: tab_id=%r instance_type=%s",
            tab_id,
            type(instance).__name__,
        )
        self.tabs[tab_id].analysis.params = instance

    def _bump_path_versions(self, tab_id: str, *resources: str) -> None:
        for resource in resources:
            self.version.bump(f"tab:{tab_id}:path:{resource}")

    def update_tab_data_path_override(self, tab_id: str, data_path: str | None) -> None:
        self._assert_owner()
        self.tabs[tab_id].save.data_path_override = data_path
        self._bump_path_versions(tab_id, "data")

    def update_tab_analysis_image_path_override(
        self, tab_id: str, image_path: str | None
    ) -> None:
        self._assert_owner()
        self.tabs[tab_id].analysis.image_path_override = image_path
        self._bump_path_versions(tab_id, "analysis_image")

    def update_tab_post_analysis_image_path_override(
        self, tab_id: str, image_path: str | None
    ) -> None:
        self._assert_owner()
        self.tabs[tab_id].post_analysis.image_path_override = image_path
        self._bump_path_versions(tab_id, "post_analysis_image")

    def set_tab_running(self, tab_id: str, running: bool) -> None:
        self._assert_owner()
        logger.debug("set_tab_running: tab_id=%r running=%s", tab_id, running)
        _ = self.tabs[tab_id]
        if (
            running
            and self.running_tab_id is not None
            and self.running_tab_id != tab_id
        ):
            raise RuntimeError(
                f"Cannot mark tab {tab_id!r} running while "
                f"{self.running_tab_id!r} is already running"
            )
        if running:
            self.running_tab_id = tab_id
        elif self.running_tab_id == tab_id:
            self.running_tab_id = None
        # Run-lock transition affects whether a tab.run_start may proceed; the tab's
        # own existence/run-state resource version moves with it.
        self.version.bump(f"tab:{tab_id}")

    def set_tab_analyzing(self, tab_id: str, analyzing: bool) -> None:
        self._assert_owner()
        logger.debug("set_tab_analyzing: tab_id=%r analyzing=%s", tab_id, analyzing)
        self.tabs[tab_id].is_analyzing = analyzing

    def set_tab_saving_data(self, tab_id: str, saving_data: bool) -> None:
        self._assert_owner()
        logger.debug(
            "set_tab_saving_data: tab_id=%r saving_data=%s", tab_id, saving_data
        )
        self.tabs[tab_id].is_saving_data = saving_data

    def is_run_active(self) -> bool:
        return self.running_tab_id is not None

    def is_tab_running(self, tab_id: str) -> bool:
        _ = self.tabs[tab_id]
        return self.running_tab_id == tab_id

    def is_tab_analyzing(self, tab_id: str) -> bool:
        return self.tabs[tab_id].is_analyzing

    def is_tab_saving_data(self, tab_id: str) -> bool:
        return self.tabs[tab_id].is_saving_data

    def is_tab_busy(self, tab_id: str) -> bool:
        tab = self.tabs[tab_id]
        return self.running_tab_id == tab_id or tab.is_analyzing or tab.is_saving_data
