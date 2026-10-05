from __future__ import annotations

import logging
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast

from zcu_tools.gui.cfg.resource import CfgResource
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.state import (
    DEFAULT_LEFT_PANEL_WIDTH as DEFAULT_LEFT_PANEL_WIDTH,
)
from zcu_tools.gui.session.state import (
    DEVICE_SET_VERSION_KEY as DEVICE_SET_VERSION_KEY,
)
from zcu_tools.gui.session.state import (
    DeviceState as DeviceState,
)
from zcu_tools.gui.session.state import (
    DeviceStatus as DeviceStatus,
)
from zcu_tools.gui.session.state import (
    SessionPreferences as SessionPreferences,
)
from zcu_tools.gui.session.state import (
    SessionState,
)
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.version_table import (
    VersionTable as VersionTable,
)

from .adapter import (
    AnalysisMode,
    ExpAdapterProtocol,
    SavePaths,
    T_AnalyzeParams,
    T_Cfg,
)
from .artifact_paths import named_image_path
from .artifact_tracker import (
    ArtifactKey,
    ArtifactKind,
    ArtifactObservation,
    ArtifactSnapshot,
    ArtifactTracker,
)

logger = logging.getLogger(__name__)

# VersionTable is the shared optimistic-concurrency mechanism (app-agnostic);
# re-exported so ``state.VersionTable`` stays resolvable. The session-core keys +
# bump↔drop contract live on SessionState; tab keys are bumped by State below.

if TYPE_CHECKING:
    from zcu_tools.plotting.plots import Plots


T_Result = TypeVar("T_Result")
T_AnalyzeResult = TypeVar("T_AnalyzeResult")


# ``Session`` is the aggregate root, but its result-bearing resources are owned by
# fixed panes.  The pane objects are deliberately small value carriers: workers
# prepare a complete replacement, and State swaps the carrier on the owner thread.
# WritebackService remains the owner of the opaque draft itself.
@dataclass
class RunPaneState(Generic[T_Result]):
    result: T_Result | None = None
    source_path: str | None = None
    source_operation_id: int | None = None


@dataclass
class AnalysisPaneState(Generic[T_AnalyzeResult, T_AnalyzeParams]):
    params: T_AnalyzeParams | None = None
    result: T_AnalyzeResult | None = None
    plots: Plots | None = None
    writeback_draft: object | None = None
    image_path_overrides: dict[str, str] = field(default_factory=dict)
    source_operation_id: int | None = None
    result_params: T_AnalyzeParams | None = None


@dataclass
class PostAnalysisPaneState(Generic[T_AnalyzeResult, T_AnalyzeParams]):
    params: T_AnalyzeParams | None = None
    result: T_AnalyzeResult | None = None
    plots: Plots | None = None
    writeback_draft: object | None = None
    image_path_overrides: dict[str, str] = field(default_factory=dict)
    source_operation_id: int | None = None
    result_params: T_AnalyzeParams | None = None


@dataclass
class SavePaneState:
    """Save owns the data path and comment drafts; image paths belong to image panes."""

    data_path_override: str | None = None
    comment: str = ""


@dataclass(frozen=True, slots=True)
class RetiredRunResource:
    result: object | None = None
    source_path: str | None = None


@dataclass(frozen=True, slots=True)
class RetiredAnalysisResource:
    params: object | None = None
    result: object | None = None
    plots: Plots | None = None
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

    @property
    def plots(self) -> tuple[Plots, ...]:
        """Detached presentation owners, de-duplicated by identity."""
        plots: list[Plots] = []
        for candidate in (self.analysis.plots, self.post_analysis.plots):
            if candidate is not None and all(candidate is not old for old in plots):
                plots.append(candidate)
        return tuple(plots)


_UNSET: object = object()


@dataclass
class Session(Generic[T_Cfg, T_Result, T_AnalyzeResult, T_AnalyzeParams]):
    adapter_name: str
    adapter: ExpAdapterProtocol
    # A handle to the tab's sole cfg authority, not a second input tree.
    cfg: CfgResource

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
    artifacts: ArtifactTracker = field(default_factory=ArtifactTracker)

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
        return self.analysis.plots is not None and bool(self.analysis.plots)

    def _adapter_save_paths(self, ctx: SessionEnv) -> SavePaths | None:
        if not ctx.database_path or not ctx.result_dir or not ctx.active_label:
            return None
        return self.adapter.make_save_paths(ctx)

    def effective_data_path(self, ctx: SessionEnv) -> str | None:
        if self.save.data_path_override is not None:
            return self.save.data_path_override
        paths = self._adapter_save_paths(ctx)
        return None if paths is None else paths.data_path

    def image_pane(
        self, key: ArtifactKey
    ) -> AnalysisPaneState[Any, Any] | PostAnalysisPaneState[Any, Any]:
        if key.kind is ArtifactKind.ANALYSIS:
            return self.analysis
        if key.kind is ArtifactKind.POST_ANALYSIS:
            return self.post_analysis
        raise ValueError("Data artifacts have no image pane")

    def effective_image_path(self, ctx: SessionEnv, key: ArtifactKey) -> str | None:
        pane = self.image_pane(key)
        name = key.figure_name
        if name is None or pane.plots is None or name not in pane.plots:
            raise KeyError(f"No current image artifact {key!r}")
        override = pane.image_path_overrides.get(name)
        if override is not None:
            return override
        paths = self._adapter_save_paths(ctx)
        if paths is None:
            return None
        try:
            name.encode("utf-8")
        except UnicodeEncodeError:
            # Keep the artifact visible; save preflight rejects this name before I/O.
            return None
        return named_image_path(paths.image_path, key)


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

    Extends ``SessionState`` (active context + device set + preferences + the
    shared version table) with measure-gui's experiment surface: the tabs and
    their run/analyze/save lifecycle. Tab version keys (``tab:<id>...``) bump the
    same shared table as the inherited session keys (decision 6).
    """

    def __init__(self, ctx: SessionEnv) -> None:
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

    def get_tab(self, tab_id: str) -> Session[Any, Any, Any, Any]:
        return self.tabs[tab_id]

    def require_run_operation(self, tab_id: str, operation_id: int) -> None:
        """Reject a Run result replaced since the caller's operation."""
        self._assert_owner()
        pane = self.tabs[tab_id].run
        if pane.result is None or pane.source_operation_id != operation_id:
            raise FailedPreconditionError(
                f"Run no longer contains operation {operation_id}'s result",
                reason_code="result_superseded",
            )

    def require_analysis_operation(
        self,
        tab_id: str,
        subtab_id: Literal["analysis", "post_analysis"],
        operation_id: int,
    ) -> None:
        """Reject a result replaced since the caller's analysis operation."""
        self._assert_owner()
        tab = self.tabs[tab_id]
        pane = tab.analysis if subtab_id == "analysis" else tab.post_analysis
        if pane.result is None or pane.source_operation_id != operation_id:
            raise FailedPreconditionError(
                f"{subtab_id} no longer contains operation {operation_id}'s result",
                reason_code="result_superseded",
            )

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
            plots=pane.plots,
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
        return AnalysisPaneState(image_path_overrides=dict(pane.image_path_overrides))

    @staticmethod
    def _empty_post_analysis_like(
        pane: PostAnalysisPaneState[Any, Any] | AnalysisPaneState[Any, Any],
    ) -> PostAnalysisPaneState[Any, Any]:
        return PostAnalysisPaneState(
            image_path_overrides=dict(pane.image_path_overrides)
        )

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

    def update_tab_result(
        self, tab_id: str, result: object, *, source_operation_id: int | None = None
    ) -> RetiredPaneResources:
        self._assert_owner()
        logger.debug(
            "update_tab_result: tab_id=%r result_type=%s", tab_id, type(result).__name__
        )
        return self._replace_run_pane(
            tab_id, RunPaneState(result=result, source_operation_id=source_operation_id)
        )

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
        retired = self._replace_run_pane(
            tab_id, RunPaneState(result=result, source_path=source_path)
        )
        self.tabs[tab_id].artifacts.reset_for_load()
        return retired

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
        if not pane.image_path_overrides and pane.plots is not None:
            pane.image_path_overrides = {
                name: path
                for name, path in tab.analysis.image_path_overrides.items()
                if name in pane.plots
            }
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
        plots: Plots | None,
        params: object | None = None,
        writeback_draft: object | None = None,
        image_path_overrides: dict[str, str] | None = None,
    ) -> RetiredPaneResources:
        """Build and commit an Analysis carrier in one State transition."""
        return self.swap_analysis_pane(
            tab_id,
            AnalysisPaneState(
                params=params,
                result=result,
                plots=plots,
                writeback_draft=writeback_draft,
                image_path_overrides=(
                    {} if image_path_overrides is None else dict(image_path_overrides)
                ),
            ),
        )

    def update_tab_analyze(
        self,
        tab_id: str,
        analyze_result: object,
        plots: Plots | None,
        writeback_draft: object | None = None,
        analyze_params_instance: object = _UNSET,
        *,
        source_operation_id: int | None = None,
    ) -> RetiredPaneResources:
        self._assert_owner()
        tab = self.tabs[tab_id]
        params = (
            tab.analysis.params
            if analyze_params_instance is _UNSET
            else analyze_params_instance
        )
        logger.debug(
            "update_tab_analyze: tab_id=%r plots=%s",
            tab_id,
            "yes" if plots is not None else "none",
        )
        return self.swap_analysis_pane(
            tab_id,
            AnalysisPaneState(
                result=analyze_result,
                source_operation_id=source_operation_id,
                result_params=deepcopy(params),
                plots=plots,
                params=params,
                writeback_draft=writeback_draft,
            ),
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
        if not pane.image_path_overrides and pane.plots is not None:
            pane.image_path_overrides = {
                name: path
                for name, path in tab.post_analysis.image_path_overrides.items()
                if name in pane.plots
            }
        tab.post_analysis = pane
        self.version.bump(f"tab:{tab_id}:post_analyze")
        return retired

    def replace_post_analysis_pane(
        self,
        tab_id: str,
        *,
        result: object,
        plots: Plots | None,
        params: object | None = None,
        writeback_draft: object | None = None,
        image_path_overrides: dict[str, str] | None = None,
    ) -> RetiredPaneResources:
        """Build and commit a Post carrier in one State transition."""
        return self.swap_post_analysis_pane(
            tab_id,
            PostAnalysisPaneState(
                params=params,
                result=result,
                plots=plots,
                writeback_draft=writeback_draft,
                image_path_overrides=(
                    {} if image_path_overrides is None else dict(image_path_overrides)
                ),
            ),
        )

    def update_tab_post_analyze(
        self,
        tab_id: str,
        post_analyze_result: object,
        plots: Plots | None,
        *,
        post_analyze_params_instance: object = _UNSET,
        writeback_draft: object | None = None,
        source_operation_id: int | None = None,
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
            "update_tab_post_analyze: tab_id=%r plots=%s",
            tab_id,
            "yes" if plots is not None else "none",
        )
        return self.swap_post_analysis_pane(
            tab_id,
            PostAnalysisPaneState(
                result=post_analyze_result,
                source_operation_id=source_operation_id,
                result_params=deepcopy(params),
                plots=plots,
                params=params,
                writeback_draft=writeback_draft,
            ),
        )

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

    def get_artifact_snapshots(self, tab_id: str) -> tuple[ArtifactSnapshot, ...]:
        """Observe current panes and shared path/comment drafts on the owner thread.

        Include only capability-declared artifacts, in Data/Analysis/Post order.
        This is the single read model for Qt and the remote tab projection.
        """
        self._assert_owner()
        tab = self.get_tab(tab_id)
        observations = [
            ArtifactObservation(
                key=ArtifactKey(ArtifactKind.DATA),
                result=tab.run.result,
                figure=None,
                path=tab.effective_data_path(self.session_env),
                comment=tab.save.comment,
            )
        ]
        capabilities = tab.adapter.capabilities
        for kind, pane, available in (
            (
                ArtifactKind.ANALYSIS,
                tab.analysis,
                capabilities.analysis is not AnalysisMode.NONE,
            ),
            (ArtifactKind.POST_ANALYSIS, tab.post_analysis, capabilities.post_analysis),
        ):
            if not available or pane.plots is None or pane.result is None:
                continue
            for name, figure in pane.plots.items():
                key = ArtifactKey(kind, name)
                observations.append(
                    ArtifactObservation(
                        key=key,
                        result=pane.result,
                        figure=figure,
                        path=tab.effective_image_path(self.session_env, key),
                    )
                )
        return tab.artifacts.project(observations)

    def update_tab_comment(self, tab_id: str, comment: str) -> None:
        """Publish the Data comment draft shared by GUI and remote saves."""
        self._assert_owner()
        self.tabs[tab_id].save.comment = comment
        self.version.bump(f"tab:{tab_id}:save")

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

    def update_tab_image_path_override(
        self, tab_id: str, key: ArtifactKey, image_path: str | None
    ) -> None:
        self._assert_owner()
        pane = self.tabs[tab_id].image_pane(key)
        name = key.figure_name
        if name is None or pane.plots is None or name not in pane.plots:
            raise KeyError(f"No current image artifact {key!r}")
        if image_path is None:
            pane.image_path_overrides.pop(name, None)
        else:
            pane.image_path_overrides[name] = image_path
        stage = (
            "analysis_image"
            if key.kind is ArtifactKind.ANALYSIS
            else "post_analysis_image"
        )
        self._bump_path_versions(tab_id, stage)

    def set_tab_running(self, tab_id: str, *, running: bool) -> None:
        """Set or clear this existing tab's run ownership on the owner thread.

        True claims the sole run slot; False releases it only if this tab owns
        it. Bump this tab's version. Raise KeyError for an unknown tab and
        RuntimeError off-owner or when claiming another tab's occupied slot.
        """
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

    def set_tab_analyzing(self, tab_id: str, *, analyzing: bool) -> None:
        """Set this existing tab's analysis busy flag on the owner thread.

        True marks analysis active; False clears it. Raise KeyError for an
        unknown tab and RuntimeError off-owner. Do not bump resource versions.
        """
        self._assert_owner()
        logger.debug("set_tab_analyzing: tab_id=%r analyzing=%s", tab_id, analyzing)
        self.tabs[tab_id].is_analyzing = analyzing

    def set_tab_saving_data(self, tab_id: str, *, saving_data: bool) -> None:
        """Set this existing tab's save busy flag on the owner thread.

        True marks saving active; False clears it. Raise KeyError for an
        unknown tab and RuntimeError off-owner. Do not bump resource versions.
        """
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
