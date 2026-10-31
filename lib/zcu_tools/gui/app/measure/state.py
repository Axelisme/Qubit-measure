from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from zcu_tools.experiment.artifacts import ArtifactRef, SaveReport
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.registry import RegisteredAdapter
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource
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
from zcu_tools.gui.session.types import ParameterContext, SessionEnv
from zcu_tools.gui.version_table import (
    VersionTable as VersionTable,
)
from zcu_tools.resources.entry import StorageBinding

from .adapter import (
    T_AnalyzeParams,
    T_Cfg,
)
from .artifacts import (
    ArtifactSectionSnapshot,
    ArtifactTracker,
)

logger = logging.getLogger(__name__)

# VersionTable is the shared optimistic-concurrency mechanism (app-agnostic);
# re-exported so ``state.VersionTable`` stays resolvable. The session-core keys +
# bump↔drop contract live on SessionState; State and pane_commit own tab keys.

if TYPE_CHECKING:
    from zcu_tools.plotting.plots import Plots

    from .writeback import WritebackDraft


T_Result = TypeVar("T_Result")
T_AnalyzeResult = TypeVar("T_AnalyzeResult")


# ``Session`` is the aggregate root, but its result-bearing resources are owned by
# fixed panes.  The pane objects are deliberately small value carriers: workers
# prepare a complete replacement, and State swaps the carrier on the owner thread.
# WritebackService remains the owner of each typed draft.
@dataclass
class RunPaneState:
    """One result and its historical acquisition evidence.

    result is the adapter-owned record or None. source_path identifies loaded
    data, not an acquisition destination. source_operation_id identifies the
    publishing GUI operation. metadata is evidence from Run or native Load, or
    None when no native evidence exists; Load retains the file's original history.
    confirmed_run_id is None until a complete Save report confirms at least one
    persistent output. It survives failed resaves of this Run, but replacement
    or a new Run resets it; it is identity, not a successful-output history.
    parameters fixes analysis/WB values and target. Run uses admission values;
    native Load reads its historical point once using its resolved alias names.
    None means no readable analysis source, never a live-context fallback.
    storage fixes the Save capability: Run admission or Load request storage.
    None means no Save destination. Native Load may have a different analysis
    target; Save still checks this capability against file history.
    stop_requested is True when the frontend sent an accepted Stop or selector
    cancel during this Run, even if the core later failed. Core-owned stops do
    not set it. Non-acquisition replacement and new Run reset it to False.
    """

    result: RunRecord[Any, Any] | None = None
    source_path: str | None = None
    source_operation_id: int | None = None
    confirmed_run_id: str | None = None
    parameters: ParameterContext | None = None
    storage: StorageBinding | None = None
    stop_requested: bool = False
    # None for readable loaded history or non-Load; otherwise the fixed Load reason.
    analysis_source_issue: str | None = None


@dataclass
class AnalysisPaneState(Generic[T_AnalyzeResult, T_AnalyzeParams]):
    """Primary result and its owned resources.

    params is the editable input or None; result is the completed payload or None
    for figures-only output. plots owns presentation, or None without figures.
    writeback_draft is the creating service's draft, or None without proposals.
    source_operation_id identifies publication, or None for untracked setup.
    result_params snapshots params at completion, or None without inputs.
    figures_replaced records whether previous nonempty plots were replaced.
    """

    params: T_AnalyzeParams | None = None
    result: T_AnalyzeResult | None = None
    plots: Plots | None = None
    writeback_draft: WritebackDraft | None = None
    source_operation_id: int | None = None
    result_params: T_AnalyzeParams | None = None
    # Whether this publication replaced this Run's previous nonempty pane plots.
    figures_replaced: bool = False


@dataclass
class PostAnalysisPaneState(Generic[T_AnalyzeResult, T_AnalyzeParams]):
    """Post result and resources, independent from the primary pane.

    params is the editable input or None; result is the completed payload or None
    for figures-only output. plots owns presentation, or None without figures.
    writeback_draft is the creating service's draft, or None without proposals.
    source_operation_id identifies publication, or None for untracked setup.
    result_params snapshots params at completion, or None without inputs.
    figures_replaced records whether previous nonempty plots were replaced.
    """

    params: T_AnalyzeParams | None = None
    result: T_AnalyzeResult | None = None
    plots: Plots | None = None
    writeback_draft: WritebackDraft | None = None
    source_operation_id: int | None = None
    result_params: T_AnalyzeParams | None = None
    # Whether this publication replaced this Run's previous nonempty pane plots.
    figures_replaced: bool = False


@dataclass
class SavePaneState:
    """Current annotation; destinations and evidence belong to ArtifactTracker."""

    comment: str = ""


@dataclass(frozen=True, slots=True)
class PaneTransitionPreview:
    """Existing facts invalidated by replacing an Analysis or Post pane.

    analysis_result is True when the old Primary result is retired.
    analysis_writeback is True when the old Primary draft is retired.
    post_analysis_result is True when the old Post result is retired.
    post_analysis_writeback is True when the old Post draft is retired.
    Result presence includes numeric data, figures-only publications and published
    empty plot collections. False means absent or preserved by this transition.
    These facts contain no references to mutable resources or transport labels.
    """

    analysis_result: bool
    analysis_writeback: bool
    post_analysis_result: bool
    post_analysis_writeback: bool


@dataclass(frozen=True, slots=True)
class RetiredRunResource:
    result: object | None = None
    source_path: str | None = None


@dataclass(frozen=True, slots=True)
class RetiredAnalysisResource:
    """Detached stage values; None fields mean that resource was absent.

    params/result retain prior input/output, plots retains presentation ownership,
    and writeback_draft retains the draft to close after publication.
    """

    params: object | None = None
    result: object | None = None
    plots: Plots | None = None
    writeback_draft: WritebackDraft | None = None


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
    def writeback_drafts(self) -> tuple[WritebackDraft, ...]:
        """All detached drafts, de-duplicated by identity."""
        drafts: list[WritebackDraft] = []
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


@dataclass
class Session(Generic[T_Cfg, T_Result, T_AnalyzeResult, T_AnalyzeParams]):
    adapter_name: str
    adapter: RegisteredAdapter
    # A handle to the tab's sole cfg authority, not a second input tree.
    cfg: TabFormResource

    # Canonical pane-owned resources.
    run: RunPaneState = field(default_factory=lambda: RunPaneState())
    analysis: AnalysisPaneState[T_AnalyzeResult, T_AnalyzeParams] = field(
        default_factory=lambda: AnalysisPaneState()
    )
    post_analysis: PostAnalysisPaneState[Any, Any] = field(
        default_factory=PostAnalysisPaneState
    )
    save: SavePaneState = field(default_factory=SavePaneState)
    artifacts: ArtifactTracker = field(default_factory=ArtifactTracker)

    # State flags are tab interaction resources, not result ownership.
    is_analyzing: bool = False

    # -- predicates (the entity answers questions about itself) ------------

    def has_run_result(self) -> bool:
        return self.run.result is not None

    def has_analyze_result(self) -> bool:
        return self.analysis.result is not None or self.analysis.plots is not None

    def has_post_analyze_result(self) -> bool:
        return (
            self.post_analysis.result is not None
            or self.post_analysis.plots is not None
        )

    def has_figure(self) -> bool:
        return self.analysis.plots is not None and bool(self.analysis.plots)


@dataclass(frozen=True)
class TabInteractionState:
    """UI facts not replaced by domain admission decisions.

    global_run_active and startup_active describe separate hardware exclusion.
    is_running selects Stop; is_analyzing also locks local editing. has_context
    gates Load and has_active_context gates Save in their existing views.
    has_run_result supports result presentation and window event reactions.
    Analysis, Post and figure presence are read from their snapshot panes.
    """

    global_run_active: bool
    is_running: bool
    is_analyzing: bool
    has_context: bool
    has_active_context: bool
    has_run_result: bool
    # True while a startup script owns admission; only Run is unavailable.
    startup_active: bool = False


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
        self._startup_active = False

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
        if pane.source_operation_id != operation_id:
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

    def set_run_pane(
        self, tab_id: str, pane: RunPaneState, artifacts: ArtifactTracker
    ) -> RetiredPaneResources:
        """Swap prepared Run/artifacts and clear Analysis/Post on the owner thread.

        pane and artifacts are fully prepared by pane_commit. Return detached
        resources without cleanup or version changes. Unknown tabs raise KeyError;
        off-owner calls raise RuntimeError before mutation.
        """
        self._assert_owner()
        tab = self.tabs[tab_id]
        retired = self._retired_all(tab)
        tab.run = pane
        tab.artifacts = artifacts
        tab.analysis = AnalysisPaneState()
        tab.post_analysis = PostAnalysisPaneState()
        return retired

    @staticmethod
    def _retired_for_analysis_transition(
        tab: Session[Any, Any, Any, Any],
        stage: Literal["analysis", "post_analysis"],
    ) -> RetiredPaneResources:
        if stage == "analysis":
            return RetiredPaneResources(
                analysis=State._retired_analysis(tab.analysis),
                post_analysis=State._retired_analysis(tab.post_analysis),
            )
        if stage == "post_analysis":
            return RetiredPaneResources(
                post_analysis=State._retired_analysis(tab.post_analysis),
            )
        raise ValueError(f"Unknown analysis stage: {stage!r}")

    def preview_pane_transition(
        self, tab_id: str, stage: Literal["analysis", "post_analysis"]
    ) -> PaneTransitionPreview:
        """Read facts retired by the next successful replacement of stage.

        Call on-owner with a live tab and analysis (Primary) or post_analysis
        (Post). This query uses the same retirement selection as the State swap.
        It does not check admission or commit, plan artifacts, change versions,
        close drafts or release plots. Unknown tabs raise KeyError, off-owner
        calls raise RuntimeError and unsupported stages raise ValueError.
        """
        self._assert_owner()
        retired = self._retired_for_analysis_transition(self.tabs[tab_id], stage)
        return PaneTransitionPreview(
            analysis_result=(
                retired.analysis.result is not None
                or retired.analysis.plots is not None
            ),
            analysis_writeback=retired.analysis.writeback_draft is not None,
            post_analysis_result=(
                retired.post_analysis.result is not None
                or retired.post_analysis.plots is not None
            ),
            post_analysis_writeback=retired.post_analysis.writeback_draft is not None,
        )

    def set_analysis_pane(
        self,
        tab_id: str,
        pane: AnalysisPaneState[Any, Any],
        artifacts: ArtifactTracker,
    ) -> RetiredPaneResources:
        """Swap prepared Analysis/artifacts and clear Post, returning old resources.

        Call on-owner with a complete pane and tracker from pane_commit. This
        swap performs no planning, cleanup or version changes. Unknown tabs raise
        KeyError; off-owner calls raise RuntimeError before mutation.
        """
        self._assert_owner()
        tab = self.tabs[tab_id]
        retired = self._retired_for_analysis_transition(tab, "analysis")
        tab.analysis = pane
        tab.post_analysis = PostAnalysisPaneState()
        tab.artifacts = artifacts
        return retired

    def set_post_analysis_pane(
        self,
        tab_id: str,
        pane: PostAnalysisPaneState[Any, Any],
        artifacts: ArtifactTracker,
    ) -> RetiredPaneResources:
        """Swap prepared Post/artifacts without changing primary Analysis.

        Return detached Post resources without cleanup or version changes.
        Missing primary raises RuntimeError before mutation. Unknown tabs raise
        KeyError; off-owner calls raise RuntimeError. pane_commit prepares inputs.
        """
        self._assert_owner()
        tab = self.tabs[tab_id]
        if not tab.has_analyze_result():
            raise RuntimeError(
                f"Cannot record post-analysis for tab {tab_id!r}: no primary "
                "analyze result"
            )
        retired = self._retired_for_analysis_transition(tab, "post_analysis")
        tab.post_analysis = pane
        tab.artifacts = artifacts
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

    def commit_save_report(self, tab_id: str, report: SaveReport) -> None:
        """Publish complete outputs and confirmed identity on State's owner thread.

        Missing historical RunMetadata raises ValueError before mutation;
        unknown tabs raise KeyError. At least one saved output confirms run_id;
        a failed resave keeps that identity. Unselected artifacts and acquisition
        identity/times stay unchanged. This command owns no operation handle.
        """
        self._assert_owner()
        tab = self.get_tab(tab_id)
        metadata = None if tab.run.result is None else tab.run.result.metadata
        if metadata is None:
            raise ValueError("Save report requires historical RunMetadata")
        tab.artifacts.apply_report(report)
        if any(item.status == "saved" for item in report.outputs):
            tab.run.confirmed_run_id = report.run_id
        self.version.bump(f"tab:{tab_id}:save")

    def get_artifact_snapshots(
        self, tab_id: str
    ) -> tuple[ArtifactSectionSnapshot, ...]:
        """Read the committed section model without adapter calls or path planning."""
        self._assert_owner()
        return self.get_tab(tab_id).artifacts.snapshot()

    def update_tab_comment(self, tab_id: str, comment: str) -> None:
        """Publish the Data comment draft shared by GUI and remote saves."""
        self._assert_owner()
        tab = self.tabs[tab_id]
        if tab.save.comment == comment:
            return
        tab.save.comment = comment
        if any(section.section == "run" for section in tab.artifacts.snapshot()):
            tab.artifacts.mark_dirty(ArtifactRef("run", "measurement_data"))
        self.version.bump(f"tab:{tab_id}:save")

    def update_tab_analyze_param_instance(self, tab_id: str, instance: object) -> None:
        self._assert_owner()
        logger.debug(
            "update_tab_analyze_param_instance: tab_id=%r instance_type=%s",
            tab_id,
            type(instance).__name__,
        )
        self.tabs[tab_id].analysis.params = instance

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

    def set_startup_active(self, *, active: bool) -> None:
        """Publish startup activity on-owner for tab Run availability.

        active is True from script admission until owner terminal delivery.
        This memory-only fact does not acquire or release the hardware gate.
        Foreign-thread mutation raises RuntimeError.
        """
        self._assert_owner()
        self._startup_active = active

    def is_startup_active(self) -> bool:
        """Read the current memory-only startup Run lock without I/O, on-owner."""
        return self._startup_active

    def is_run_active(self) -> bool:
        return self.running_tab_id is not None

    def is_tab_running(self, tab_id: str) -> bool:
        _ = self.tabs[tab_id]
        return self.running_tab_id == tab_id

    def is_tab_analyzing(self, tab_id: str) -> bool:
        return self.tabs[tab_id].is_analyzing

    def is_tab_busy(self, tab_id: str) -> bool:
        tab = self.tabs[tab_id]
        return self.running_tab_id == tab_id or tab.is_analyzing
