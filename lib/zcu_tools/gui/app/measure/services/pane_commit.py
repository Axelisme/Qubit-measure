"""Publish pane replacements and retire their owned presentation and drafts.

Call on State's owner thread with a complete pane whose producer has finished.
Planning errors propagate before publication. Retirement errors are logged and
isolated after publication, never rolling back the new pane or its versions.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any

from zcu_tools.experiment.artifacts import ArtifactRef
from zcu_tools.resources.entry import ArtifactKey, SaveContext, SaveLayout

from ..artifacts import ArtifactDeclaration, ArtifactTracker
from ..state import (
    AnalysisPaneState,
    PostAnalysisPaneState,
    RetiredPaneResources,
    RunPaneState,
    Session,
    State,
)

if TYPE_CHECKING:
    from zcu_tools.plotting.plots import Plots

logger = logging.getLogger(__name__)


def discard_unpublished_plots(plots: Plots) -> None:
    """Stop an unsuccessful producer and detach even if its final draw fails.

    Call when the operation discards its unpublished presentation. Finish without
    presenting, then always release attachments. Cleanup errors propagate to the
    operation's failure-isolation policy; detached Figures remain caller-owned.
    """
    try:
        plots.finish(present=False)
    finally:
        plots.release()


def retire_resources(retired: RetiredPaneResources) -> None:
    """Close detached drafts and release plots, logging each failure separately.

    Use only after publication or tab removal. Continue through all resources;
    this command does not restore State or retry failed retirement.
    """
    for draft in retired.writeback_drafts:
        try:
            draft.teardown()
        except Exception:
            logger.exception("Retired pane draft teardown failed")
    for plots in retired.plots:
        try:
            plots.release()
        except Exception:
            logger.exception("Retired pane plot release failed")


def commit_run(state: State, tab_id: str, pane: RunPaneState) -> None:
    """Replace Run and invalidate Analysis/Post, then retire their resources.

    pane fixes the result, acquisition evidence and storage destination; None
    result clears Run. Unknown tabs raise KeyError; off-owner calls raise
    RuntimeError before mutation. Planning and cleanup follow the module contract.
    """
    # This read validates owner-thread access before adapter/path planning.
    state.get_artifact_snapshots(tab_id)
    tab = state.get_tab(tab_id)
    layout = _save_layout(pane, tab=tab)
    artifacts = ArtifactTracker()
    if pane.result is not None and layout is not None:
        artifacts.replace_section(
            "run",
            (
                ArtifactDeclaration(
                    ArtifactRef("run", "measurement_data"),
                    (ArtifactKey("run", "measurement_data", "data"),),
                ),
            ),
            layout=layout,
        )
    retired = state.set_run_pane(tab_id, pane, artifacts)
    state.version.bump(f"tab:{tab_id}:result")
    state.version.bump(f"tab:{tab_id}:analyze")
    state.version.bump(f"tab:{tab_id}:post_analyze")
    retire_resources(retired)


def commit_analysis(
    state: State, tab_id: str, pane: AnalysisPaneState[Any, Any]
) -> None:
    """Replace primary Analysis, invalidate Post, and retire both old stages.

    pane contains the completed result/plots, editable params and optional draft.
    Snapshot params into result_params so later edits cannot change saved inputs.
    Unknown tabs and off-owner calls fail as for commit_run. Cleanup never rolls
    back publication; planning errors leave the previous panes intact.
    """
    state.get_artifact_snapshots(tab_id)
    tab = state.get_tab(tab_id)
    artifacts = deepcopy(tab.artifacts)
    layout = _save_layout(tab.run, tab=tab)
    if layout is not None:
        artifacts.replace_section(
            "analysis", _pane_declarations("analysis", pane), layout=layout
        )
        artifacts.replace_section("post_analysis", (), layout=layout)
    pane.result_params = deepcopy(pane.params)
    pane.figures_replaced = bool(tab.analysis.plots)
    retired = state.set_analysis_pane(tab_id, pane, artifacts)
    state.version.bump(f"tab:{tab_id}:analyze")
    state.version.bump(f"tab:{tab_id}:post_analyze")
    retire_resources(retired)


def commit_post_analysis(
    state: State, tab_id: str, pane: PostAnalysisPaneState[Any, Any]
) -> None:
    """Replace Post without altering primary Analysis and retire the old Post.

    pane contains a completed Post result/plots, params and optional draft.
    Snapshot params into result_params. A missing primary raises RuntimeError
    before publication. Unknown tabs, owner-thread and failure rules match
    commit_analysis, including figures-only (None result) publications.
    """
    state.get_artifact_snapshots(tab_id)
    tab = state.get_tab(tab_id)
    if not tab.has_analyze_result():
        raise RuntimeError(
            f"Cannot record post-analysis for tab {tab_id!r}: no primary analyze result"
        )
    artifacts = deepcopy(tab.artifacts)
    layout = _save_layout(tab.run, tab=tab)
    if layout is not None:
        artifacts.replace_section(
            "post_analysis", _pane_declarations("post_analysis", pane), layout=layout
        )
    pane.result_params = deepcopy(pane.params)
    pane.figures_replaced = bool(tab.post_analysis.plots)
    retired = state.set_post_analysis_pane(tab_id, pane, artifacts)
    state.version.bump(f"tab:{tab_id}:post_analyze")
    retire_resources(retired)


def _save_layout(
    run: RunPaneState, *, tab: Session[Any, Any, Any, Any]
) -> SaveLayout | None:
    """Plan against this result's source, never tab birth or live selection."""
    metadata = None if run.result is None else run.result.metadata
    storage = run.storage
    if metadata is None or storage is None:
        return None
    saved_at = datetime.now().astimezone()
    record = run.result
    stem = tab.adapter.save_stem(
        SaveContext(
            name=tab.adapter_name,
            run_id=metadata.run_id,
            point=metadata.snapshot.point,
            roles=metadata.snapshot.roles,
            cfg=(
                None
                if record is None or record.cfg is None
                else record.cfg.model_dump(mode="json")
            ),
            saved_at=saved_at,
        )
    )
    return SaveLayout(
        result_path=storage.result_path,
        database_path=storage.database_path,
        stem=stem,
        point=metadata.snapshot.point,
        saved_at=saved_at,
    )


def _pane_declarations(
    section: str,
    pane: AnalysisPaneState[Any, Any] | PostAnalysisPaneState[Any, Any],
) -> tuple[ArtifactDeclaration, ...]:
    """Declare successful pane figures and JSON, including JSON-only results."""
    if pane.result is None and pane.plots is None:
        return ()
    ref = ArtifactRef(section, "results")
    figures = (
        ()
        if pane.plots is None
        else tuple(
            ArtifactKey(section, ref.name, "figure", name) for name in pane.plots
        )
    )
    return (
        ArtifactDeclaration(
            ref,
            figures + (ArtifactKey(section, ref.name, "analysis", section),),
        ),
    )
