"""Build complete State-shaped snapshots for view-only widget fixtures.

State and SaveService own the real lifecycle. These synthetic facts let widget
specimens render that read model without inventing a second widget tracker.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from unittest.mock import MagicMock

from matplotlib.figure import Figure
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.measure.artifact_tracker import (
    ArtifactKey,
    ArtifactKind,
    ArtifactSnapshot,
    SaveStatus,
)
from zcu_tools.gui.app.measure.services import TabSnapshot
from zcu_tools.gui.app.measure.state import TabInteractionState
from zcu_tools.plotting.plots import NonPresentingHost, Plots


def ready_figures(figure: Figure | None) -> Plots | None:
    if figure is None:
        return None
    plots = Plots(NonPresentingHost())
    plots.adopt("fit", figure)
    plots.finish()
    return plots


def with_artifacts(
    snapshot: TabSnapshot,
    status_overrides: Mapping[ArtifactKind, SaveStatus] | None = None,
) -> TabSnapshot:
    caps = snapshot.capabilities
    state = snapshot.interaction
    paths = snapshot.paths
    if caps is None or state is None or paths is None:
        raise ValueError("View specimen requires capabilities, interaction and paths")

    facts = [(ArtifactKey(ArtifactKind.DATA), state.has_run_result, paths.data.path)]
    for kind, enabled, pane, image_paths in (
        (
            ArtifactKind.ANALYSIS,
            caps.analysis is not AnalysisMode.NONE,
            snapshot.analysis,
            paths.analysis_images,
        ),
        (
            ArtifactKind.POST_ANALYSIS,
            caps.post_analysis,
            snapshot.post_analysis,
            paths.post_analysis_images,
        ),
    ):
        if enabled and pane is not None and pane.result is not None and pane.figures:
            for name in pane.figures:
                facts.append((ArtifactKey(kind, name), True, image_paths[name].path))
    overrides = status_overrides or {}
    return replace(
        snapshot,
        artifacts=tuple(
            ArtifactSnapshot(
                key=key,
                status=overrides.get(
                    key.kind,
                    SaveStatus.NOT_SAVED if has_result else SaveStatus.NO_RESULT,
                ),
                default_path=path,
                last_saved_path=None,
                is_saveable=has_result,
            )
            for key, has_result, path in facts
        ),
    )


@dataclass
class _DummyParams:
    x: int = 1


def make_tab_snapshot(
    tab_id: str,
    *,
    has_run: bool = False,
    has_analysis: bool = False,
    has_post: bool = False,
    analysis_mode: AnalysisMode = AnalysisMode.FIT,
    post_cap: bool = False,
    load_cap: bool = False,
    has_active_context: bool = True,
    has_context: bool = True,
    is_running: bool = False,
    is_analyzing: bool = False,
    is_saving: bool = False,
    data_path: str | None = None,
    analysis_path: str | None = None,
    post_path: str | None = None,
    analysis_has_figure: bool | None = None,
    post_has_figure: bool | None = None,
    data_status: SaveStatus | None = None,
    analysis_status: SaveStatus | None = None,
    post_status: SaveStatus | None = None,
) -> TabSnapshot:
    """Create a view specimen with requested pane presence, paths and save status.

    Presence flags create opaque fake results and named fit figures. Optional
    paths override the specimen defaults; unspecified statuses derive from
    result presence. This builder does not run State or save operations.
    """
    from zcu_tools.gui.app.measure.services.ports import (
        AnalysisPaneSnapshot,
        PathResourceSnapshot,
        PostAnalysisPaneSnapshot,
        RunPaneSnapshot,
        SavePaneSnapshot,
        TabPathsSnapshot,
    )

    caps = AdapterCapabilities(
        analysis=analysis_mode, post_analysis=post_cap, load_data=load_cap
    )
    run_result = object() if has_run else None
    ana_result = object() if has_analysis else None
    post_result = object() if has_post else None
    if analysis_has_figure is None:
        fig = Figure() if has_analysis else None
    elif analysis_has_figure:
        fig = Figure()
    else:
        fig = None
    if post_has_figure is None:
        post_fig = Figure() if has_post else None
    elif post_has_figure:
        post_fig = Figure()
    else:
        post_fig = None

    data_ps = PathResourceSnapshot(
        override=data_path, path=data_path or ("/tmp/data.h5" if has_run else None)
    )
    ana_ps = PathResourceSnapshot(
        override=analysis_path,
        path=analysis_path or ("/tmp/a.png" if has_analysis else None),
    )
    post_ps = PathResourceSnapshot(
        override=post_path, path=post_path or ("/tmp/p.png" if has_post else None)
    )

    analysis_plots = None
    if fig is not None:
        analysis_plots = Plots(NonPresentingHost())
        analysis_plots.adopt("fit", fig)
        analysis_plots.finish()
    post_plots = None
    if post_fig is not None:
        post_plots = Plots(NonPresentingHost())
        post_plots.adopt("fit", post_fig)
        post_plots.finish()

    snapshot = TabSnapshot(
        adapter_name="fake",
        cfg_schema=MagicMock(),
        tab_id=tab_id,
        interaction=TabInteractionState(
            global_run_active=False,
            is_running=is_running,
            is_analyzing=is_analyzing,
            is_saving_data=is_saving,
            has_context=has_context,
            has_active_context=has_active_context,
            has_soc=True,
            has_run_result=has_run,
            has_analyze_result=has_analysis,
            has_figure=bool(fig is not None),
            has_post_analyze_result=has_post,
        ),
        capabilities=caps,
        run=RunPaneSnapshot(result=run_result, source_path=None),
        analysis=AnalysisPaneSnapshot(
            params=_DummyParams() if has_analysis else None,
            result=ana_result,
            figures=analysis_plots,
            writeback_items=(),
            image_paths={"fit": ana_ps} if fig is not None else {},
        ),
        post_analysis=PostAnalysisPaneSnapshot(
            params=_DummyParams() if has_post else None,
            result=post_result,
            figures=post_plots,
            writeback_items=(),
            image_paths={"fit": post_ps} if post_fig is not None else {},
        ),
        save=SavePaneSnapshot(data_path=data_ps),
        paths=TabPathsSnapshot(
            data=data_ps,
            analysis_images={"fit": ana_ps} if fig is not None else {},
            post_analysis_images={"fit": post_ps} if post_fig is not None else {},
        ),
    )
    statuses = {
        kind: status
        for kind, status in (
            (ArtifactKind.DATA, data_status),
            (ArtifactKind.ANALYSIS, analysis_status),
            (ArtifactKind.POST_ANALYSIS, post_status),
        )
        if status is not None
    }
    return with_artifacts(snapshot, statuses)
