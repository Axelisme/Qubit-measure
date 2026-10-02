"""Build complete State-shaped snapshots for view-only widget fixtures.

State and SaveService own the real lifecycle. These synthetic facts let widget
specimens render that read model without inventing a second widget tracker.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from matplotlib.figure import Figure
from zcu_tools.gui.app.measure.adapter import AnalysisMode
from zcu_tools.gui.app.measure.artifact_tracker import (
    ArtifactKey,
    ArtifactKind,
    ArtifactSnapshot,
    SaveStatus,
)
from zcu_tools.gui.app.measure.services import TabSnapshot
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
