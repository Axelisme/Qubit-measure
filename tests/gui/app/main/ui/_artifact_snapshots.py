"""Build complete State-shaped snapshots for view-only widget fixtures.

State and SaveService own the real lifecycle. These synthetic facts let widget
specimens render that read model without inventing a second widget tracker.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from zcu_tools.gui.app.main.adapter import AnalysisMode
from zcu_tools.gui.app.main.artifact_tracker import (
    ArtifactKind,
    ArtifactSnapshot,
    SaveStatus,
)
from zcu_tools.gui.app.main.services import TabSnapshot


def with_artifacts(
    snapshot: TabSnapshot,
    status_overrides: Mapping[ArtifactKind, SaveStatus] | None = None,
) -> TabSnapshot:
    caps = snapshot.capabilities
    state = snapshot.interaction
    paths = snapshot.paths
    if caps is None or state is None or paths is None:
        raise ValueError("View specimen requires capabilities, interaction and paths")

    facts = (
        (ArtifactKind.DATA, True, state.has_run_result, True, paths.data.path),
        (
            ArtifactKind.ANALYSIS,
            caps.analysis is not AnalysisMode.NONE,
            state.has_analyze_result,
            snapshot.analysis is not None and snapshot.analysis.figure is not None,
            paths.analysis_image.path,
        ),
        (
            ArtifactKind.POST_ANALYSIS,
            caps.post_analysis,
            state.has_post_analyze_result,
            snapshot.post_analysis is not None
            and snapshot.post_analysis.figure is not None,
            paths.post_analysis_image.path,
        ),
    )
    overrides = status_overrides or {}
    return replace(
        snapshot,
        artifacts=tuple(
            ArtifactSnapshot(
                kind=kind,
                status=overrides.get(
                    kind,
                    SaveStatus.NOT_SAVED if has_result else SaveStatus.NO_RESULT,
                ),
                default_path=path,
                last_saved_path=None,
                is_saveable=has_result and has_figure,
            )
            for kind, enabled, has_result, has_figure, path in facts
            if enabled
        ),
    )
