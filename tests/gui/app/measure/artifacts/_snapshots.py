"""Build declared artifact facts for view-only widget specimens.

State and SaveService own the real lifecycle. These specimens reuse the public
tracker and planner without serializing payloads or writing files.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from matplotlib.figure import Figure
from zcu_tools.experiment.artifacts import ArtifactRef
from zcu_tools.gui.app.measure.adapter import AnalysisMode
from zcu_tools.gui.app.measure.artifacts import (
    ArtifactDeclaration,
    ArtifactTracker,
)
from zcu_tools.gui.app.measure.services import TabSnapshot
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.entry import ArtifactKey, SaveLayout


def ready_figures(figure: Figure | None) -> Plots | None:
    """Wrap a named fit figure in a finished, non-presenting collection, or None."""
    if figure is None:
        return None
    plots = Plots(NonPresentingHost())
    plots.adopt("fit", figure)
    plots.finish()
    return plots


def with_artifacts(snapshot: TabSnapshot) -> TabSnapshot:
    """Replace specimen artifact facts with current Not saved declarations.

    Requires capabilities, raising ValueError if absent. A Run result declares
    measurement data. Each supported successful analysis pane declares its JSON
    and named figures, including JSON-only results. Missing results declare
    nothing. Paths come from SaveLayout under the worktree's .pytest-tmp tree;
    planning may inspect existing paths but never creates directories or files.
    The input snapshot and its pane resources remain unchanged.
    """
    caps = snapshot.capabilities
    if caps is None:
        raise ValueError("View specimen requires capabilities")
    layout = SaveLayout(
        result_path=Path(".pytest-tmp/view-artifacts/Result"),
        database_path=Path(".pytest-tmp/view-artifacts/Database"),
        run_id="view-specimen",
        point="p",
        saved_at=datetime(2026, 10, 8, tzinfo=timezone.utc),
    )
    tracker = ArtifactTracker()
    if snapshot.run is not None and snapshot.run.result is not None:
        ref = ArtifactRef("run", "measurement_data")
        tracker.replace_section(
            ref.section,
            (ArtifactDeclaration(ref, (ArtifactKey(ref.section, ref.name, "data"),)),),
            layout=layout,
        )
    for section, enabled, pane in (
        ("analysis", caps.analysis is not AnalysisMode.NONE, snapshot.analysis),
        ("post_analysis", caps.post_analysis, snapshot.post_analysis),
    ):
        if not enabled or pane is None or pane.result is None:
            continue
        ref = ArtifactRef(section, "results")
        figures = (
            tuple(
                ArtifactKey(section, ref.name, "figure", name) for name in pane.figures
            )
            if pane.figures is not None
            else ()
        )
        tracker.replace_section(
            section,
            (
                ArtifactDeclaration(
                    ref,
                    figures + (ArtifactKey(section, ref.name, "analysis", section),),
                ),
            ),
            layout=layout,
        )
    return replace(snapshot, artifacts=tracker.snapshot())
