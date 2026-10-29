"""Tab render projections and saved artifact evidence for independent panes."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.artifacts import ArtifactRef
from zcu_tools.gui.app.measure.artifacts import SaveStatus
from zcu_tools.gui.app.measure.services.pane_commit import (
    commit_analysis,
    commit_post_analysis,
)
from zcu_tools.gui.app.measure.services.tab import TabService
from zcu_tools.gui.app.measure.state import (
    AnalysisPaneState,
    PostAnalysisPaneState,
)
from zcu_tools.gui.app.measure.writeback import ParameterWriteback, WritebackService
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.entry import ArtifactKey

from tests.gui.app.measure._form_fakes import form_resources
from tests.gui.app.measure._native_save_support import (
    NativeSaveSession,
    PaneSummary,
    PostSummary,
    native_save_session,
)
from tests.gui.app.measure._pane_adapter import (
    PaneParams,
)


@pytest.fixture
def native_session(tmp_path, qapp):
    with native_save_session(tmp_path, qapp) as session:
        yield session


def test_snapshot_exposes_independent_panes_and_paths(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    state, tab_id = session.state, "tab"
    run_result = state.get_tab(tab_id).run.result
    primary, post = PaneSummary(), PostSummary()
    primary_plots = Plots(NonPresentingHost())
    primary_plots.subplots("fit")
    primary_plots.finish()
    post_plots = Plots(NonPresentingHost())
    post_plots.subplots("fit")
    post_plots.finish()
    writeback = WritebackService(session.context, arb_names=lambda: ())
    primary_draft, post_draft = (
        writeback.create_draft(
            [ParameterWriteback("general.ext.a", description, proposed_value=value)],
            parameters=session.context.read_context(),
            experiment_name="synthetic",
        )
        for description, value in (("primary", 1), ("post", 2))
    )
    commit_analysis(
        state,
        tab_id,
        AnalysisPaneState(
            result=primary,
            plots=primary_plots,
            params=PaneParams(initial_state="excited"),
            writeback_draft=primary_draft,
        ),
    )
    commit_post_analysis(
        state,
        tab_id,
        PostAnalysisPaneState(
            result=post,
            plots=post_plots,
            params=PaneParams(initial_state="ground"),
            writeback_draft=post_draft,
        ),
    )
    tabs = TabService(
        state, MagicMock(), form_resources(state), MagicMock(), writeback=writeback
    )
    snapshot = tabs.get_snapshot(tab_id)
    assert snapshot.run is not None and snapshot.run.result is run_result
    assert snapshot.analysis is not None and snapshot.analysis.result is primary
    assert snapshot.post_analysis is not None and snapshot.post_analysis.result is post
    assert [item.description for item in snapshot.analysis.writeback_items] == [
        "primary"
    ]
    assert [item.description for item in snapshot.post_analysis.writeback_items] == [
        "post"
    ]
    assert [
        artifact.ref for section in snapshot.artifacts for artifact in section.artifacts
    ] == [
        ArtifactRef("run", "measurement_data"),
        ArtifactRef("analysis", "results"),
        ArtifactRef("post_analysis", "results"),
    ]
    assert all(
        artifact.status is SaveStatus.NOT_SAVED
        for section in snapshot.artifacts
        for artifact in section.artifacts
    )
    report = session.save_artifacts()
    saved = tabs.get_snapshot(tab_id)
    assert {
        member.key: tuple(
            (output.path, output.status, output.error) for output in member.outputs
        )
        for section in saved.artifacts
        for artifact in section.artifacts
        for member in artifact.members
    } == {
        key: tuple(
            (item.output.path, item.status, item.error)
            for item in report.outputs
            if item.key == key
        )
        for key in {item.key for item in report.outputs}
    }
    assert all(
        artifact.status is SaveStatus.SAVED
        for section in saved.artifacts
        for artifact in section.artifacts
    )
    primary_plots.release()
    post_plots.release()


def test_named_panes_project_paths_and_retire_original_collections(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    state, tab_id = session.state, "tab"
    original = Plots(NonPresentingHost())
    old_figure, _ = original.subplots("old")
    original.finish()
    commit_analysis(
        state, tab_id, AnalysisPaneState(result=PaneSummary(), plots=original)
    )
    replacement = Plots(NonPresentingHost())
    first, _ = replacement.subplots("fit")
    second, _ = replacement.subplots("diagnostic")
    replacement.finish()
    commit_analysis(
        state, tab_id, AnalysisPaneState(result=PaneSummary(), plots=replacement)
    )
    png = BytesIO()
    old_figure.savefig(png, format="png")
    assert png.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    post = Plots(NonPresentingHost())
    post_figure, _ = post.subplots("fit")
    post.finish()
    commit_post_analysis(
        state, tab_id, PostAnalysisPaneState(result=PostSummary(), plots=post)
    )
    tabs = TabService(
        state, MagicMock(), form_resources(state), MagicMock(), writeback=MagicMock()
    )
    snapshot = tabs.get_snapshot(tab_id)
    assert (
        snapshot.analysis is not None
        and snapshot.analysis.figures is not None
        and snapshot.analysis.figures is replacement
    )
    assert (
        snapshot.post_analysis is not None
        and snapshot.post_analysis.figures is not None
        and snapshot.post_analysis.figures is post
    )
    assert snapshot.analysis.figures["fit"] is first
    assert snapshot.analysis.figures["diagnostic"] is second
    assert snapshot.post_analysis.figures["fit"] is post_figure
    assert [
        member.key
        for section in snapshot.artifacts
        for artifact in section.artifacts
        for member in artifact.members
    ] == [
        ArtifactKey("run", "measurement_data", "data"),
        ArtifactKey("analysis", "results", "figure", "fit"),
        ArtifactKey("analysis", "results", "figure", "diagnostic"),
        ArtifactKey("analysis", "results", "analysis", "analysis"),
        ArtifactKey("post_analysis", "results", "figure", "fit"),
        ArtifactKey("post_analysis", "results", "analysis", "post_analysis"),
    ]
    session.save_artifacts()
    assert all(
        artifact.status is SaveStatus.SAVED
        for section in tabs.get_snapshot(tab_id).artifacts
        for artifact in section.artifacts
    )
    newer = Plots(NonPresentingHost())
    newer.subplots("fit")
    newer.finish()
    commit_analysis(state, tab_id, AnalysisPaneState(result=PaneSummary(), plots=newer))
    assert state.get_tab(tab_id).post_analysis.plots is None
    current = tabs.get_snapshot(tab_id)
    assert [
        artifact.ref for section in current.artifacts for artifact in section.artifacts
    ] == [
        ArtifactRef("run", "measurement_data"),
        ArtifactRef("analysis", "results"),
    ]
    analysis = current.artifacts[1].artifacts[0]
    assert analysis.status is SaveStatus.NOT_SAVED
    assert [member.key for member in analysis.members] == [
        ArtifactKey("analysis", "results", "figure", "fit"),
        ArtifactKey("analysis", "results", "analysis", "analysis"),
    ]
    assert all(
        output.status == "reserved"
        for member in analysis.members
        for output in member.outputs
    )
    replacement.release()
    post.release()
    newer.release()
