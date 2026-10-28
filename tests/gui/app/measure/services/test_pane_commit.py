"""Pane publication owns retirement; use real State, drafts and plot producers."""

from dataclasses import replace

import pytest
from matplotlib.figure import Figure
from zcu_tools.gui.app.measure.services import pane_commit
from zcu_tools.gui.app.measure.state import (
    AnalysisPaneState,
    PostAnalysisPaneState,
    RunPaneState,
)
from zcu_tools.gui.app.measure.writeback import WritebackService
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.entry import ArtifactKey

from tests.gui.app.measure._native_save_support import native_save_session


class RecordingHost(NonPresentingHost):
    def __init__(self, *, fail_release: bool = False) -> None:
        self.released: list[Figure] = []
        self.fail_release = fail_release

    def release(self, figure: Figure) -> None:
        self.released.append(figure)
        if self.fail_release:
            raise RuntimeError("host release failed")


@pytest.fixture
def session(tmp_path, qapp):
    with native_save_session(tmp_path, qapp) as value:
        yield value


def test_run_commit_retires_dependents_and_replans_native_artifacts(session):
    state = session.state
    tab = state.get_tab("tab")
    writeback = WritebackService(session.context, arb_names=lambda: ())
    primary = writeback.create_draft([], parameters=None, experiment_name="saving")
    post = writeback.create_draft([], parameters=None, experiment_name="saving")
    host = RecordingHost()
    plots = Plots(host)
    figure, _ = plots.subplots("fit")
    plots.finish()
    tab.analysis.writeback_draft = primary
    tab.analysis.plots = plots
    tab.post_analysis.writeback_draft = post
    versions = {
        key: state.version.get(f"tab:tab:{key}")
        for key in ("result", "analyze", "post_analyze")
    }
    record = replace(
        tab.run.result,
        metadata=session.metadata.model_copy(update={"run_id": "replacement"}),
    )
    pane = RunPaneState(
        result=record, parameters=tab.run.parameters, storage=tab.run.storage
    )

    pane_commit.commit_run(state, "tab", pane)

    assert tab.run is pane
    assert tab.analysis.result is None and tab.post_analysis.result is None
    assert not primary.is_active and not post.is_active
    assert host.released == [figure]
    assert {key: state.version.get(f"tab:tab:{key}") for key in versions} == {
        key: value + 1 for key, value in versions.items()
    }
    sections = state.get_artifact_snapshots("tab")
    assert len(sections) == 1
    member = sections[0].artifacts[0].members[0]
    assert member.key == ArtifactKey("run", "measurement_data", "data")
    assert (
        member.outputs[0].path
        == session.entry.database_path / "runs" / "saving_replacement" / "data.h5"
    )


def test_analysis_commit_retires_both_stages_and_clears_post_artifacts(session):
    state = session.state
    tab = state.get_tab("tab")
    writeback = WritebackService(session.context, arb_names=lambda: ())
    primary = writeback.create_draft([], parameters=None, experiment_name="saving")
    post = writeback.create_draft([], parameters=None, experiment_name="saving")
    host = RecordingHost()
    plots = Plots(host)
    figure, _ = plots.subplots("old")
    plots.finish()
    tab.analysis = AnalysisPaneState(result="old", plots=plots, writeback_draft=primary)
    tab.post_analysis = PostAnalysisPaneState(result="post", writeback_draft=post)
    versions = {
        key: state.version.get(f"tab:tab:{key}")
        for key in ("result", "analyze", "post_analyze")
    }
    run_artifact = state.get_artifact_snapshots("tab")[0]
    params = {"gain": 2}
    pane = AnalysisPaneState(result={"mean": 3}, params=params)

    pane_commit.commit_analysis(state, "tab", pane)

    assert tab.analysis is pane
    assert tab.analysis.figures_replaced
    assert tab.analysis.result_params == {"gain": 2}
    params["gain"] = 9
    assert tab.analysis.result_params == {"gain": 2}
    assert not primary.is_active and not post.is_active
    assert host.released == [figure]
    assert not tab.has_post_analyze_result()
    assert state.version.get("tab:tab:result") == versions["result"]
    assert state.version.get("tab:tab:analyze") == versions["analyze"] + 1
    assert state.version.get("tab:tab:post_analyze") == versions["post_analyze"] + 1
    sections = state.get_artifact_snapshots("tab")
    assert sections[0] == run_artifact
    assert [section.section for section in sections] == ["run", "analysis"]
    member = sections[1].artifacts[0].members[0]
    assert member.key == ArtifactKey("analysis", "results", "analysis", "analysis")
    assert (
        member.outputs[0].path
        == session.entry.database_path
        / "runs"
        / "saving_20261008T000000Z-a1b2c3"
        / "analysis"
        / "analysis.json"
    )


def test_post_commit_preserves_primary_and_retires_only_previous_post(session):
    state = session.state
    tab = state.get_tab("tab")
    pane_commit.commit_analysis(state, "tab", AnalysisPaneState(result="primary"))
    primary = tab.analysis
    primary_artifacts = state.get_artifact_snapshots("tab")
    writeback = WritebackService(session.context, arb_names=lambda: ())
    draft = writeback.create_draft([], parameters=None, experiment_name="saving")
    host = RecordingHost()
    old_plots = Plots(host)
    old_figure, _ = old_plots.subplots("old-post")
    old_plots.finish()
    tab.post_analysis = PostAnalysisPaneState(
        result="old-post", plots=old_plots, writeback_draft=draft
    )
    versions = state.version.snapshot()
    plots = Plots(NonPresentingHost())
    plots.subplots("correction")
    plots.finish()
    pane = PostAnalysisPaneState(result=None, plots=plots, params={"scale": 3})

    pane_commit.commit_post_analysis(state, "tab", pane)

    assert tab.post_analysis is pane
    assert pane.figures_replaced
    assert pane.result_params == {"scale": 3}
    assert tab.analysis is primary
    assert not draft.is_active
    assert host.released == [old_figure]
    assert state.version.snapshot() == {
        **versions,
        "tab:tab:post_analyze": versions["tab:tab:post_analyze"] + 1,
    }
    sections = state.get_artifact_snapshots("tab")
    assert sections[:2] == primary_artifacts
    assert [member.key for member in sections[2].artifacts[0].members] == [
        ArtifactKey("post_analysis", "results", "figure", "correction"),
        ArtifactKey("post_analysis", "results", "analysis", "post_analysis"),
    ]
    plots.release()


class FailingWriteback(WritebackService):
    def teardown_draft(self, draft):
        raise RuntimeError("draft teardown failed")


@pytest.mark.parametrize("stage", ["run", "analysis", "post_analysis"])
def test_cleanup_failure_is_logged_without_rollback_or_skipping_other_resources(
    session, caplog, stage
):
    state = session.state
    tab = state.get_tab("tab")
    failing = FailingWriteback(session.context, arb_names=lambda: ())
    healthy = WritebackService(session.context, arb_names=lambda: ())
    bad_draft = failing.create_draft([], parameters=None, experiment_name="saving")
    good_draft = healthy.create_draft([], parameters=None, experiment_name="saving")
    host = RecordingHost(fail_release=True)
    plots = Plots(host)
    figure, _ = plots.subplots("old")
    plots.finish()
    tab.analysis = AnalysisPaneState(result="primary", writeback_draft=good_draft)
    tab.post_analysis = PostAnalysisPaneState(
        result="post", plots=plots, writeback_draft=bad_draft
    )

    if stage == "run":
        pane = RunPaneState()
        pane_commit.commit_run(state, "tab", pane)
        assert tab.run is pane
    elif stage == "analysis":
        pane = AnalysisPaneState(result="replacement")
        pane_commit.commit_analysis(state, "tab", pane)
        assert tab.analysis is pane
    else:
        pane = PostAnalysisPaneState(result="replacement")
        pane_commit.commit_post_analysis(state, "tab", pane)
        assert tab.post_analysis is pane
        assert good_draft.is_active

    assert host.released == [figure]
    assert "draft teardown failed" in caplog.text
    assert "host release failed" in caplog.text
    if stage != "post_analysis":
        assert not good_draft.is_active
