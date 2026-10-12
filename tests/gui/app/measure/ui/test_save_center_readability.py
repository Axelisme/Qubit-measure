"""User-facing save evidence over real State, tracker reports and Qt widgets."""

from collections.abc import Generator
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit
from zcu_tools.experiment.run_saving import ArtifactRef, OutputResult, SaveReport
from zcu_tools.gui.app.measure.artifact_tracker import ArtifactDeclaration
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.events.completion import (
    SaveArtifactsFinishedPayload,
    SaveDataFinishedPayload,
)
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.ui.artifact_save_center import ArtifactSaveCenter
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.resources.entry import ArtifactKey, Output, ResultEntry, SaveLayout

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui.app.measure._native_save_support import (
    GUI_ORIGIN,
    AnalysisSavingAdapter,
    NativeSaveSession,
    PaneParams,
    PaneSummary,
    native_save_session,
)


@pytest.fixture
def saving_view(
    tmp_path: Path,
    qapp: QApplication,
) -> Generator[
    tuple[NativeSaveSession, Controller, MainWindow, str, ArtifactSaveCenter]
]:
    with native_save_session(tmp_path, qapp) as session:
        registry = Registry()
        registry.register("saving", AnalysisSavingAdapter)
        ctrl = Controller(
            session.state,
            registry,
            None,
            session.bus,
            progress_transport=DirectProgressTransport(),
            project_root=str(tmp_path),
        )
        tab_id = ctrl.new_tab("saving")
        original = session.state.get_tab("tab")
        session.state.update_tab_result(
            tab_id,
            original.run.result,
            metadata=session.metadata,
            parameters=original.run.parameters,
        )
        window = MainWindow(ctrl, dialog_presenter=RecordingDialogPresenter())
        window.add_tab_widget(tab_id, "saving")
        center = window.findChild(ArtifactSaveCenter)
        assert center is not None
        window.resize(1600, 900)
        window.show()
        center.show()
        qapp.processEvents()
        try:
            yield session, ctrl, window, tab_id, center
        finally:
            assert ctrl._background_svc.quiesce()
            ctrl.tab_control.close_tab(tab_id)
            window.deleteLater()
            qapp.processEvents()


def test_single_artifact_title_and_output_filenames(saving_view) -> None:
    session, ctrl, window, tab_id, center = saving_view
    session.state.update_tab_analyze(
        tab_id, PaneSummary(), None, analyze_params_instance=PaneParams()
    )
    window.refresh_tab_interaction(tab_id)
    labels = [label.text() for label in center.findChildren(QLabel)]
    assert "Measurement data" in labels
    assert "Analysis results" in labels
    outputs = [
        output
        for section in ctrl.get_tab_snapshot(tab_id).artifacts
        for artifact in section.artifacts
        for member in artifact.members
        for output in member.outputs
    ]
    assert all(output.path.name in labels for output in outputs)
    assert not any("reserved" in text for text in labels)


def test_multiple_artifact_titles_retain_names(saving_view) -> None:
    session, _, window, tab_id, center = saving_view
    storage = session.state.get_tab(tab_id).run.storage
    assert storage is not None
    layout = SaveLayout(
        result_path=storage.result_path,
        database_path=storage.database_path,
        run_id=session.metadata.run_id,
        point=storage.point,
        saved_at=datetime(2026, 10, 8, tzinfo=timezone.utc),
    )
    tracker = session.state.get_tab(tab_id).artifacts
    tracker.replace_section(
        "analysis",
        tuple(
            ArtifactDeclaration(
                ArtifactRef("analysis", name),
                (ArtifactKey("analysis", name, "analysis", name),),
            )
            for name in ("fit", "diagnostics")
        ),
        layout=layout,
    )
    window.refresh_tab_interaction(tab_id)
    labels = [label.text() for label in center.findChildren(QLabel)]
    assert "Analysis results: fit" in labels
    assert "Analysis results: diagnostics" in labels
    tracker.replace_section(
        "analysis",
        (
            ArtifactDeclaration(
                ArtifactRef("analysis", "fit"),
                (ArtifactKey("analysis", "fit", "analysis", "fit"),),
            ),
        ),
        layout=layout,
    )
    window.refresh_tab_interaction(tab_id)
    assert "Analysis results" in [label.text() for label in center.findChildren(QLabel)]


def test_paths_use_run_storage_after_switching_entry(
    saving_view, tmp_path, qapp
) -> None:
    from zcu_tools.plotting.plots import NonPresentingHost, Plots

    session, ctrl, window, tab_id, center = saving_view
    plots = Plots(NonPresentingHost())
    plots.subplots("fit")
    plots.finish(present=False)
    session.state.update_tab_analyze(
        tab_id, PaneSummary(), plots, analyze_params_instance=PaneParams()
    )
    other = ResultEntry.create(
        "different",
        result_root=tmp_path / "other_result",
        database_root=tmp_path / "other_database",
    )
    session.context.install_entry(other)
    window.refresh_tab_interaction(tab_id)
    editors = center.findChildren(QLineEdit)
    outputs = [
        output
        for section in ctrl.get_tab_snapshot(tab_id).artifacts
        for artifact in section.artifacts
        for member in artifact.members
        for output in member.outputs
    ]
    assert [edit.toolTip() for edit in editors] == [str(out.path) for out in outputs]
    # At a generous width the entire relative text is visible and copyable.
    for edit in editors:
        edit.resize(1200, 30)
    qapp.processEvents()
    assert [edit.text() for edit in editors] == [
        str(out.path.relative_to(tmp_path)) for out in outputs
    ]
    assert any(edit.text().startswith("Database/original/runs/") for edit in editors)
    assert any(
        edit.text().startswith("result/original/points/p/figures/") for edit in editors
    )
    editor = editors[0]
    editor.resize(200, 30)
    qapp.processEvents()
    assert "…" in editor.text()
    assert editor.text().startswith("Database/")
    assert editor.text().endswith("data.h5")
    assert editor.toolTip() == str(outputs[0].path)


@pytest.mark.parametrize("failed", [False, True])
def test_reports_show_inline_symbols_and_summary(saving_view, failed) -> None:
    session, ctrl, window, tab_id, center = saving_view
    snap = ctrl.get_tab_snapshot(tab_id)
    outputs = tuple(
        OutputResult(
            member.key,
            Output(out.format, out.path),
            "failed" if failed and index == 1 else "saved",
            "copy denied: " + "long failure detail " * 15
            if failed and index == 1
            else None,
        )
        for section in snap.artifacts
        for artifact in section.artifacts
        for member in artifact.members
        for index, out in enumerate(member.outputs)
    )
    report = SaveReport(session.metadata.run_id, outputs)
    session.state.commit_save_report(tab_id, report)
    session.bus.emit(
        SaveArtifactsFinishedPayload(
            tab_id=tab_id,
            report=report,
            error="output failed" if failed else None,
        )
    )
    window.refresh_tab_interaction(tab_id)
    labels = center.findChildren(QLabel)
    assert sum(label.text() == "✓" for label in labels) == (1 if failed else 2)
    if failed:
        failure = next(label for label in labels if label.text().startswith("✗"))
        assert "copy denied" in failure.text()
        assert failure.wordWrap()
    message = window.get_view_snapshot().status_message
    assert session.metadata.run_id in message
    assert ("Saved 1 of 2 files; 1 failed" if failed else "Saved 2 files") in message
    assert str(outputs[0].output.path) not in message


def test_native_save_completion_uses_summary(saving_view) -> None:
    session, ctrl, window, tab_id, _ = saving_view
    report = ctrl.save_control.save_data(tab_id, origin=GUI_ORIGIN)
    message = window.get_view_snapshot().status_message
    assert "Saved 2 files" in message
    assert report.run_id in message
    assert str(report.outputs[0].output.path) not in message
    # The same completion route is used when a remote command publishes it.
    session.bus.emit(
        SaveDataFinishedPayload(
            tab_id=tab_id,
            data_path=str(report.outputs[0].output.path),
            report=report,
        )
    )
    assert window.get_view_snapshot().status_message == message


def test_hint_tracks_declared_support_not_current_outputs(saving_view) -> None:
    session, ctrl, window, tab_id, center = saving_view
    session.state.clear_tab_results(tab_id)
    window.refresh_tab_interaction(tab_id)
    hint_text = "This experiment does not save to the result entry yet."
    assert not any(
        not label.isHidden() and label.text() == hint_text
        for label in center.findChildren(QLabel)
    )
    assert not center.save_all_button.isEnabled()
    # experiment_tag is the existing declaration, not an inference from methods.
    session.state.get_tab(tab_id).adapter = replace_adapter = AnalysisSavingAdapter()
    replace_adapter.experiment_tag = None
    window.refresh_tab_interaction(tab_id)
    hints = [
        label for label in center.findChildren(QLabel) if label.text() == hint_text
    ]
    assert len(hints) == 1
    hint = hints[0]
    assert not hint.isHidden()
    assert "color:" in hint.styleSheet()
    assert not center.save_all_button.isEnabled()
