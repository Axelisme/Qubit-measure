"""User-facing save evidence over real State, tracker reports and Qt widgets."""

from collections.abc import Generator
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from shutil import rmtree

import pytest
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QTextEdit
from zcu_tools.experiment.artifacts import ArtifactRef, OutputResult, SaveReport
from zcu_tools.gui.app.measure.artifacts import ArtifactDeclaration, ArtifactSaveCenter
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.events.completion import (
    SaveArtifactsFinishedPayload,
    SaveDataFinishedPayload,
)
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.pane_commit import commit_analysis, commit_run
from zcu_tools.gui.app.measure.state import AnalysisPaneState, RunPaneState
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.resources.entry import ArtifactKey, Output, ResultEntry, SaveLayout

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._native_save_support import (
    GUI_ORIGIN,
    AnalysisSavingAdapter,
    NativeSaveSession,
    PaneParams,
    PaneSummary,
    native_save_session,
)
from tests.gui.app.measure._record_support import require_record


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
        tab_id = ctrl.tab_control.new_tab("saving")
        original = session.state.get_tab("tab")
        commit_run(
            session.state,
            tab_id,
            RunPaneState(
                result=replace(require_record(original.run), metadata=session.metadata),
                storage=None
                if original.run.parameters is None
                else original.run.parameters.storage,
                parameters=original.run.parameters,
            ),
            plan_layout=session.save_service.plan_layout,
        )
        ctrl.update_tab_analyze_param_instance(tab_id, PaneParams())
        window = MainWindow(
            ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
        )
        ctrl.add_view(window)
        window.add_tab_widget(tab_id, "saving")
        window.refresh_tab_analyze_form(tab_id)
        center = window.findChild(ArtifactSaveCenter)
        assert center is not None
        window.resize(1600, 900)
        window.show()
        center.show()
        qapp.processEvents()
        try:
            yield session, ctrl, window, tab_id, center
        finally:
            ctrl.close()
            ctrl.tab_control.close_tab(tab_id)
            window.deleteLater()
            qapp.processEvents()


@pytest.fixture
def unbound_saving_view(
    tmp_path: Path, qapp: QApplication, request: pytest.FixtureRequest
) -> Generator[tuple[ComponentDriver, bool]]:
    """Publicly load a portable native file, then expose only the visible driver.

    The partial variant obstructs the external Labber directory with a file;
    native output still succeeds. No writer or controller method is patched.
    """
    with native_save_session(tmp_path, qapp) as session:
        session.save()
        portable = tmp_path / "portable" / "data.h5"
        portable.parent.mkdir()
        portable.write_bytes(session.native_data_path.read_bytes())
        rmtree(tmp_path / "Database")
        if request.param:
            blocked = tmp_path / "Database/original/Labber"
            blocked.parent.mkdir(parents=True)
            blocked.write_text("Labber destination unavailable")
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
        tab_id = ctrl.tab_control.new_tab("saving")
        ctrl.run_analyze_control.load_tab_result(tab_id, str(portable))
        window = MainWindow(
            ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
        )
        ctrl.add_view(window)
        window.add_tab_widget(tab_id, "saving")
        window.refresh_tab_analyze_form(tab_id)
        window.resize(1600, 900)
        window.show()
        qapp.processEvents()
        try:
            yield ComponentDriver(window), bool(request.param)
        finally:
            ctrl.close()
            ctrl.tab_control.close_tab(tab_id)
            window.deleteLater()
            qapp.processEvents()


@pytest.mark.parametrize(
    "unbound_saving_view", [False, True], indirect=True, ids=["success", "partial"]
)
def test_loaded_unbound_declarations_and_actual_save_outputs(
    unbound_saving_view: tuple[ComponentDriver, bool], tmp_path: Path
) -> None:
    driver, partial = unbound_saving_view
    images = tmp_path / "unbound-save"
    driver.select_tab("Data")
    driver.screenshot(images / "01-loaded-not-saved.png")
    visible = driver.visible().widgets
    assert any(w.text == "Measurement data" for w in visible)
    assert any(w.text == "○ NOT SAVED" for w in visible)
    assert driver.observe("Save All").enabled
    assert driver.observe("Save").enabled
    assert any(w.text == "data.h5" for w in visible)
    assert any("runs/" in w.text or "runs/" in w.tooltip for w in visible)

    driver.select_tab("Analysis")
    driver.screenshot(images / "02-analysis-ready.png")
    driver.click("Analyze")
    qt_until(
        lambda: driver.observe("Analyze").enabled, label="loaded analysis finished"
    )
    driver.screenshot(images / "03-analysis-finished.png")
    driver.select_tab("Data")
    driver.screenshot(images / "04-analysis-preview.png")
    assert any(w.text == "analysis.json" for w in driver.visible().widgets)
    assert any(w.text == "○ NOT SAVED" for w in driver.visible().widgets)
    driver.select_tab("Post-Analysis")
    driver.screenshot(images / "05-post-ready.png")
    driver.click("Run Post-Analysis")
    qt_until(
        lambda: driver.observe("Run Post-Analysis").enabled,
        label="loaded post-analysis finished",
    )
    driver.screenshot(images / "06-post-finished.png")
    driver.select_tab("Data")
    driver.screenshot(images / "07-post-preview.png")
    assert any(w.text == "post_analysis.json" for w in driver.visible().widgets)
    preview_paths = {
        w.tooltip
        for w in driver.visible().widgets
        if w.kind == "QLineEdit" and w.tooltip
    }
    driver.click("Save All")
    driver.screenshot(images / "08-save-report.png")
    visible = driver.visible().widgets
    assert any(w.text == "data.h5" for w in visible)
    assert any(
        "saving_20261008T000000Z-a1b2c3/data.h5" in w.text
        or "saving_20261008T000000Z-a1b2c3/data.h5" in w.tooltip
        for w in visible
    )
    assert any(w.text == "✓" for w in visible)
    if partial:
        assert any(w.text == "◐ PARTIALLY SAVED" for w in visible)
        assert driver.observe("Save All").enabled
        assert all(w.enabled for w in visible if w.text == "Save")
        assert "Saved 3 of 4 files" in driver.status().text
        assert "1 failed" in driver.status().text
    else:
        assert "Saved 4 files" in driver.status().text
        assert any(w.text == "✓ SAVED" for w in visible)
        assert not driver.observe("Save All").enabled
        assert all(not w.enabled for w in visible if w.text == "Save")
    assert preview_paths == {
        w.tooltip for w in visible if w.kind == "QLineEdit" and w.tooltip
    }


def test_single_artifact_title_and_output_filenames(saving_view) -> None:
    session, ctrl, window, tab_id, center = saving_view
    commit_analysis(
        session.state,
        tab_id,
        AnalysisPaneState(result=PaneSummary(), plots=None, params=PaneParams()),
        plan_layout=session.save_service.plan_layout,
    )
    window.refresh_tab_interaction(tab_id)
    labels = [label.text() for label in center.findChildren(QLabel)]
    assert "Measurement data" in labels
    assert "Analysis results" in labels
    outputs = [
        output
        for section in ctrl.tab_control.get_tab_snapshot(tab_id).artifacts
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
        stem=f"saving_{session.metadata.run_id}",
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


def test_paths_keep_recorded_names_after_switching_entry(
    saving_view, tmp_path, qapp
) -> None:
    from zcu_tools.plotting.plots import NonPresentingHost, Plots

    session, ctrl, window, tab_id, center = saving_view
    plots = Plots(NonPresentingHost())
    plots.subplots("fit")
    plots.finish(present=False)
    commit_analysis(
        session.state,
        tab_id,
        AnalysisPaneState(result=PaneSummary(), plots=plots, params=PaneParams()),
        plan_layout=session.save_service.plan_layout,
    )
    other = ResultEntry.create(
        "different",
        result_root=tmp_path / "other_result",
        database_root=tmp_path / "other_database",
    )
    session.context.install_entry(other)
    window.refresh_tab_interaction(tab_id)
    # Paths are read on the visible Data pane, where widths are laid out.
    window.select_tab_pane(tab_id, "data")
    qapp.processEvents()
    editors = center.findChildren(QLineEdit)
    outputs = [
        output
        for section in ctrl.tab_control.get_tab_snapshot(tab_id).artifacts
        for artifact in section.artifacts
        for member in artifact.members
        for output in member.outputs
    ]
    assert [edit.toolTip() for edit in editors] == [str(out.path) for out in outputs]
    # At a generous width the entire relative text is visible and copyable.
    for edit in editors:
        edit.setFixedWidth(1200)
    qapp.processEvents()
    assert [edit.text() for edit in editors] == [
        str(out.path.relative_to(tmp_path)) for out in outputs
    ]
    assert any(edit.text().startswith("Database/original/runs/") for edit in editors)
    assert any(
        edit.text().startswith("result/original/points/p/figures/") for edit in editors
    )
    editor = editors[0]
    editor.setFixedWidth(200)
    qapp.processEvents()
    assert "…" in editor.text()
    assert editor.text().startswith("Database/")
    assert editor.text().endswith("data.h5")
    assert editor.toolTip() == str(outputs[0].path)


@pytest.mark.parametrize("failed", [False, True])
def test_reports_show_inline_symbols_and_summary(saving_view, failed) -> None:
    session, ctrl, window, tab_id, center = saving_view
    snap = ctrl.tab_control.get_tab_snapshot(tab_id)
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
        failure = next(
            text
            for text in center.findChildren(QTextEdit)
            if text.toPlainText().startswith("✗")
        )
        assert "copy denied" in failure.toPlainText()
        assert failure.isReadOnly()
    message = window.get_view_snapshot()["status"]
    assert session.metadata.run_id in message
    assert ("Saved 1 of 2 files; 1 failed" if failed else "Saved 2 files") in message
    assert str(outputs[0].output.path) not in message


def test_native_save_completion_uses_summary(saving_view) -> None:
    session, ctrl, window, tab_id, _ = saving_view
    report = ctrl.save_control.save_data(tab_id, origin=GUI_ORIGIN)
    message = window.get_view_snapshot()["status"]
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
    assert window.get_view_snapshot()["status"] == message


def test_save_all_disables_until_analysis_is_run_again(
    saving_view, tmp_path: Path
) -> None:
    _, _, window, _, _ = saving_view
    driver = ComponentDriver(window)
    images = tmp_path / "save-availability"
    driver.select_tab("Data")
    driver.screenshot(images / "01-unsaved-data.png")
    assert driver.observe("Save All").enabled
    assert driver.observe("Save").enabled

    driver.click("Save All")
    driver.screenshot(images / "02-all-saved.png")
    assert "Saved 2 files" in driver.status().text
    assert not driver.observe("Save All").enabled
    assert not driver.observe("Save").enabled

    driver.select_tab("Analysis")
    driver.screenshot(images / "03-analysis.png")
    driver.click("Analyze")
    qt_until(lambda: driver.observe("Analyze").enabled, label="analysis finished")
    driver.screenshot(images / "04-analysis-finished.png")
    driver.select_tab("Data")
    qt_until(
        lambda: any(w.text == "Analysis results" for w in driver.visible().widgets),
        label="analysis artifact displayed",
    )
    driver.screenshot(images / "05-unsaved-analysis.png")
    assert driver.observe("Save All").enabled
    saves = [w for w in driver.visible().widgets if w.text == "Save"]
    assert len(saves) == 2 and all(w.enabled for w in saves)

    driver.click("Save All")
    driver.screenshot(images / "06-analysis-saved.png")
    assert not driver.observe("Save All").enabled
    saves = [w for w in driver.visible().widgets if w.text == "Save"]
    assert len(saves) == 2 and all(not w.enabled for w in saves)

    driver.select_tab("Analysis")
    driver.screenshot(images / "07-analysis-again.png")
    driver.click("Analyze")
    qt_until(lambda: driver.observe("Analyze").enabled, label="reanalysis finished")
    driver.screenshot(images / "08-reanalysis-finished.png")
    driver.select_tab("Data")
    driver.screenshot(images / "09-reanalysis-unsaved.png")
    assert driver.observe("Save All").enabled
    assert any(w.text == "○ NOT SAVED" for w in driver.visible().widgets)
    saves = [w for w in driver.visible().widgets if w.text == "Save"]
    assert len(saves) == 2 and all(w.enabled for w in saves)
