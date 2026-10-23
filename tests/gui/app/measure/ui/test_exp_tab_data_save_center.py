"""Data-center projection through public State, Controller and Qt seams."""

from dataclasses import replace
from pathlib import Path

import pytest
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import QApplication, QLineEdit, QTextEdit
from zcu_tools.experiment.artifacts import ArtifactRef, OutputResult, SaveReport
from zcu_tools.gui.app.measure.artifacts import ArtifactSaveCenter, SaveStatus
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.events.completion import SaveDataFinishedPayload
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.resources.entry import Output

from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._native_save_support import (
    GUI_ORIGIN,
    NativeSaveSession,
    PaneParams,
    PaneSummary,
)

_RUN = ArtifactRef("run", "measurement_data")


def test_named_selection_drives_screenshot_and_survives_same_collection_refresh(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
    tmp_path: Path,
) -> None:
    from zcu_tools.plotting.plots import NonPresentingHost, Plots

    session, ctrl, window, tab_id = live_ui
    plots = Plots(NonPresentingHost())
    for name, ys in (("fit", [0, 1]), ("diagnostic", [1, 0])):
        _, axes = plots.subplots(name)
        axes.plot([0, 1], ys)
    plots.finish(present=False)
    session.state.update_tab_analyze(
        tab_id, PaneSummary(), plots, analyze_params_instance=PaneParams()
    )
    snap = ctrl.get_tab_snapshot(tab_id)
    window.refresh_tab_figure(tab_id, snap)
    window.refresh_tab_interaction(tab_id, snap)
    window.resize(1280, 800)
    window.show()
    QApplication.processEvents()
    flow = ComponentDriver(window)
    flow.select_tab("Analysis")
    flow.screenshot(tmp_path / "fit.png")
    fit_png = window.take_figure_screenshot_for_subtab(tab_id, "analysis")
    flow.click(object_name="figures.tile.analysis.1")
    diagnostic_png = window.take_figure_screenshot_for_subtab(tab_id, "analysis")
    assert fit_png.startswith(b"\x89PNG") and diagnostic_png.startswith(b"\x89PNG")
    assert fit_png != diagnostic_png
    flow.screenshot(tmp_path / "diagnostic.png")
    window.refresh_tab_figure(tab_id, snap)
    assert (
        window.take_figure_screenshot_for_subtab(tab_id, "analysis") == diagnostic_png
    )
    analysis = next(
        section for section in snap.artifacts if section.section == "analysis"
    )
    assert [artifact.ref for artifact in analysis.artifacts] == [
        ArtifactRef("analysis", "results")
    ]
    assert [
        member.key.member_name
        for member in analysis.artifacts[0].members
        if member.key.member == "figure"
    ] == ["fit", "diagnostic"]

    replacement = Plots(NonPresentingHost())
    replacement.subplots("new-fit")
    replacement.finish(present=False)
    session.state.update_tab_analyze(
        tab_id, PaneSummary(), replacement, analyze_params_instance=PaneParams()
    )
    window.refresh_tab_figure(tab_id, ctrl.get_tab_snapshot(tab_id))
    assert window.take_figure_screenshot_for_subtab(tab_id, "analysis").startswith(
        b"\x89PNG"
    )
    session.state.update_tab_analyze(tab_id, None, None)
    cleared = ctrl.get_tab_snapshot(tab_id)
    window.refresh_tab_figure(tab_id, cleared)
    window.refresh_tab_interaction(tab_id, cleared)
    assert [section.section for section in cleared.artifacts] == ["run"]
    with pytest.raises(FailedPreconditionError, match="no selected figure"):
        window.take_figure_screenshot_for_subtab(tab_id, "analysis")


@pytest.mark.parametrize(
    ("context", "running", "analyzing", "enabled"),
    [
        (True, False, False, True),
        (False, False, False, False),
        (True, True, False, False),
        (True, False, True, False),
    ],
)
def test_load_data_gates(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
    context: bool,
    running: bool,
    analyzing: bool,
    enabled: bool,
) -> None:
    _, ctrl, window, tab_id = live_ui
    snap = ctrl.get_tab_snapshot(tab_id)
    assert snap.interaction is not None
    center = window.findChild(ArtifactSaveCenter)
    assert center is not None
    for active in (True, False):
        center.update_interaction(
            replace(
                snap,
                interaction=replace(
                    snap.interaction,
                    has_context=context,
                    has_active_context=active,
                    is_running=running,
                    is_analyzing=analyzing,
                ),
            )
        )
        assert center.load_button.isEnabled() is enabled
    assert snap.capabilities is not None
    unsupported = ArtifactSaveCenter(replace(snap.capabilities, load_data=False))
    try:
        unsupported.update_interaction(snap)
        assert unsupported.load_button.isHidden()
    finally:
        unsupported.deleteLater()


def test_remote_save_completion_refreshes_state_owned_status(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
) -> None:
    session, ctrl, window, tab_id = live_ui
    tab = window.findChild(ExpTabWidget)
    center = window.findChild(ArtifactSaveCenter)
    assert tab is not None and center is not None
    assert tab.has_unsaved_data()
    before = ctrl.get_tab_snapshot(tab_id)
    report = ctrl.save_control.save_data(tab_id, origin=GUI_ORIGIN)
    after = ctrl.get_tab_snapshot(tab_id)
    assert after.artifacts[0].artifacts[0].status is SaveStatus.SAVED
    # Restore the view alone: the completion must read already-committed State.
    center.update_interaction(before)
    assert tab.has_unsaved_data()
    data = next(
        output for output in report.outputs if output.output.format == "data_h5"
    )
    session.bus.emit(
        SaveDataFinishedPayload(
            tab_id=tab_id, data_path=str(data.output.path), report=report
        )
    )
    assert not tab.has_unsaved_data()
    paths = [edit.toolTip() for edit in center.findChildren(QLineEdit)]
    assert paths == [str(output.output.path) for output in report.outputs]


def test_live_comment_typing_preserves_undo_through_state_projection(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
    qapp: QApplication,
) -> None:
    session, ctrl, window, tab_id = live_ui
    ctrl.save_control.set_comment(tab_id, "original")
    report = ctrl.save_control.save_data(tab_id, origin=GUI_ORIGIN)
    window.refresh_tab_interaction(tab_id)
    center = window.findChild(ArtifactSaveCenter)
    assert center is not None
    editor = center.findChild(QTextEdit)
    assert editor is not None
    center.set_comment_text("programmatic")
    assert session.state.get_tab(tab_id).save.comment == "original"
    window.refresh_tab_interaction(tab_id)
    assert editor.toPlainText() == "original"
    editor.setFocus()
    cursor = editor.textCursor()
    cursor.setPosition(4)
    editor.setTextCursor(cursor)
    for event_type in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
        qapp.sendEvent(
            editor,
            QKeyEvent(
                event_type, Qt.Key.Key_Exclam, Qt.KeyboardModifier.NoModifier, "!"
            ),
        )
    qapp.processEvents()
    assert editor.toPlainText() == "orig!inal"
    assert session.state.get_tab(tab_id).save.comment == "orig!inal"
    dirty = ctrl.get_tab_snapshot(tab_id).artifacts[0].artifacts[0]
    assert dirty.status is SaveStatus.UNSAVED_CHANGES
    assert editor.textCursor().position() == 5
    document = editor.document()
    assert document is not None and document.isUndoAvailable()
    window.refresh_tab_interaction(tab_id)
    assert editor.textCursor().position() == 5
    editor.undo()
    qapp.processEvents()
    assert session.state.get_tab(tab_id).save.comment == "original"
    assert (
        ctrl.get_tab_snapshot(tab_id).artifacts[0].artifacts[0].status
        is SaveStatus.UNSAVED_CHANGES
    )
    assert [edit.toolTip() for edit in center.findChildren(QLineEdit)] == [
        str(output.output.path) for output in report.outputs
    ]
    ctrl.save_control.save_data(tab_id, origin=GUI_ORIGIN)
    assert (
        ctrl.get_tab_snapshot(tab_id).artifacts[0].artifacts[0].status
        is SaveStatus.SAVED
    )
    ctrl.save_control.set_comment(tab_id, "external edit")
    window.refresh_tab_interaction(tab_id)
    assert editor.toPlainText() == "external edit"


@pytest.mark.parametrize(
    ("status", "unsaved"),
    [
        (SaveStatus.NOT_SAVED, True),
        (SaveStatus.PARTIALLY_SAVED, True),
        (SaveStatus.UNSAVED_CHANGES, True),
        (SaveStatus.SAVED, False),
    ],
)
def test_tab_close_query_reads_only_data_artifact(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
    status: SaveStatus,
    unsaved: bool,
) -> None:
    session, ctrl, window, tab_id = live_ui
    tab = window.findChild(ExpTabWidget)
    assert tab is not None
    if status is not SaveStatus.NOT_SAVED:
        current = ctrl.get_tab_snapshot(tab_id)
        outputs = tuple(
            OutputResult(
                member.key,
                Output(output.format, output.path),
                "failed"
                if status is SaveStatus.PARTIALLY_SAVED and index == 1
                else "saved",
                "blocked output"
                if status is SaveStatus.PARTIALLY_SAVED and index == 1
                else None,
            )
            for section in current.artifacts
            for artifact in section.artifacts
            for member in artifact.members
            for index, output in enumerate(member.outputs)
        )
        session.state.commit_save_report(
            tab_id, SaveReport(session.metadata.run_id, outputs)
        )
        if status is SaveStatus.UNSAVED_CHANGES:
            ctrl.save_control.set_comment(tab_id, "changed")
    session.state.update_tab_analyze(
        tab_id, PaneSummary(), None, analyze_params_instance=PaneParams()
    )
    current = ctrl.get_tab_snapshot(tab_id)
    assert (
        next(
            artifact.status
            for section in current.artifacts
            for artifact in section.artifacts
            if artifact.ref == _RUN
        )
        is status
    )
    window.refresh_tab_interaction(tab_id, current)
    assert tab.has_unsaved_data() is unsaved
    session.state.clear_tab_results(tab_id)
    window.refresh_tab_interaction(tab_id)
    assert not tab.has_unsaved_data()
