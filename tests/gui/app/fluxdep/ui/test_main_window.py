"""Headless tests for the fluxdep-gui MainWindow shell.

Drives the window through the Controller (not the user dialogs, which would
block), and asserts the spectrum list reflects State.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from qtpy.QtWidgets import QStackedWidget, QWidget
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.event_bus import (
    ActiveSpectrumChangedPayload,
    SpectrumAddedPayload,
    SpectrumChangedPayload,
    SpectrumRemovedPayload,
)
from zcu_tools.gui.app.fluxdep.state import FluxDepState
from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow


@pytest.fixture
def window(qapp):
    ctrl = Controller(FluxDepState())
    win = MainWindow(ctrl)
    yield win
    win.close()
    win.deleteLater()


def _list_labels(win: MainWindow) -> list[str]:
    items = [win._list.item(i) for i in range(win._list.count())]
    return [it.text() for it in items if it is not None]


def _editor(win: MainWindow) -> QWidget:
    stack = win.findChild(QStackedWidget)
    assert stack is not None
    widget = stack.currentWidget()
    assert widget is not None
    return widget


def test_window_builds_empty(window):
    assert window._list.count() == 0
    assert window.windowTitle() == "fluxdep-gui"


def test_window_close_removes_event_bus_subscriptions(window):
    for payload_type in (
        SpectrumAddedPayload,
        SpectrumRemovedPayload,
        SpectrumChangedPayload,
        ActiveSpectrumChangedPayload,
    ):
        assert window._ctrl.bus._subs.get(payload_type)

    window.close()

    for payload_type in (
        SpectrumAddedPayload,
        SpectrumRemovedPayload,
        SpectrumChangedPayload,
        ActiveSpectrumChangedPayload,
    ):
        assert window._ctrl.bus._subs.get(payload_type, []) == []


def test_load_refreshes_list(window, spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    name = window._ctrl.load_spectrum(filepath, spec_type="OneTone")
    labels = _list_labels(window)
    assert len(labels) == 1
    assert name in labels[0]
    assert "OneTone" in labels[0]
    assert "new" in labels[0]  # not yet aligned


def test_stage_marker_updates_on_alignment(window, spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    name = window._ctrl.load_spectrum(filepath, spec_type="OneTone")
    window._ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    assert "align" in _list_labels(window)[0]
    window._ctrl.set_points(name, np.array([0.0, 1.0]), np.array([5.0, 5.1]))
    assert "pts" in _list_labels(window)[0]


def test_remove_updates_list(window, spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    name = window._ctrl.load_spectrum(filepath, spec_type="OneTone")
    window._ctrl.remove_spectrum(name)
    assert _list_labels(window) == []


def test_selecting_list_item_sets_active(window, spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    name = window._ctrl.load_spectrum(filepath, spec_type="OneTone")
    window._list.setCurrentRow(0)  # emits currentItemChanged
    assert window._ctrl.state.active_spectrum == name


# --- editing-area stage switching ------------------------------------------


def test_active_unaligned_shows_line_picker(window, spectrum_hdf5):
    from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget

    filepath, *_ = spectrum_hdf5
    name = window._ctrl.load_spectrum(filepath, spec_type="OneTone")
    window._ctrl.set_active_spectrum(name)
    assert isinstance(window._current_editor, LinePickerWidget)


def test_aligned_onetone_shows_onetone_widget(window, spectrum_hdf5):
    from zcu_tools.gui.app.fluxdep.ui.interactive.onetone import OneToneWidget

    filepath, *_ = spectrum_hdf5
    name = window._ctrl.load_spectrum(filepath, spec_type="OneTone")
    window._ctrl.set_active_spectrum(name)
    window._ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    assert isinstance(window._current_editor, OneToneWidget)


def test_aligned_twotone_shows_find_points_widget(window, spectrum_hdf5):
    from zcu_tools.gui.app.fluxdep.ui.interactive.find_points import FindPointsWidget

    filepath, *_ = spectrum_hdf5
    name = window._ctrl.load_spectrum(filepath, spec_type="TwoTone")
    window._ctrl.set_active_spectrum(name)
    window._ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    assert isinstance(window._current_editor, FindPointsWidget)


def test_selected_shows_result_preview(window, spectrum_hdf5):
    from zcu_tools.gui.app.fluxdep.ui.interactive.result_preview import (
        ResultPreviewWidget,
    )

    filepath, *_ = spectrum_hdf5
    name = window._ctrl.load_spectrum(filepath, spec_type="OneTone")
    window._ctrl.set_active_spectrum(name)
    window._ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    window._ctrl.set_points(name, np.array([0.0, 1.0]), np.array([5.0, 5.1]))
    # finished → a read-only result preview (not the placeholder)
    assert isinstance(window._current_editor, ResultPreviewWidget)


def test_repick_lines_reopens_line_picker(window, spectrum_hdf5):
    from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget

    filepath, *_ = spectrum_hdf5
    name = window._ctrl.load_spectrum(filepath, spec_type="OneTone")
    window._ctrl.set_active_spectrum(name)
    window._ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    window._ctrl.set_points(name, np.array([0.0, 1.0]), np.array([5.0, 5.1]))
    from qtpy.QtWidgets import QPushButton

    next(
        b
        for b in window._current_editor.findChildren(QPushButton)
        if b.text() == "Re-pick lines"
    ).click()
    assert not window._ctrl.state.spectrums[name].aligned
    assert isinstance(window._current_editor, LinePickerWidget)


def test_reselect_points_reopens_selector(window, spectrum_hdf5):
    from zcu_tools.gui.app.fluxdep.ui.interactive.onetone import OneToneWidget

    filepath, *_ = spectrum_hdf5
    name = window._ctrl.load_spectrum(filepath, spec_type="OneTone")
    window._ctrl.set_active_spectrum(name)
    window._ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    window._ctrl.set_points(name, np.array([0.0, 1.0]), np.array([5.0, 5.1]))
    from qtpy.QtWidgets import QPushButton

    next(
        b
        for b in window._current_editor.findChildren(QPushButton)
        if b.text() == "Re-select points"
    ).click()
    assert not window._ctrl.state.spectrums[name].points_completed
    assert isinstance(window._current_editor, OneToneWidget)


@pytest.mark.parametrize("kind", ["OneTone", "TwoTone"])
def test_empty_finish_advances_preview_and_reselect_can_finish_again(
    qapp, onetone_controller, twotone_controller, kind
):
    from qtpy.QtWidgets import QPushButton
    from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget
    from zcu_tools.gui.app.fluxdep.ui.interactive.result_preview import (
        ResultPreviewWidget,
    )

    ctrl = onetone_controller if kind == "OneTone" else twotone_controller
    name = "one" if kind == "OneTone" else "two"
    win = MainWindow(ctrl)
    try:
        for attempt in range(2):
            editor = _editor(win)
            assert editor is not None
            if kind == "OneTone":
                context = ctrl.interactive.current_onetone_pick()
            else:
                context = ctrl.interactive.current_twotone_pick()
            assert context is not None
            if kind == "OneTone":
                context.plugin.set_threshold.execute(context.session, 5.0)
            else:
                context.plugin.clear.execute(context.session, None)
            next(
                b for b in editor.findChildren(QPushButton) if b.text() == "Finish"
            ).click()
            entry = ctrl.state.spectrums[name]
            assert entry.points_completed and entry.point_count == 0
            preview = _editor(win)
            assert isinstance(preview, ResultPreviewWidget)
            assert "✓pts" in _list_labels(win)[0]
            if attempt == 0:
                next(
                    b
                    for b in preview.findChildren(QPushButton)
                    if b.text() == "Re-select points"
                ).click()
                assert not ctrl.state.spectrums[name].points_completed
                assert not isinstance(_editor(win), ResultPreviewWidget)
        preview = _editor(win)
        assert isinstance(preview, ResultPreviewWidget)
        next(
            b for b in preview.findChildren(QPushButton) if b.text() == "Re-pick lines"
        ).click()
        assert isinstance(_editor(win), LinePickerWidget)
        assert not ctrl.state.spectrums[name].aligned
        assert ctrl.state.spectrums[name].points_completed
        assert "✓pts" not in _list_labels(win)[0]
        editor = _editor(win)
        next(
            b for b in editor.findChildren(QPushButton) if b.text() == "Finish"
        ).click()
        assert isinstance(_editor(win), ResultPreviewWidget)
        assert ctrl.state.spectrums[name].point_count == 0
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_empty_preview_reselect_can_commit_nonempty_points(qapp, onetone_controller):
    from qtpy.QtWidgets import QPushButton
    from zcu_tools.gui.app.fluxdep.ui.interactive.result_preview import (
        ResultPreviewWidget,
    )

    ctrl = onetone_controller
    ctrl.set_points("one", np.empty(0), np.empty(0))
    win = MainWindow(ctrl)
    try:
        preview = _editor(win)
        assert isinstance(preview, ResultPreviewWidget)
        next(
            b
            for b in preview.findChildren(QPushButton)
            if b.text() == "Re-select points"
        ).click()
        context = ctrl.interactive.current_onetone_pick()
        assert context is not None
        context.plugin.set_threshold.execute(context.session, 0.1)
        next(
            b for b in _editor(win).findChildren(QPushButton) if b.text() == "Finish"
        ).click()
        assert isinstance(_editor(win), ResultPreviewWidget)
        assert ctrl.state.spectrums["one"].points_completed
        assert ctrl.state.spectrums["one"].point_count == 2
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_remove_focuses_next_spectrum(window, spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    a = window._ctrl.load_spectrum(filepath, spec_type="OneTone")
    # second load replaces same basename; use a distinct file path via tmp copy
    import shutil
    import tempfile

    second_path = os.path.join(tempfile.mkdtemp(), "other.hdf5")
    shutil.copy(filepath, second_path)
    b = window._ctrl.load_spectrum(second_path, spec_type="OneTone")
    window._ctrl.set_active_spectrum(a)
    window._on_remove_clicked()  # remove a → focus the next one (b)
    assert window._ctrl.state.active_spectrum == b


def test_analyze_refuses_completed_empty_spectrum(window, spectrum_hdf5, monkeypatch):
    from qtpy.QtWidgets import QMessageBox, QPushButton
    from zcu_tools.gui.app.fluxdep.ui.analyze_panel import AnalyzePanelWidget

    filepath, *_ = spectrum_hdf5
    ctrl = window._ctrl
    name = ctrl.load_spectrum(filepath, "OneTone")
    ctrl.set_alignment(name, 0.0, 1.0)
    ctrl.set_points(name, np.empty(0), np.empty(0))
    errors = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda parent, title, text: errors.append((title, text)),
    )
    next(b for b in window.findChildren(QPushButton) if b.text() == "Analyze…").click()
    assert errors and errors[0][0] == "No points"
    assert window.findChild(AnalyzePanelWidget) is None


def test_button_labels(window):
    from qtpy.QtWidgets import QPushButton

    labels = [b.text() for b in window.findChildren(QPushButton)]
    assert "Add…" in labels  # raw spectrum add
    assert "Restore" in labels  # processed spectrums.hdf5
    assert "Export" in labels
