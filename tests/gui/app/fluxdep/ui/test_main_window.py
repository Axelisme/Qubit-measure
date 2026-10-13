"""Headless tests for the fluxdep-gui MainWindow shell.

Drives the window through the Controller (not the user dialogs, which would
block), and asserts the spectrum list reflects State.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.models import TransitionDict
from zcu_tools.analysis.fluxdep.search import ParamBounds
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.event_bus import (
    ActiveSpectrumChangedPayload,
    SpectrumAddedPayload,
    SpectrumChangedPayload,
    SpectrumRemovedPayload,
)
from zcu_tools.gui.app.fluxdep.state import FluxDepState
from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import EventOrigin
from zcu_tools.gui.project import ProjectInfo

from tests.gui.app.fluxdep._search_result import (
    perfect_fit_result,
    synthetic_search_input,
)
from tests.gui.app.fluxdep._window import (
    command,
    current_page,
    editor,
    fit_panel,
    list_labels,
    pages,
    project_label,
    show_page,
    spectra_list,
)


@pytest.fixture
def window(qapp):
    ctrl = Controller(FluxDepState())
    win = MainWindow(ctrl)
    yield win
    win.close()
    win.deleteLater()


def test_window_builds_empty(window):
    assert spectra_list(window).count() == 0
    assert window.windowTitle() == "fluxdep-gui"


def test_shell_exposes_top_bar_commands_and_two_pages(window):
    for name in (
        "fluxdep.project",
        "fluxdep.spectrum.add",
        "fluxdep.spectra.restore",
        "fluxdep.spectra.export",
        "fluxdep.spectrum.remove",
        "control_indicator",
    ):
        command(window, name)
    tabs = pages(window)
    assert [tabs.tabText(i) for i in range(tabs.count())] == ["Spectra", "Fit"]
    assert current_page(window) == "spectra"


def test_project_label_names_the_project_after_setup(window):
    assert "Q5_2D" not in project_label(window).text()
    window._ctrl.setup_project(ProjectInfo(chip_name="Q5_2D", qub_name="Q1"))
    assert project_label(window).text() == "Project: Q5_2D / Q1"


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
    labels = list_labels(window)
    assert len(labels) == 1
    assert name in labels[0]
    assert "OneTone" in labels[0]
    assert "loaded" in labels[0]  # not yet aligned
    assert "0 pts" in labels[0]


def test_stage_marker_updates_on_alignment(window, spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    name = window._ctrl.load_spectrum(filepath, spec_type="OneTone")
    window._ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    assert "aligned" in list_labels(window)[0]
    window._ctrl.set_points(name, np.array([0.0, 1.0]), np.array([5.0, 5.1]))
    label = list_labels(window)[0]
    assert "sampled" in label and "2 pts" in label


def test_list_rows_show_pending_draft_and_filter_exclusions(qapp, cross_controller):
    ctrl = cross_controller
    win = MainWindow(ctrl)
    try:
        assert not any("draft" in label for label in list_labels(win))
        assert not any("excluded" in label for label in list_labels(win))
        ctrl.interactive.begin_line_pick("b")
        rows = dict(zip(ctrl.list_spectrums(), list_labels(win), strict=True))
        assert "align draft" in rows["b"]
        assert "draft" not in rows["a"]
        ctrl.interactive.cancel("b")
        ctrl.interactive.begin_sample_pick("a")
        rows = dict(zip(ctrl.list_spectrums(), list_labels(win), strict=True))
        assert "sample draft" in rows["a"]
        ctrl.interactive.cancel("a")
        # Joint cloud order is a (3 points), empty, b (1 point).
        ctrl.set_selection(np.array([True, False, True, True]))
        rows = dict(zip(ctrl.list_spectrums(), list_labels(win), strict=True))
        assert "1 excluded" in rows["a"] and "3 pts" in rows["a"]
        assert "excluded" not in rows["b"]
        assert not any("draft" in label for label in rows.values())
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_remove_updates_list(window, spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    name = window._ctrl.load_spectrum(filepath, spec_type="OneTone")
    window._ctrl.remove_spectrum(name)
    assert list_labels(window) == []


def test_selecting_list_item_sets_active(window, spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    name = window._ctrl.load_spectrum(filepath, spec_type="OneTone")
    spectra_list(window).setCurrentRow(0)  # emits currentItemChanged
    assert window._ctrl.state.active_spectrum == name


# --- editing-area stage switching ------------------------------------------


def test_picking_unaligned_from_list_opens_line_picker(window, spectrum_hdf5):
    from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget

    filepath, *_ = spectrum_hdf5
    name = window._ctrl.load_spectrum(filepath, spec_type="OneTone")
    # A human list pick is the Local write intention that opens the stage picker.
    spectra_list(window).setCurrentRow(0)
    assert window._ctrl.state.active_spectrum == name
    assert isinstance(editor(window), LinePickerWidget)
    assert window._ctrl.interactive.current_line_pick() is not None


def test_aligned_onetone_shows_onetone_widget(window, spectrum_hdf5):
    from zcu_tools.gui.app.fluxdep.ui.interactive.onetone import OneToneWidget

    filepath, *_ = spectrum_hdf5
    name = window._ctrl.load_spectrum(filepath, spec_type="OneTone")
    window._ctrl.set_active_spectrum(name)
    window._ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    assert isinstance(editor(window), OneToneWidget)


def test_aligned_twotone_shows_find_points_widget(window, spectrum_hdf5):
    from zcu_tools.gui.app.fluxdep.ui.interactive.find_points import FindPointsWidget

    filepath, *_ = spectrum_hdf5
    name = window._ctrl.load_spectrum(filepath, spec_type="TwoTone")
    window._ctrl.set_active_spectrum(name)
    window._ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    assert isinstance(editor(window), FindPointsWidget)


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
    assert isinstance(editor(window), ResultPreviewWidget)


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
        for b in editor(window).findChildren(QPushButton)
        if b.text() == "Re-pick lines"
    ).click()
    assert not window._ctrl.state.spectrums[name].aligned
    assert isinstance(editor(window), LinePickerWidget)


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
        for b in editor(window).findChildren(QPushButton)
        if b.text() == "Re-select points"
    ).click()
    assert not window._ctrl.state.spectrums[name].points_completed
    assert isinstance(editor(window), OneToneWidget)


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
            shown = editor(win)
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
                b for b in shown.findChildren(QPushButton) if b.text() == "Finish"
            ).click()
            entry = ctrl.state.spectrums[name]
            assert entry.points_completed and entry.point_count == 0
            preview = editor(win)
            assert isinstance(preview, ResultPreviewWidget)
            assert "sampled" in list_labels(win)[0]
            if attempt == 0:
                next(
                    b
                    for b in preview.findChildren(QPushButton)
                    if b.text() == "Re-select points"
                ).click()
                assert not ctrl.state.spectrums[name].points_completed
                assert not isinstance(editor(win), ResultPreviewWidget)
        preview = editor(win)
        assert isinstance(preview, ResultPreviewWidget)
        next(
            b for b in preview.findChildren(QPushButton) if b.text() == "Re-pick lines"
        ).click()
        assert isinstance(editor(win), LinePickerWidget)
        assert not ctrl.state.spectrums[name].aligned
        assert ctrl.state.spectrums[name].points_completed
        label = list_labels(win)[0]
        assert "sampled" not in label and "align draft" in label
        shown = editor(win)
        next(b for b in shown.findChildren(QPushButton) if b.text() == "Finish").click()
        assert isinstance(editor(win), ResultPreviewWidget)
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
        preview = editor(win)
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
            b for b in editor(win).findChildren(QPushButton) if b.text() == "Finish"
        ).click()
        assert isinstance(editor(win), ResultPreviewWidget)
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
    command(window, "fluxdep.spectrum.remove").click()  # remove a → focus b
    assert window._ctrl.state.active_spectrum == b


def test_fit_page_without_points_opens_no_filter_input(window, spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    ctrl = window._ctrl
    name = ctrl.load_spectrum(filepath, "OneTone")
    ctrl.set_alignment(name, 0.0, 1.0)
    ctrl.set_points(name, np.empty(0), np.empty(0))
    show_page(window, "fit")
    panel = fit_panel(window)
    assert panel is not None and panel.current_tab == "filter"
    assert ctrl.interactive.current_cross_selection() is None


def _record_fit(ctrl: Controller, search_db: str) -> None:
    ctrl.set_fit_params(
        search_db,
        (3.0, 12.0),
        (0.3, 1.5),
        (0.2, 1.2),
        TransitionDict({"transitions": [(0, 1)]}),
        None,
        None,
    )
    bounds = ParamBounds(EJ=(3.0, 12.0), EC=(0.3, 1.5), EL=(0.2, 1.2))
    fluxs, freqs = ctrl.selected_pointcloud()
    ctrl.record_search_result(
        perfect_fit_result(fluxs, freqs, bounds),
        synthetic_search_input(ctrl.state, fluxs, freqs, bounds),
    )


def test_spectrum_work_does_not_replace_the_fit_page(
    qapp, cross_controller, search_database_hdf5
):
    ctrl = cross_controller
    _record_fit(ctrl, search_database_hdf5)
    win = MainWindow(ctrl)
    win.show()
    try:
        show_page(win, "fit")
        panel = fit_panel(win)
        assert panel is not None
        panel.show_tab("search")
        # Display and stage changes of the active spectrum keep Fit on screen.
        ctrl.set_active_spectrum("b")
        ctrl.reset_points("b")
        assert current_page(win) == "fit"
        assert fit_panel(win) is panel and panel.current_tab == "search"

        show_page(win, "spectra")
        spectra_list(win).setCurrentRow(0)
        assert ctrl.state.active_spectrum == "a"
        assert fit_panel(win) is panel
        show_page(win, "fit")
        assert fit_panel(win) is panel and panel.current_tab == "search"
        assert ctrl.state.fit.has_result
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_agent_display_change_brings_spectra_page_back(qapp, cross_controller):
    ctrl = cross_controller
    win = MainWindow(ctrl)
    try:
        show_page(win, "fit")
        with ctrl.bus.origin(EventOrigin(kind="agent")):
            ctrl.set_active_spectrum("b")
        assert current_page(win) == "spectra"
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()
