"""Headless tests for the fluxdep-gui MainWindow shell.

Drives the window through the Controller (not the user dialogs, which would
block), and asserts the spectrum list reflects State.
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.text import Text
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

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui.app.fluxdep._search_result import (
    perfect_fit_result,
    synthetic_search_input,
)
from tests.gui.app.fluxdep._window import (
    choose_file,
    command,
    current_page,
    current_stage,
    drawn_row_text,
    editor,
    fit_panel,
    list_labels,
    list_tooltips,
    pages,
    picker_button,
    project_label,
    show_page,
    spectra_list,
    stage_button,
    stage_status,
)


@pytest.fixture
def ctrl():
    """The Controller the window fixture drives."""
    return Controller(FluxDepState())


@pytest.fixture
def window(qapp, ctrl):
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


def test_project_label_names_the_project_after_setup(window, ctrl):
    assert "Q5_2D" not in project_label(window).text()
    ctrl.setup_project(ProjectInfo(chip_name="Q5_2D", qub_name="Q1"))
    assert project_label(window).text() == "Project: Q5_2D / Q1"


def test_window_close_removes_event_bus_subscriptions(window, ctrl):
    for payload_type in (
        SpectrumAddedPayload,
        SpectrumRemovedPayload,
        SpectrumChangedPayload,
        ActiveSpectrumChangedPayload,
    ):
        assert ctrl.bus._subs.get(payload_type)

    window.close()

    for payload_type in (
        SpectrumAddedPayload,
        SpectrumRemovedPayload,
        SpectrumChangedPayload,
        ActiveSpectrumChangedPayload,
    ):
        assert ctrl.bus._subs.get(payload_type, []) == []


def test_load_refreshes_list(window, ctrl, spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    name = ctrl.load_spectrum(filepath, spec_type="OneTone")
    # The literal name on one line, kind · stage · published points below it.
    assert list_labels(window) == [f"{name}\nOneTone · loaded · 0 pts"]
    assert list_tooltips(window) == [name]


_LONG_NAME = "a_very_long_spectrum_name_from_a_nested_measurement_run_Q1_flux_1534.hdf5"


@pytest.fixture
def long_name_window(qapp, cross_controller):
    """A shown 1200x800 window whose list holds a, empty and a long-named b."""
    ctrl = cross_controller
    entry = ctrl.state.spectrums["b"]
    ctrl.remove_spectrum("b")
    ctrl.state.put_spectrum(replace(entry, name=_LONG_NAME))
    win = MainWindow(ctrl)
    win.resize(1200, 800)
    win.show()
    qapp.processEvents()
    yield win
    win.close()
    win.deleteLater()
    qapp.processEvents()


def test_list_rows_put_the_name_above_its_details(long_name_window, cross_controller):
    win = long_name_window
    ctrl = cross_controller
    ctrl.interactive.begin_line_pick(_LONG_NAME)
    ctrl.set_selection(np.array([True, False, True, True]))
    assert list_labels(win) == [
        "a\nTwoTone · sampled · 3 pts · 1 excluded",
        "empty\nTwoTone · sampled · 0 pts",
        f"{_LONG_NAME}\nTwoTone · sampled · 1 pts · align draft",
    ]
    assert list_tooltips(win) == ["a", "empty", _LONG_NAME]
    spectra = spectra_list(win)
    assert spectra.sizeHintForRow(0) >= 2 * spectra.fontMetrics().lineSpacing(), (
        "each row must be two text lines tall"
    )
    # A name wider than its cell is drawn elided in the middle; details stay whole.
    first, details = drawn_row_text(win, 2, 220).split("\n")
    head, ellipsis, tail = first.partition("…")
    assert ellipsis and head and tail
    assert _LONG_NAME.startswith(head) and _LONG_NAME.endswith(tail)
    assert details == "TwoTone · sampled · 1 pts · align draft"
    assert drawn_row_text(win, 0, 220) == "a\nTwoTone · sampled · 3 pts · 1 excluded"


def test_list_never_scrolls_sideways(qapp, long_name_window):
    scrollbar = spectra_list(long_name_window).horizontalScrollBar()
    assert scrollbar is not None
    for width in (1200, 700):
        long_name_window.resize(width, 800)
        qapp.processEvents()
        assert not scrollbar.isVisible()


def test_stage_marker_updates_on_alignment(window, ctrl, spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    name = ctrl.load_spectrum(filepath, spec_type="OneTone")
    ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    assert "aligned" in list_labels(window)[0]
    ctrl.set_points(name, np.array([0.0, 1.0]), np.array([5.0, 5.1]))
    label = list_labels(window)[0]
    assert "sampled" in label and "2 pts" in label


def test_list_row_pending_matches_spectrum_list_for_a_spectrum_named_selection(
    qapp, cross_controller
):
    # "selection" also addresses the global Filter draft; the row must report
    # the same pending kind as spectrum.list, which attributes it to no spectrum.
    ctrl = cross_controller
    entry = ctrl.state.spectrums["b"]
    ctrl.remove_spectrum("b")
    ctrl.state.put_spectrum(replace(entry, name="selection"))
    win = MainWindow(ctrl)
    try:
        ctrl.interactive.begin_cross_selection()
        rows = dict(zip(ctrl.list_spectrums(), list_labels(win), strict=True))
        assert "sampled" in rows["selection"]
        assert not any("draft" in label for label in rows.values())
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


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


def test_remove_updates_list(window, ctrl, spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    name = ctrl.load_spectrum(filepath, spec_type="OneTone")
    ctrl.remove_spectrum(name)
    assert list_labels(window) == []


def test_selecting_list_item_sets_active(window, ctrl, spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    name = ctrl.load_spectrum(filepath, spec_type="OneTone")
    spectra_list(window).setCurrentRow(0)  # emits currentItemChanged
    assert ctrl.state.active_spectrum == name


# --- editing-area stage switching ------------------------------------------


def test_picking_from_list_shows_the_published_preview_first(
    window, ctrl, spectrum_hdf5
):
    from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget
    from zcu_tools.gui.app.fluxdep.ui.interactive.result_preview import (
        ResultPreviewWidget,
    )

    filepath, *_ = spectrum_hdf5
    name = ctrl.load_spectrum(filepath, spec_type="OneTone")
    # Picking a spectrum only shows it; a stage click is the intent to edit.
    spectra_list(window).setCurrentRow(0)
    assert ctrl.state.active_spectrum == name
    assert isinstance(editor(window), ResultPreviewWidget)
    assert ctrl.interactive.inspect(name) is None
    assert current_stage(window) is None
    stage_button(window, "align").click()
    assert isinstance(editor(window), LinePickerWidget)
    assert ctrl.interactive.current_line_pick(name) is not None
    assert current_stage(window) == "align"


def test_stage_bar_runs_align_then_sample_above_the_editing_area(
    qapp, cross_controller
):
    from qtpy.QtCore import QPoint
    from qtpy.QtWidgets import QStackedWidget

    win = MainWindow(cross_controller)
    win.resize(1200, 800)
    win.show()
    qapp.processEvents()
    try:
        area = win.findChild(QStackedWidget, "fluxdep.editor")
        assert area is not None
        area_top = area.mapTo(win, QPoint(0, 0)).y()
        align = stage_button(win, "align").mapTo(win, QPoint(0, 0))
        sample = stage_button(win, "sample").mapTo(win, QPoint(0, 0))
        assert align.y() < area_top and sample.y() < area_top
        assert align.y() == sample.y() and align.x() < sample.x()
        assert stage_button(win, "align").isVisible()
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_stage_bar_marks_open_locked_done_and_draft_stages(window, ctrl, spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    name = ctrl.load_spectrum(filepath, spec_type="OneTone")
    ctrl.set_active_spectrum(name)
    # Unaligned: Align can be opened; Sample stays locked until Align is done.
    assert stage_status(window, "align") == "open"
    assert stage_status(window, "sample") == "locked"
    assert stage_button(window, "align").isEnabled()
    assert not stage_button(window, "sample").isEnabled()
    stage_button(window, "sample").click()
    assert ctrl.interactive.inspect(name) is None

    # Publishing the alignment advances the holder to a sampling draft.
    ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    assert stage_status(window, "align") == "done"
    assert stage_status(window, "sample") == "draft"
    assert current_stage(window) == "sample"
    ctrl.interactive.cancel(name)
    # Aligned without a draft: both stages can be opened.
    assert stage_status(window, "align") == "done"
    assert stage_status(window, "sample") == "open"
    assert stage_button(window, "align").isEnabled()
    assert stage_button(window, "sample").isEnabled()
    assert current_stage(window) is None

    ctrl.set_points(name, np.array([0.0, 1.0]), np.array([5.0, 5.1]))
    # Sampled: both stages are done and either can be reopened.
    assert stage_status(window, "align") == "done"
    assert stage_status(window, "sample") == "done"
    assert stage_button(window, "align").isEnabled()
    assert stage_button(window, "sample").isEnabled()

    stage_button(window, "align").click()
    assert stage_status(window, "align") == "draft"
    assert stage_status(window, "sample") == "done"
    assert current_stage(window) == "align"


def test_aligned_onetone_shows_onetone_widget(window, ctrl, spectrum_hdf5):
    from zcu_tools.gui.app.fluxdep.ui.interactive.onetone import OneToneWidget

    filepath, *_ = spectrum_hdf5
    name = ctrl.load_spectrum(filepath, spec_type="OneTone")
    ctrl.set_active_spectrum(name)
    ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    assert isinstance(editor(window), OneToneWidget)


def test_aligned_twotone_shows_find_points_widget(window, ctrl, spectrum_hdf5):
    from zcu_tools.gui.app.fluxdep.ui.interactive.find_points import FindPointsWidget

    filepath, *_ = spectrum_hdf5
    name = ctrl.load_spectrum(filepath, spec_type="TwoTone")
    ctrl.set_active_spectrum(name)
    ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    assert isinstance(editor(window), FindPointsWidget)


def test_selected_shows_result_preview(window, ctrl, spectrum_hdf5):
    from zcu_tools.gui.app.fluxdep.ui.interactive.result_preview import (
        ResultPreviewWidget,
    )

    filepath, *_ = spectrum_hdf5
    name = ctrl.load_spectrum(filepath, spec_type="OneTone")
    ctrl.set_active_spectrum(name)
    ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    ctrl.set_points(name, np.array([0.0, 1.0]), np.array([5.0, 5.1]))
    # finished → a read-only result preview (not the placeholder)
    assert isinstance(editor(window), ResultPreviewWidget)


def _sampled_onetone(ctrl, spectrum_hdf5) -> str:
    filepath, *_ = spectrum_hdf5
    name = ctrl.load_spectrum(filepath, spec_type="OneTone")
    ctrl.set_active_spectrum(name)
    ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    ctrl.set_points(name, np.array([0.0, 1.0]), np.array([5.0, 5.1]))
    return name


def test_unaligned_spectrum_opens_only_the_align_stage(window, ctrl, spectrum_hdf5):
    from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget
    from zcu_tools.gui.app.fluxdep.ui.interactive.result_preview import (
        ResultPreviewWidget,
    )

    filepath, *_ = spectrum_hdf5
    name = ctrl.load_spectrum(filepath, spec_type="OneTone")
    # A display change alone shows the published spectrum without input.
    ctrl.set_active_spectrum(name)
    assert isinstance(editor(window), ResultPreviewWidget)
    assert ctrl.interactive.inspect(name) is None
    stage_button(window, "align").click()
    assert isinstance(editor(window), LinePickerWidget)
    assert ctrl.interactive.current_line_pick(name) is not None


def test_stage_realign_opens_a_draft_without_unpublishing(window, ctrl, spectrum_hdf5):
    from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget

    name = _sampled_onetone(ctrl, spectrum_hdf5)
    stage_button(window, "align").click()
    picker = editor(window)
    assert isinstance(picker, LinePickerWidget)
    entry = ctrl.state.spectrums[name]
    assert entry.aligned and entry.points_completed and entry.point_count == 2
    label = list_labels(window)[0]
    assert "sampled" in label and "align draft" in label
    # Re-aligning an aligned spectrum can still move on to sampling.
    assert stage_button(window, "sample").isEnabled()


def test_stage_sample_opens_sampling_and_keeps_points_until_finish(
    window, ctrl, spectrum_hdf5
):
    from zcu_tools.gui.app.fluxdep.ui.interactive.onetone import OneToneWidget

    name = _sampled_onetone(ctrl, spectrum_hdf5)
    stage_button(window, "sample").click()
    assert isinstance(editor(window), OneToneWidget)
    entry = ctrl.state.spectrums[name]
    assert entry.points_completed and entry.point_count == 2
    assert "sample draft" in list_labels(window)[0]


def test_gui_add_opens_line_picker_for_the_loaded_spectrum(qapp, spectrum_hdf5):
    from qtpy.QtWidgets import QApplication, QDialogButtonBox, QPushButton
    from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget

    ctrl = Controller(FluxDepState())
    win = MainWindow(ctrl)
    win.show()
    try:
        command(win, "fluxdep.spectrum.add").click()
        dialog = QApplication.activeModalWidget()
        assert dialog is not None
        browse = next(
            b for b in dialog.findChildren(QPushButton) if b.text() == "Browse…"
        )
        choose_file(spectrum_hdf5[0], browse.click)
        box = dialog.findChild(QDialogButtonBox)
        assert box is not None
        ok = box.button(QDialogButtonBox.StandardButton.Ok)
        assert ok is not None and ok.isEnabled()
        ok.click()
        name = ctrl.state.active_spectrum
        assert name is not None
        picker = editor(win)
        assert isinstance(picker, LinePickerWidget)
        active = ctrl.interactive.inspect(name)
        assert active is not None and active.kind == "line"
        assert current_page(win) == "spectra"
        # An unaligned spectrum has no sampling step to switch to yet.
        assert not stage_button(win, "sample").isEnabled()
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_realign_from_sample_draft_confirms_and_cancel_keeps_draft(
    qapp, onetone_controller
):
    from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget
    from zcu_tools.gui.app.fluxdep.ui.interactive.onetone import OneToneWidget

    ctrl = onetone_controller
    dialogs = RecordingDialogPresenter(destructive_answers=[False, True])
    win = MainWindow(ctrl, dialog_presenter=dialogs)
    try:
        sampler = editor(win)
        assert isinstance(sampler, OneToneWidget)
        draft = ctrl.interactive.inspect("one")
        assert draft is not None and draft.kind == "onetone"
        draft.context.plugin.set_threshold.execute(draft.context.session, 0.1)

        stage_button(win, "align").click()
        assert [call.kind for call in dialogs.calls] == ["destructive_confirm"]
        assert ctrl.interactive.inspect("one") == draft
        assert editor(win) is sampler
        assert draft.context.session.snapshot().threshold == 0.1

        stage_button(win, "align").click()
        assert len(dialogs.calls) == 2
        assert isinstance(editor(win), LinePickerWidget)
        replaced = ctrl.interactive.inspect("one")
        assert replaced is not None and replaced.kind == "line"
        assert ctrl.state.spectrums["one"].aligned
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_stage_switch_confirms_and_cancel_keeps_the_other_draft(qapp, cross_controller):
    from zcu_tools.gui.app.fluxdep.ui.interactive.find_points import FindPointsWidget
    from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget

    ctrl = cross_controller
    dialogs = RecordingDialogPresenter(destructive_answers=[False, True, False])
    win = MainWindow(ctrl, dialog_presenter=dialogs)
    try:
        stage_button(win, "align").click()
        aligner = editor(win)
        assert isinstance(aligner, LinePickerWidget)
        draft = ctrl.interactive.inspect("a")
        assert draft is not None and draft.kind == "line"
        # Reopening the stage that holds the draft asks nothing and keeps it.
        stage_button(win, "align").click()
        assert dialogs.calls == []
        assert ctrl.interactive.inspect("a") == draft
        aligner = editor(win)
        assert isinstance(aligner, LinePickerWidget)

        # Declining keeps the align draft, its picker and the stage marks.
        stage_button(win, "sample").click()
        assert [call.kind for call in dialogs.calls] == ["destructive_confirm"]
        assert ctrl.interactive.inspect("a") == draft
        assert editor(win) is aligner
        assert stage_status(win, "align") == "draft"
        assert current_stage(win) == "align"

        # Confirming discards it and opens sampling; published points remain.
        stage_button(win, "sample").click()
        assert [call.kind for call in dialogs.calls] == ["destructive_confirm"] * 2
        assert isinstance(editor(win), FindPointsWidget)
        replaced = ctrl.interactive.inspect("a")
        assert replaced is not None and replaced.kind == "twotone"
        assert stage_status(win, "align") == "done"
        assert stage_status(win, "sample") == "draft"
        assert current_stage(win) == "sample"
        assert ctrl.state.spectrums["a"].point_count == 3

        # Going back to Align asks again; declining keeps the sampling draft.
        stage_button(win, "align").click()
        assert len(dialogs.calls) == 3
        assert ctrl.interactive.inspect("a") == replaced
        assert isinstance(editor(win), FindPointsWidget)
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("kind", ["OneTone", "TwoTone"])
def test_empty_finish_advances_preview_and_reselect_can_finish_again(
    qapp, onetone_controller, twotone_controller, kind
):
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
            picker_button(shown, "finish").click()
            entry = ctrl.state.spectrums[name]
            assert entry.points_completed and entry.point_count == 0
            preview = editor(win)
            assert isinstance(preview, ResultPreviewWidget)
            assert "sampled" in list_labels(win)[0]
            if attempt == 0:
                stage_button(win, "sample").click()
                assert ctrl.state.spectrums[name].points_completed
                assert "sample draft" in list_labels(win)[0]
                assert not isinstance(editor(win), ResultPreviewWidget)
        preview = editor(win)
        assert isinstance(preview, ResultPreviewWidget)
        stage_button(win, "align").click()
        assert isinstance(editor(win), LinePickerWidget)
        assert ctrl.state.spectrums[name].aligned
        assert ctrl.state.spectrums[name].points_completed
        label = list_labels(win)[0]
        assert "sampled" in label and "align draft" in label
        picker_button(editor(win), "finish").click()
        assert isinstance(editor(win), ResultPreviewWidget)
        assert ctrl.state.spectrums[name].point_count == 0
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_empty_preview_reselect_can_commit_nonempty_points(qapp, onetone_controller):
    from zcu_tools.gui.app.fluxdep.ui.interactive.result_preview import (
        ResultPreviewWidget,
    )

    ctrl = onetone_controller
    ctrl.set_points("one", np.empty(0), np.empty(0))
    win = MainWindow(ctrl)
    try:
        preview = editor(win)
        assert isinstance(preview, ResultPreviewWidget)
        stage_button(win, "sample").click()
        context = ctrl.interactive.current_onetone_pick()
        assert context is not None
        context.plugin.set_threshold.execute(context.session, 0.1)
        picker_button(editor(win), "finish").click()
        assert isinstance(editor(win), ResultPreviewWidget)
        assert ctrl.state.spectrums["one"].points_completed
        assert ctrl.state.spectrums["one"].point_count == 2
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_remove_focuses_next_spectrum(window, ctrl, spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    a = ctrl.load_spectrum(filepath, spec_type="OneTone")
    # second load replaces same basename; use a distinct file path via tmp copy
    import shutil
    import tempfile

    second_path = os.path.join(tempfile.mkdtemp(), "other.hdf5")
    shutil.copy(filepath, second_path)
    b = ctrl.load_spectrum(second_path, spec_type="OneTone")
    ctrl.set_active_spectrum(a)
    command(window, "fluxdep.spectrum.remove").click()  # remove a → focus b
    assert ctrl.state.active_spectrum == b


def test_fit_page_without_points_opens_no_filter_input(window, ctrl, spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
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


def _show_canvas(window: MainWindow) -> FigureCanvasQTAgg:
    panel = fit_panel(window)
    assert panel is not None
    return next(
        canvas
        for canvas in panel.findChildren(FigureCanvasQTAgg)
        if canvas.figure.axes
        and canvas.figure.axes[0].get_title().startswith("EJ/EC/EL")
    )


def _figure_labels(canvas: FigureCanvasQTAgg) -> list[Text]:
    """Titles and axis labels of every axes, including secondary axes."""
    texts: list[Text] = []
    for axes in canvas.figure.axes:
        for owner in (axes, *axes.child_axes):
            texts += [owner.title, owner.xaxis.label, owner.yaxis.label]
    return [text for text in texts if text.get_text()]


def test_show_keeps_title_and_axis_labels_inside_a_1200x800_window(
    qapp, cross_controller, search_database_hdf5
):
    ctrl = cross_controller
    _record_fit(ctrl, search_database_hdf5)
    win = MainWindow(ctrl)
    win.resize(1600, 1000)
    win.show()
    try:
        show_page(win, "fit")
        panel = fit_panel(win)
        assert panel is not None
        panel.show_tab("show")
        qapp.processEvents()
        # Redraw at the larger size; the Outdated line lengthens the title.
        ctrl.set_alignment("a", 0.55, 1.05)
        qapp.processEvents()
        win.resize(1200, 800)
        qapp.processEvents()
        canvas = _show_canvas(win)
        canvas.draw()
        renderer = canvas.get_renderer()
        frame = canvas.figure.bbox
        labels = _figure_labels(canvas)
        assert any("Outdated" in text.get_text() for text in labels)
        for text in labels:
            box = text.get_window_extent(renderer)
            inside = (
                box.x0 >= frame.x0
                and box.x1 <= frame.x1
                and box.y0 >= frame.y0
                and box.y1 <= frame.y1
            )
            assert inside, f"{text.get_text()!r} is clipped by the canvas"
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


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


def test_agent_display_change_keeps_the_users_page(qapp, cross_controller):
    from zcu_tools.gui.app.fluxdep.ui.interactive.result_preview import (
        ResultPreviewWidget,
    )

    ctrl = cross_controller
    win = MainWindow(ctrl)
    try:
        show_page(win, "fit")
        before = editor(win)
        with ctrl.bus.origin(EventOrigin(kind="agent")):
            ctrl.set_active_spectrum("b")
        assert current_page(win) == "fit"
        shown = editor(win)
        assert isinstance(shown, ResultPreviewWidget) and shown is not before
        assert ctrl.state.active_spectrum == "b"
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_agent_spectrum_work_updates_spectra_page_while_user_stays_on_fit(
    qapp, cross_controller
):
    from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget

    ctrl = cross_controller
    win = MainWindow(ctrl)
    win.show()
    try:
        show_page(win, "fit")
        panel = fit_panel(win)
        assert panel is not None
        panel.show_tab("search")
        ctrl.control_identity.connect("agent-link", "agent", "owner")
        with ctrl.bus.origin(EventOrigin(kind="agent")):
            context = ctrl.interactive.begin_line_pick("b")
            context.plugin.execute_command(context.session, "swap_lines", {})
        assert current_page(win) == "fit"
        assert panel.current_tab == "search"
        assert ctrl.state.active_spectrum == "b"
        shown = editor(win)
        assert isinstance(shown, LinePickerWidget)
        assert shown.get_result() == (
            context.session.snapshot().flux_half,
            context.session.snapshot().flux_int,
        )
        assert "align draft" in list_labels(win)[2]
        # The stage bar follows the agent's draft without switching pages.
        assert stage_status(win, "align") == "draft"
        assert stage_status(win, "sample") == "done"
        assert current_stage(win) == "align"
        with ctrl.bus.origin(EventOrigin(kind="agent")):
            ctrl.interactive.cancel("b")
            sampling = ctrl.interactive.begin_sample_pick("b")
            sampling.plugin.execute_command(sampling.session, "clear", {})
        assert current_page(win) == "fit"
        assert stage_status(win, "align") == "done"
        assert stage_status(win, "sample") == "draft"
        assert current_stage(win) == "sample"
    finally:
        ctrl.control_identity.reclaim_local()
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_non_holder_stage_click_views_without_opening_a_picker(qapp, cross_controller):
    from zcu_tools.gui.app.fluxdep.ui.interactive.find_points import FindPointsWidget
    from zcu_tools.gui.app.fluxdep.ui.interactive.result_preview import (
        ResultPreviewWidget,
    )

    ctrl = cross_controller
    dialogs = RecordingDialogPresenter()
    win = MainWindow(ctrl, dialog_presenter=dialogs)
    try:
        ctrl.control_identity.connect("agent-link", "agent", "owner")
        # A non-holder click views the published spectrum and opens no input.
        for stage in ("align", "sample"):
            assert stage_button(win, stage).isEnabled()
            stage_button(win, stage).click()
            assert ctrl.interactive.inspect("a") is None
            assert isinstance(editor(win), ResultPreviewWidget)
            assert current_stage(win) is None
        with ctrl.bus.origin(EventOrigin(kind="agent")):
            agent_draft = ctrl.interactive.begin_sample_pick("a")
        assert isinstance(editor(win), FindPointsWidget)
        assert stage_status(win, "sample") == "draft"
        # Align shows the published preview; the agent's draft stays untouched.
        stage_button(win, "align").click()
        assert isinstance(editor(win), ResultPreviewWidget)
        active = ctrl.interactive.inspect("a")
        assert active is not None and active.context is agent_draft
        assert stage_status(win, "sample") == "draft"
        assert current_stage(win) is None
        # Sample shows the agent's draft again, never a new or discarded one.
        stage_button(win, "sample").click()
        assert isinstance(editor(win), FindPointsWidget)
        again = ctrl.interactive.inspect("a")
        assert again == active
        assert current_stage(win) == "sample"
        assert dialogs.calls == []
    finally:
        ctrl.control_identity.reclaim_local()
        win.close()
        win.deleteLater()
        qapp.processEvents()


def _background(widget) -> tuple[np.ndarray, tuple[float, float]]:
    """Image array and colour limits of the first axes of an editor's figure."""
    image = widget.figure.axes[0].get_images()[0]
    return np.asarray(image.get_array()), image.get_clim()


def test_onetone_line_picker_draws_the_sample_picker_background(
    qapp, onetone_controller
):
    from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget
    from zcu_tools.gui.app.fluxdep.ui.interactive.onetone import OneToneWidget

    ctrl = onetone_controller
    win = MainWindow(ctrl)
    try:
        with ctrl.bus.origin(EventOrigin(kind="agent")):
            ctrl.interactive.begin_sample_pick("one")
        sample = editor(win)
        assert isinstance(sample, OneToneWidget)
        expected, expected_clim = _background(sample)
        ctrl.interactive.cancel("one")
        with ctrl.bus.origin(EventOrigin(kind="agent")):
            ctrl.interactive.begin_line_pick("one")
        lines = editor(win)
        assert isinstance(lines, LinePickerWidget)
        background, clim = _background(lines)
        np.testing.assert_allclose(background, expected)
        assert clim == pytest.approx(expected_clim)
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_twotone_line_picker_keeps_the_phase_projection_of_its_preview(
    qapp, twotone_controller
):
    from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget
    from zcu_tools.gui.app.fluxdep.ui.interactive.result_preview import (
        ResultPreviewWidget,
    )

    ctrl = twotone_controller
    preview = ResultPreviewWidget(ctrl.state.spectrums["two"])
    expected, _ = _background(preview)
    preview.deleteLater()
    win = MainWindow(ctrl)
    try:
        ctrl.interactive.cancel("two")  # the window opened the sample picker
        with ctrl.bus.origin(EventOrigin(kind="agent")):
            ctrl.interactive.begin_line_pick("two")
        lines = editor(win)
        assert isinstance(lines, LinePickerWidget)
        background, _ = _background(lines)
        np.testing.assert_allclose(background, expected)
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()
