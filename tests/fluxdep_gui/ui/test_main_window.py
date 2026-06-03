"""Headless tests for the fluxdep-gui MainWindow shell.

Drives the window through the Controller (not the user dialogs, which would
block), and asserts the spectrum list reflects State.
"""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.fluxdep_gui.controller import Controller
from zcu_tools.fluxdep_gui.state import FluxDepState
from zcu_tools.fluxdep_gui.ui.main_window import MainWindow


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


def test_window_builds_empty(window):
    assert window._list.count() == 0
    assert window.windowTitle() == "fluxdep-gui"


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
    from zcu_tools.fluxdep_gui.ui.interactive.line_picker import LinePickerWidget

    filepath, *_ = spectrum_hdf5
    name = window._ctrl.load_spectrum(filepath, spec_type="OneTone")
    window._ctrl.set_active_spectrum(name)
    assert isinstance(window._current_editor, LinePickerWidget)


def test_aligned_onetone_shows_onetone_widget(window, spectrum_hdf5):
    from zcu_tools.fluxdep_gui.ui.interactive.onetone import OneToneWidget

    filepath, *_ = spectrum_hdf5
    name = window._ctrl.load_spectrum(filepath, spec_type="OneTone")
    window._ctrl.set_active_spectrum(name)
    window._ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    assert isinstance(window._current_editor, OneToneWidget)


def test_aligned_twotone_shows_find_points_widget(window, spectrum_hdf5):
    from zcu_tools.fluxdep_gui.ui.interactive.find_points import FindPointsWidget

    filepath, *_ = spectrum_hdf5
    name = window._ctrl.load_spectrum(filepath, spec_type="TwoTone")
    window._ctrl.set_active_spectrum(name)
    window._ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    assert isinstance(window._current_editor, FindPointsWidget)


def test_selected_returns_to_placeholder(window, spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    name = window._ctrl.load_spectrum(filepath, spec_type="OneTone")
    window._ctrl.set_active_spectrum(name)
    window._ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    window._ctrl.set_points(name, np.array([0.0, 1.0]), np.array([5.0, 5.1]))
    assert window._current_editor is None  # back to placeholder
    assert window._editor_stack.currentWidget() is window._placeholder
