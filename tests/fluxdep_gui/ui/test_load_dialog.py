"""Headless tests for LoadSpectrumDialog (preview + transpose + inherit)."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.fluxdep_gui.ui.load_dialog import LoadSpectrumDialog


@pytest.fixture
def dialog(qapp):
    d = LoadSpectrumDialog(loaded_names=["existing.hdf5"])
    yield d
    d.deleteLater()


def test_builds_with_no_file(dialog):
    # no file chosen → OK disabled, no request
    assert dialog.result_request() is None
    if dialog._ok_button is not None:
        assert dialog._ok_button.isEnabled() is False


def test_inherit_dropdown_lists_loaded_names(dialog):
    items = [dialog._inherit.itemText(i) for i in range(dialog._inherit.count())]
    assert items == ["(none)", "existing.hdf5"]
    # default selection is (none) → None
    assert dialog._inherit.currentData() is None


def test_result_request_reflects_choices(dialog):
    # simulate a chosen file + type + transpose + inherit, bypassing the OS dialog
    dialog._filepath = "/tmp/legacy.hdf5"
    dialog._type.setCurrentText("TwoTone")
    dialog._transpose_btn.setChecked(True)  # toggles _transpose via signal
    dialog._inherit.setCurrentText("existing.hdf5")

    req = dialog.result_request()
    assert req is not None
    assert req.filepath == "/tmp/legacy.hdf5"
    assert req.spec_type == "TwoTone"
    assert req.transpose_axes is True
    assert req.inherit_from == "existing.hdf5"


def test_preview_blank_fallback_does_not_crash(dialog):
    # no raw cached → renders the "No preview" fallback without raising
    dialog._raw = None
    dialog._render_preview()  # must not raise


def test_preview_renders_2d_and_transpose_toggles(dialog):
    flux = np.linspace(0.0, 1.0, 6)
    freqs = np.linspace(5.0, 6.0, 4)
    signals = np.ones((6, 4), dtype=np.complex128)
    dialog._raw = (signals, flux, freqs)
    dialog._render_preview()  # normal orientation
    dialog._transpose_btn.setChecked(True)
    dialog._render_preview()  # transposed orientation — must not raise
    assert dialog._transpose is True
