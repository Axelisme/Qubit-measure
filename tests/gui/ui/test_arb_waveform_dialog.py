from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from zcu_tools.gui.app.main.ui.arb_waveform_dialog import (
    ArbWaveformDialog,
    _PreviewCanvas,
)
from zcu_tools.meta_tool import (
    ArbWaveformData,
    ArbWaveformInfo,
    FormulaRecipe,
    render_formula_recipe,
)


class _FakeController:
    def __init__(self) -> None:
        self.assets: dict[str, ArbWaveformData] = {}

    def list_arb_waveform_infos(self) -> list[ArbWaveformInfo]:
        return [
            ArbWaveformInfo(
                data_key=name,
                duration=data.duration,
                sample_count=int(data.time.size),
                has_q=data.has_q,
                peak_abs=data.peak_abs,
                has_recipe=data.recipe is not None,
                mtime=time.time(),
                file_size=0,
                recipe=data.recipe,
            )
            for name, data in sorted(self.assets.items())
        ]

    def list_arb_waveforms(self) -> list[str]:
        return sorted(self.assets)

    def load_arb_waveform_data(self, data_key: str) -> ArbWaveformData:
        return self.assets[data_key]

    def set_arb_waveform(
        self, data_key: str, recipe: Any, *, overwrite: bool = False
    ) -> dict[str, object]:
        if data_key in self.assets and not overwrite:
            raise RuntimeError("exists")
        self.assets[data_key] = render_formula_recipe(recipe)
        return {"success": True, "status": "overwritten" if overwrite else "created"}

    def rename_arb_waveform(self, old_data_key: str, new_data_key: str) -> None:
        self.assets[new_data_key] = self.assets.pop(old_data_key)

    def delete_arb_waveform(self, data_key: str) -> None:
        del self.assets[data_key]


@pytest.fixture(autouse=True)
def _qt(qapp):  # noqa: ARG001
    yield


# ---------------------------------------------------------------------------
# Existing tests (updated for debounce state machine)
# ---------------------------------------------------------------------------


def test_dialog_validates_segments_and_saves_preview(  # noqa: ARG001
    qapp, monkeypatch
) -> None:
    from qtpy.QtCore import Qt  # type: ignore[attr-defined]
    from qtpy.QtTest import QTest  # type: ignore[attr-defined]
    from qtpy.QtWidgets import (  # type: ignore[attr-defined]
        QHeaderView,
        QLabel,
        QPushButton,
    )

    ctrl = _FakeController()
    dlg = ArbWaveformDialog(ctrl)  # type: ignore[arg-type]

    assert dlg._data_key_edit.text() == "arb_data1"
    assert dlg._segment_table.rowCount() == 3
    default_formula_item = dlg._segment_table.item(0, 1)
    assert default_formula_item is not None
    assert default_formula_item.text().startswith("exp(")
    assert dlg._save_btn.isEnabled()
    assert dlg._warning.isHidden()
    button_texts = {button.text() for button in dlg.findChildren(QPushButton)}
    assert "New" not in button_texts
    assert "Create ML" not in button_texts
    assert "Preview" not in button_texts
    label_texts = [label.text() for label in dlg.findChildren(QLabel)]
    assert any("<b>I</b>=imaginary unit" in text for text in label_texts)
    assert dlg._normalize_check.isChecked()

    dlg._segment_table.selectRow(0)
    dlg._insert_after_btn.click()
    assert dlg._segment_table.rowCount() == 4
    dlg._delete_segment_btn.click()
    assert dlg._segment_table.rowCount() == 3

    formula_item = dlg._segment_table.item(0, 1)
    assert formula_item is not None
    # "unknown_symbol" has valid structure (non-empty string) but fails render.
    # With the new debounce design, render error surfaces only after the timer fires.
    formula_item.setText("unknown_symbol")
    # Wait for debounce to fire (250 ms + slack).
    QTest.qWait(300)
    assert not dlg._save_btn.isEnabled()
    assert not dlg._warning.isHidden()
    assert "unsupported" in dlg._warning.text()

    # Fixing the formula triggers cheap validation immediately (no qWait needed).
    formula_item.setText("sin(2*pi*t)")
    assert dlg._save_btn.isEnabled()
    assert dlg._warning.isHidden()
    QTest.qWait(300)
    assert dlg._preview._i_line is not None
    label = dlg._preview._i_line.get_label()
    assert isinstance(label, str)
    assert label.startswith("I:")

    dlg._normalize_check.setChecked(False)
    dlg._save_recipe()
    assert "arb_data1" in ctrl.assets
    recipe = ctrl.assets["arb_data1"].recipe
    assert recipe is not None
    assert recipe.normalize == "none"
    duration_item = dlg._asset_table.item(0, 1)
    assert duration_item is not None
    assert duration_item.text() == "1.00"

    header = dlg._asset_table.horizontalHeader()
    assert dlg._asset_table.horizontalScrollBarPolicy() == (
        Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.ResizeToContents
    assert header.sectionResizeMode(2) == QHeaderView.ResizeMode.Stretch
    dlg.refresh()
    assert dlg._asset_table.horizontalScrollBarPolicy() == (
        Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.ResizeToContents
    assert header.sectionResizeMode(2) == QHeaderView.ResizeMode.Stretch

    monkeypatch.setattr(
        "zcu_tools.gui.app.main.ui.arb_waveform_dialog.QInputDialog.getText",
        lambda *args, **kwargs: ("renamed_data", True),
    )
    dlg._rename_btn.click()
    assert "renamed_data" in ctrl.assets
    assert "arb_data1" not in ctrl.assets


def test_preview_canvas_autoscales_y_axis_for_normalized_and_raw(qapp) -> None:  # noqa: ARG001
    data = render_formula_recipe(
        {
            "segments": [{"duration": 0.002, "formula": "0.25 + 0.5*I"}],
            "normalize": "none",
        }
    )
    canvas = _PreviewCanvas()

    canvas.plot("subunit", data, normalize=True)

    assert canvas._i_line is not None
    assert canvas._q_line is not None
    assert canvas._ax.get_ylabel() == "Normalized amplitude"
    assert np.asarray(canvas._i_line.get_ydata(), dtype=float)[0] == pytest.approx(
        0.25 / data.peak_abs
    )
    assert np.asarray(canvas._q_line.get_ydata(), dtype=float)[0] == pytest.approx(
        0.5 / data.peak_abs
    )
    normalized_y_min, normalized_y_max = canvas._ax.get_ylim()
    assert -0.1 < normalized_y_min < 0.0
    assert 1.0 < normalized_y_max < 1.1

    canvas.plot("subunit", data, normalize=False)

    assert canvas._i_line is not None
    assert canvas._q_line is not None
    assert canvas._ax.get_ylabel() == "Amplitude"
    assert np.asarray(canvas._i_line.get_ydata(), dtype=float)[0] == pytest.approx(0.25)
    assert np.asarray(canvas._q_line.get_ydata(), dtype=float)[0] == pytest.approx(0.5)
    raw_y_min, raw_y_max = canvas._ax.get_ylim()
    assert -0.1 < raw_y_min < 0.0
    assert 0.5 < raw_y_max < 1.0
    assert raw_y_max < normalized_y_max


# ---------------------------------------------------------------------------
# A. Debounce state machine tests
# ---------------------------------------------------------------------------


def test_A1_cheap_structure_error_is_immediate(qapp) -> None:  # noqa: ARG001
    """Structural errors (empty formula, non-numeric duration) disable Save synchronously."""
    from qtpy.QtTest import QTest  # type: ignore[attr-defined]

    ctrl = _FakeController()
    dlg = ArbWaveformDialog(ctrl)  # type: ignore[arg-type]

    # Wait for initial debounce to settle so we start from a known-good state.
    QTest.qWait(300)
    assert dlg._save_btn.isEnabled()

    duration_item = dlg._segment_table.item(0, 0)
    assert duration_item is not None
    duration_item.setText("abc")  # non-numeric duration → structure error

    # No qWait: cheap validation fires synchronously.
    assert not dlg._save_btn.isEnabled()
    assert not dlg._warning.isHidden()
    assert dlg._structure_error is not None


def test_A2_deep_render_error_surfaces_after_debounce(qapp) -> None:  # noqa: ARG001
    """Formula that passes structure but fails render → error only after debounce timer."""
    from qtpy.QtTest import QTest  # type: ignore[attr-defined]

    ctrl = _FakeController()
    dlg = ArbWaveformDialog(ctrl)  # type: ignore[arg-type]

    # Settle initial render.
    QTest.qWait(300)

    formula_item = dlg._segment_table.item(0, 1)
    assert formula_item is not None
    formula_item.setText("unknown_symbol")

    # Immediately after edit: structure is valid, render_error cleared → Save enabled.
    assert dlg._save_btn.isEnabled()
    assert dlg._warning.isHidden()
    assert dlg._render_error is None

    # After debounce: render ran and found the unknown symbol.
    QTest.qWait(300)
    assert not dlg._save_btn.isEnabled()
    assert not dlg._warning.isHidden()
    assert dlg._render_error is not None
    assert "unsupported" in dlg._render_error


def test_A3_data_key_change_does_not_trigger_render(qapp, monkeypatch) -> None:  # noqa: ARG001
    """Editing data_key must never call render_formula_recipe."""
    from qtpy.QtTest import QTest  # type: ignore[attr-defined]

    ctrl = _FakeController()
    render_call_count = 0
    original_render = render_formula_recipe

    def counting_render(recipe: Any) -> ArbWaveformData:
        nonlocal render_call_count
        render_call_count += 1
        return original_render(recipe)

    monkeypatch.setattr(
        "zcu_tools.gui.app.main.ui.arb_waveform_dialog.render_formula_recipe",
        counting_render,
    )
    dlg = ArbWaveformDialog(ctrl)  # type: ignore[arg-type]
    QTest.qWait(300)  # let initial render fire
    count_after_init = render_call_count

    # Type into data_key field and wait past debounce window.
    dlg._data_key_edit.setText("some_valid_key")
    QTest.qWait(300)

    assert render_call_count == count_after_init, (
        "data_key edit must not trigger render_formula_recipe"
    )

    # Invalid data_key → Save disabled synchronously.
    dlg._data_key_edit.setText("1invalid")
    assert not dlg._save_btn.isEnabled()

    # Fix data_key → Save re-enabled synchronously (no render needed).
    dlg._data_key_edit.setText("valid_key")
    assert dlg._save_btn.isEnabled()
    assert render_call_count == count_after_init


def test_A4_save_uses_valid_recipe_without_waiting_for_debounce(qapp) -> None:  # noqa: ARG001
    """Save uses _valid_recipe set by cheap path; service renders once internally."""
    from qtpy.QtTest import QTest  # type: ignore[attr-defined]

    ctrl = _FakeController()
    dlg = ArbWaveformDialog(ctrl)  # type: ignore[arg-type]
    QTest.qWait(300)  # initial render

    # Modify formula; do NOT wait for debounce.
    formula_item = dlg._segment_table.item(0, 1)
    assert formula_item is not None
    formula_item.setText("sin(2*pi*t)")
    # cheap path set _valid_recipe immediately; Save is enabled.
    assert dlg._save_btn.isEnabled()
    assert dlg._valid_recipe is not None

    dlg._save_recipe()

    assert "arb_data1" in ctrl.assets
    stored_recipe = ctrl.assets["arb_data1"].recipe
    assert stored_recipe is not None
    assert stored_recipe.segments[0].formula == "sin(2*pi*t)"


def test_A5_render_failure_leaves_no_half_baked_state(qapp) -> None:  # noqa: ARG001
    """After a render failure: _valid_data is None, Save disabled, preview unchanged."""
    from qtpy.QtTest import QTest  # type: ignore[attr-defined]

    ctrl = _FakeController()
    dlg = ArbWaveformDialog(ctrl)  # type: ignore[arg-type]
    QTest.qWait(300)
    assert dlg._valid_data is not None  # initial render succeeded

    formula_item = dlg._segment_table.item(0, 1)
    assert formula_item is not None
    formula_item.setText("unknown_symbol")
    QTest.qWait(300)

    assert dlg._valid_recipe is not None  # cheap path: structure valid
    assert dlg._valid_data is None  # deep path: render failed, no half-baked data
    assert not dlg._save_btn.isEnabled()
    assert not dlg._warning.isHidden()

    # Recovery: fix formula.
    formula_item.setText("cos(2*pi*t)")
    QTest.qWait(300)

    assert dlg._valid_data is not None
    assert dlg._save_btn.isEnabled()
    assert dlg._warning.isHidden()
    assert dlg._preview._i_line is not None


def test_A6_save_with_deep_invalid_formula_reports_save_failed(
    qapp, monkeypatch
) -> None:  # noqa: ARG001
    """Save in the optimistic window with a render-failing formula shows 'Save failed'."""
    from qtpy.QtTest import QTest  # type: ignore[attr-defined]

    ctrl = _FakeController()
    dlg = ArbWaveformDialog(ctrl)  # type: ignore[arg-type]
    QTest.qWait(300)

    formula_item = dlg._segment_table.item(0, 1)
    assert formula_item is not None
    formula_item.setText("unknown_symbol")
    # Do NOT wait for debounce — we're in the optimistic Save-enabled window.
    assert dlg._save_btn.isEnabled()

    critical_calls: list[tuple[str, str]] = []

    def fake_critical(parent: Any, title: str, msg: str) -> None:
        critical_calls.append((title, msg))

    monkeypatch.setattr(
        "zcu_tools.gui.app.main.ui.arb_waveform_dialog.QMessageBox.critical",
        fake_critical,
    )
    dlg._save_recipe()

    # Service attempted render → failed → "Save failed" box shown.
    assert len(critical_calls) == 1
    assert critical_calls[0][0] == "Save failed"
    # No asset was stored.
    assert "arb_data1" not in ctrl.assets


# ---------------------------------------------------------------------------
# B. save/load split try
# ---------------------------------------------------------------------------


def test_B7_reload_failure_reports_reload_failed_not_save_failed(
    qapp, monkeypatch
) -> None:  # noqa: ARG001
    """When save succeeds but reload fails, the user sees 'Reload failed', not 'Save failed'.

    The patched load fails only on the first call so that the subsequent
    _on_asset_selection_changed triggered by refresh() can load normally;
    _on_asset_selection_changed only catches ArbWaveformError, not RuntimeError.
    """
    from qtpy.QtTest import QTest  # type: ignore[attr-defined]

    ctrl = _FakeController()
    dlg = ArbWaveformDialog(ctrl)  # type: ignore[arg-type]
    QTest.qWait(300)

    # Fail only the first call to load_arb_waveform_data after save (simulates a
    # transient disk error exactly at reload time, not at subsequent selection loads).
    original_load = ctrl.load_arb_waveform_data
    fail_next: list[bool] = [True]  # mutable cell; no nonlocal needed

    def failing_load_once(data_key: str) -> ArbWaveformData:
        if data_key in ctrl.assets and fail_next[0]:
            fail_next[0] = False
            raise RuntimeError("disk read failure")
        return original_load(data_key)

    monkeypatch.setattr(ctrl, "load_arb_waveform_data", failing_load_once)

    critical_calls: list[tuple[str, str]] = []

    def fake_critical(parent: Any, title: str, msg: str) -> None:
        critical_calls.append((title, msg))

    monkeypatch.setattr(
        "zcu_tools.gui.app.main.ui.arb_waveform_dialog.QMessageBox.critical",
        fake_critical,
    )

    dlg._save_recipe()

    # Asset was saved successfully.
    assert "arb_data1" in ctrl.assets
    assert dlg._current_data_key == "arb_data1"
    # Error reported as "Reload failed", not "Save failed".
    assert len(critical_calls) == 1
    assert critical_calls[0][0] == "Reload failed"


# ---------------------------------------------------------------------------
# E. Typed recipe
# ---------------------------------------------------------------------------


def test_E12_recipe_from_ui_returns_formula_recipe(qapp) -> None:  # noqa: ARG001
    """_recipe_from_ui always returns a typed FormulaRecipe, never a plain dict."""
    ctrl = _FakeController()
    dlg = ArbWaveformDialog(ctrl)  # type: ignore[arg-type]

    result = dlg._recipe_from_ui()
    assert isinstance(result, FormulaRecipe)
    assert len(result.segments) == 3
    assert result.normalize in ("peak", "none")
