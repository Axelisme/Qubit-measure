from __future__ import annotations

import time
from pathlib import Path
from typing import Literal, Never
from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from zcu_tools.gui.session.services import arb_waveform
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.arb_waveform_dialog import ArbWaveformDialog
from zcu_tools.gui.widgets import error_colour
from zcu_tools.resources.waveform_assets import (
    ArbWaveformData,
    ArbWaveformInfo,
    FormulaRecipe,
    render_formula_recipe,
)

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver

from ._waveform_timer import PreviewClock


class _FakeController:
    def __init__(self) -> None:
        self.assets: dict[str, ArbWaveformData] = {}

    def list_infos(self) -> list[ArbWaveformInfo]:
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

    def list_data_keys(self) -> list[str]:
        return sorted(self.assets)

    def load_data(self, data_key: str) -> ArbWaveformData:
        return self.assets[data_key]

    def set_formula(
        self,
        data_key: str,
        recipe: FormulaRecipe | dict[str, object],
        *,
        overwrite: bool = False,
    ) -> Literal["created", "overwritten"]:
        if data_key in self.assets and not overwrite:
            raise RuntimeError("exists")
        self.assets[data_key] = render_formula_recipe(recipe)
        return "overwritten" if overwrite else "created"

    def get_preview(self, data_key: str) -> Never:
        raise AssertionError("The dialog must render locally, not request a remote PNG")

    def rename(self, old_data_key: str, new_data_key: str) -> None:
        self.assets[new_data_key] = self.assets.pop(old_data_key)

    def delete(self, data_key: str) -> None:
        del self.assets[data_key]


@pytest.fixture(autouse=True, name="_qt")
def fixture_qt(qapp):
    """Keep the QApplication available throughout each test."""
    yield


@pytest.fixture
def preview_clock(monkeypatch: pytest.MonkeyPatch):
    clock = PreviewClock()
    monkeypatch.setattr(
        "zcu_tools.gui.session.ui.arb_waveform_dialog.QTimer", clock.timer
    )
    try:
        yield clock
    finally:
        clock.close()


class _Editor:
    """Locate controls through the dialog's public Qt widget tree."""

    def __init__(self, dialog: ArbWaveformDialog) -> None:
        from qtpy.QtWidgets import QCheckBox, QGroupBox, QLabel, QLineEdit, QTableWidget

        self.dialog = dialog
        group = next(
            g for g in dialog.findChildren(QGroupBox) if g.title() == "Formula Recipe"
        )
        self.key = next(iter(group.findChildren(QLineEdit)))
        self.normalize = next(iter(group.findChildren(QCheckBox)))
        self.segments = next(iter(group.findChildren(QTableWidget)))
        self.warning = next(
            label for label in group.findChildren(QLabel) if not label.text()
        )
        self.assets = next(
            t for t in dialog.findChildren(QTableWidget) if t.columnCount() == 3
        )
        self.save = self.button("Save")

    def button(self, text: str):
        from qtpy.QtWidgets import QPushButton

        return next(
            b for b in self.dialog.findChildren(QPushButton) if b.text() == text
        )


def _preview_axes(dialog: ArbWaveformDialog) -> Axes:
    canvas = dialog.findChild(FigureCanvasQTAgg)
    assert canvas is not None
    return canvas.figure.axes[0]


def _assert_formula_error_recovery(
    dlg: ArbWaveformDialog, editor: _Editor, preview_clock: PreviewClock
) -> None:
    """Verify formula errors recover Save availability and the rendered preview."""
    formula_item = editor.segments.item(0, 1)
    assert formula_item is not None
    # "unknown_symbol" has valid structure (non-empty string) but fails render.
    # With the new debounce design, render error surfaces only after the timer fires.
    formula_item.setText("unknown_symbol")
    preview_clock.drain()
    assert not editor.save.isEnabled()
    assert not editor.warning.isHidden()
    assert "unsupported" in editor.warning.text()

    # Fixing the formula triggers cheap validation immediately (no qWait needed).
    formula_item.setText("sin(2*pi*t)")
    assert editor.save.isEnabled()
    assert editor.warning.isHidden()
    preview_clock.drain()
    label = _preview_axes(dlg).lines[0].get_label()
    assert isinstance(label, str)
    assert label.startswith("I:")


# ---------------------------------------------------------------------------
# Existing tests (updated for debounce state machine)
# ---------------------------------------------------------------------------


def test_dialog_validates_segments_and_saves_preview(
    qapp, monkeypatch, preview_clock: PreviewClock
) -> None:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QHeaderView,
        QLabel,
        QPushButton,
    )

    ctrl = _FakeController()
    dlg = ArbWaveformDialog(ctrl)
    editor = _Editor(dlg)

    assert editor.key.text() == "arb_data1"
    assert editor.segments.rowCount() == 3
    default_formula_item = editor.segments.item(0, 1)
    assert default_formula_item is not None
    assert default_formula_item.text().startswith("exp(")
    assert editor.save.isEnabled()
    assert editor.warning.isHidden()
    button_texts = {button.text() for button in dlg.findChildren(QPushButton)}
    assert "New" not in button_texts
    assert "Create ML" not in button_texts
    assert "Preview" not in button_texts
    label_texts = [label.text() for label in dlg.findChildren(QLabel)]
    assert any("<b>I</b>=imaginary unit" in text for text in label_texts)
    assert editor.normalize.isChecked()

    editor.segments.selectRow(0)
    editor.button("Insert After").click()
    assert editor.segments.rowCount() == 4
    editor.button("Delete Segment").click()
    assert editor.segments.rowCount() == 3

    _assert_formula_error_recovery(dlg, editor, preview_clock)

    editor.normalize.setChecked(False)
    editor.save.click()
    assert "arb_data1" in ctrl.assets
    recipe = ctrl.assets["arb_data1"].recipe
    assert recipe is not None
    assert recipe.normalize == "none"
    duration_item = editor.assets.item(0, 1)
    assert duration_item is not None
    assert duration_item.text() == "1.00"

    header = editor.assets.horizontalHeader()
    assert header is not None
    assert editor.assets.horizontalScrollBarPolicy() == (
        Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.ResizeToContents
    assert header.sectionResizeMode(2) == QHeaderView.ResizeMode.Stretch
    dlg.refresh()
    assert editor.assets.horizontalScrollBarPolicy() == (
        Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.ResizeToContents
    assert header.sectionResizeMode(2) == QHeaderView.ResizeMode.Stretch

    monkeypatch.setattr(
        "zcu_tools.gui.session.ui.arb_waveform_dialog.QInputDialog.getText",
        lambda *args, **kwargs: ("renamed_data", True),
    )
    editor.button("Rename").click()
    assert "renamed_data" in ctrl.assets
    assert "arb_data1" not in ctrl.assets


def test_preview_canvas_autoscales_y_axis_for_normalized_and_raw(
    qapp, preview_clock: PreviewClock
) -> None:
    data = render_formula_recipe(
        {
            "segments": [{"duration": 0.002, "formula": "0.25 + 0.5*I"}],
            "normalize": "none",
        }
    )
    ctrl = _FakeController()
    ctrl.assets["subunit"] = data
    dlg = ArbWaveformDialog(ctrl)
    try:
        editor = _Editor(dlg)
        editor.assets.selectRow(0)
        editor.normalize.setChecked(True)
        preview_clock.drain()

        axes = _preview_axes(dlg)
        i_line = next(
            (
                line
                for line in axes.lines
                if isinstance(label := line.get_label(), str) and label.startswith("I:")
            ),
            None,
        )
        q_line = next(
            (
                line
                for line in axes.lines
                if isinstance(label := line.get_label(), str) and label.startswith("Q:")
            ),
            None,
        )
        assert i_line is not None
        assert q_line is not None
        assert axes.get_ylabel() == "Normalized amplitude"
        assert np.asarray(i_line.get_ydata(), dtype=float)[0] == pytest.approx(
            0.25 / data.peak_abs
        )
        assert np.asarray(q_line.get_ydata(), dtype=float)[0] == pytest.approx(
            0.5 / data.peak_abs
        )
        normalized_y_min, normalized_y_max = axes.get_ylim()
        assert -0.1 < normalized_y_min < 0.0
        assert 1.0 < normalized_y_max < 1.1

        editor.normalize.setChecked(False)
        preview_clock.drain()

        axes = _preview_axes(dlg)
        i_line = next(
            (
                line
                for line in axes.lines
                if isinstance(label := line.get_label(), str) and label.startswith("I:")
            ),
            None,
        )
        q_line = next(
            (
                line
                for line in axes.lines
                if isinstance(label := line.get_label(), str) and label.startswith("Q:")
            ),
            None,
        )
        assert i_line is not None
        assert q_line is not None
        assert axes.get_ylabel() == "Amplitude"
        assert np.asarray(i_line.get_ydata(), dtype=float)[0] == pytest.approx(0.25)
        assert np.asarray(q_line.get_ydata(), dtype=float)[0] == pytest.approx(0.5)
        raw_y_min, raw_y_max = axes.get_ylim()
        assert -0.1 < raw_y_min < 0.0
        assert 0.5 < raw_y_max < 1.0
        assert raw_y_max < normalized_y_max
    finally:
        dlg.close()
        dlg.deleteLater()


# ---------------------------------------------------------------------------
# A. Debounce state machine tests
# ---------------------------------------------------------------------------


def test_A1_cheap_structure_error_is_immediate(
    qapp, preview_clock: PreviewClock
) -> None:
    """Structural errors disable Save synchronously and cancel pending rendering."""
    ctrl = _FakeController()
    dlg = ArbWaveformDialog(ctrl)
    editor = _Editor(dlg)
    preview_clock.drain()
    initial_lines = tuple(_preview_axes(dlg).lines)
    assert initial_lines
    assert editor.save.isEnabled()
    deliveries = preview_clock.deliveries

    duration_item = editor.segments.item(0, 0)
    assert duration_item is not None
    # A valid edit arranges a render which the next structural error must cancel.
    duration_item.setText("0.3")
    duration_item.setText("abc")
    assert not editor.save.isEnabled()
    assert not editor.warning.isHidden()
    assert "duration" in editor.warning.text()
    preview_clock.drain()
    assert preview_clock.deliveries == deliveries
    assert tuple(_preview_axes(dlg).lines) == initial_lines


def test_A2_deep_render_error_surfaces_after_debounce(
    qapp, preview_clock: PreviewClock
) -> None:
    """Deep errors appear only on timeout; a burst renders only its final recipe."""
    ctrl = _FakeController()
    dlg = ArbWaveformDialog(ctrl)
    editor = _Editor(dlg)
    preview_clock.drain()
    initial_lines = tuple(_preview_axes(dlg).lines)
    assert initial_lines
    deliveries = preview_clock.deliveries

    formula_item = editor.segments.item(0, 1)
    assert formula_item is not None
    formula_item.setText("sin(2*pi*t)")
    # These are virtual milliseconds, not a wall-clock performance assertion.
    preview_clock.advance(200)
    formula_item.setText("unknown_symbol")
    preview_clock.advance(50)  # The first deadline must have been replaced.
    assert preview_clock.deliveries == deliveries
    assert editor.save.isEnabled()
    assert editor.warning.isHidden()
    assert tuple(_preview_axes(dlg).lines) == initial_lines

    preview_clock.drain()
    assert preview_clock.deliveries == deliveries + 1
    assert not editor.save.isEnabled()
    assert not editor.warning.isHidden()
    assert "unsupported" in editor.warning.text()
    assert tuple(_preview_axes(dlg).lines) == initial_lines


def test_A3_data_key_change_does_not_trigger_render(
    qapp, preview_clock: PreviewClock
) -> None:
    """Key edits neither schedule timeouts nor replace the rendered artists."""
    ctrl = _FakeController()
    dlg = ArbWaveformDialog(ctrl)
    editor = _Editor(dlg)
    preview_clock.drain()
    initial_lines = tuple(_preview_axes(dlg).lines)
    assert initial_lines
    deliveries = preview_clock.deliveries

    editor.key.setText("some_valid_key")
    preview_clock.drain()
    assert preview_clock.deliveries == deliveries
    assert tuple(_preview_axes(dlg).lines) == initial_lines

    editor.key.setText("1invalid")
    assert not editor.save.isEnabled()
    editor.key.setText("valid_key")
    assert editor.save.isEnabled()
    preview_clock.drain()
    assert preview_clock.deliveries == deliveries
    assert tuple(_preview_axes(dlg).lines) == initial_lines


def test_A4_save_uses_valid_recipe_without_waiting_for_debounce(
    qapp, preview_clock: PreviewClock
) -> None:
    """Save persists the latest edits before delivery and cancels pending preview work."""
    ctrl = _FakeController()
    dlg = ArbWaveformDialog(ctrl)
    editor = _Editor(dlg)
    preview_clock.drain()
    deliveries = preview_clock.deliveries

    formula_item = editor.segments.item(0, 1)
    assert formula_item is not None
    formula_item.setText("sin(2*pi*t)")
    assert editor.save.isEnabled()
    editor.save.click()

    assert "arb_data1" in ctrl.assets
    stored_recipe = ctrl.assets["arb_data1"].recipe
    assert stored_recipe is not None
    assert stored_recipe.segments[0].formula == "sin(2*pi*t)"
    saved_lines = tuple(_preview_axes(dlg).lines)
    assert saved_lines
    preview_clock.drain()
    assert preview_clock.deliveries == deliveries
    assert tuple(_preview_axes(dlg).lines) == saved_lines
    saved_data = ctrl.assets["arb_data1"]
    np.testing.assert_allclose(
        np.asarray(saved_lines[0].get_ydata(), dtype=float),
        saved_data.idata / saved_data.peak_abs,
    )


def test_A5_render_failure_leaves_no_half_baked_state(
    qapp, preview_clock: PreviewClock
) -> None:
    """Failed rendering retains the last good plot; recovery uses the final edits."""
    ctrl = _FakeController()
    dlg = ArbWaveformDialog(ctrl)
    editor = _Editor(dlg)
    preview_clock.drain()
    initial_lines = tuple(_preview_axes(dlg).lines)
    assert initial_lines

    formula_item = editor.segments.item(0, 1)
    assert formula_item is not None
    formula_item.setText("unknown_symbol")
    preview_clock.drain()
    assert tuple(_preview_axes(dlg).lines) == initial_lines
    assert not editor.save.isEnabled()
    assert not editor.warning.isHidden()

    formula_item.setText("cos(2*pi*t)")
    preview_clock.drain()
    assert editor.save.isEnabled()
    assert editor.warning.isHidden()
    recovered_lines = tuple(_preview_axes(dlg).lines)
    assert recovered_lines != initial_lines
    # Save independently renders the current recipe through the asset collaborator.
    editor.save.click()
    saved_data = ctrl.assets["arb_data1"]
    assert saved_data.recipe is not None
    assert saved_data.recipe.segments[0].formula == "cos(2*pi*t)"
    np.testing.assert_allclose(
        np.asarray(recovered_lines[0].get_ydata(), dtype=float),
        saved_data.idata / saved_data.peak_abs,
    )


def test_A6_save_with_deep_invalid_formula_reports_save_failed(
    qapp, monkeypatch, preview_clock: PreviewClock
) -> None:
    """Save in the optimistic window with a render-failing formula shows 'Save failed'."""
    ctrl = _FakeController()
    dialogs = RecordingDialogPresenter()
    dlg = ArbWaveformDialog(ctrl, dialog_presenter=dialogs)
    editor = _Editor(dlg)
    preview_clock.drain()

    formula_item = editor.segments.item(0, 1)
    assert formula_item is not None
    formula_item.setText("unknown_symbol")
    # Do NOT wait for debounce — we're in the optimistic Save-enabled window.
    assert editor.save.isEnabled()

    editor.save.click()

    # Service attempted render → failed → "Save failed" box shown.
    assert len(dialogs.calls) == 1
    assert dialogs.calls[0].kind == "critical"
    assert dialogs.calls[0].title == "Save failed"
    # No asset was stored.
    assert "arb_data1" not in ctrl.assets


# ---------------------------------------------------------------------------
# B. save/load split try
# ---------------------------------------------------------------------------


def test_B7_reload_failure_reports_reload_failed_not_save_failed(
    qapp, monkeypatch, preview_clock: PreviewClock
) -> None:
    """When save succeeds but reload fails, the user sees 'Reload failed', not 'Save failed'.

    The patched load fails only on the first call so that the subsequent
    _on_asset_selection_changed triggered by refresh() can load normally;
    _on_asset_selection_changed only catches ArbWaveformError, not RuntimeError.
    """
    ctrl = _FakeController()
    dialogs = RecordingDialogPresenter()
    dlg = ArbWaveformDialog(ctrl, dialog_presenter=dialogs)
    editor = _Editor(dlg)
    preview_clock.drain()

    # Fail only the first call to load_data after save (simulates a
    # transient disk error exactly at reload time, not at subsequent selection loads).
    original_load = ctrl.load_data
    fail_next: list[bool] = [True]  # mutable cell; no nonlocal needed

    def failing_load_once(data_key: str) -> ArbWaveformData:
        if data_key in ctrl.assets and fail_next[0]:
            fail_next[0] = False
            raise RuntimeError("disk read failure")
        return original_load(data_key)

    monkeypatch.setattr(ctrl, "load_data", failing_load_once)

    editor.save.click()

    # Asset was saved successfully.
    assert "arb_data1" in ctrl.assets
    assert editor.assets.currentRow() >= 0
    selected = editor.assets.item(editor.assets.currentRow(), 0)
    assert selected is not None
    assert selected.text() == "arb_data1"
    # Error reported as "Reload failed", not "Save failed".
    assert len(dialogs.calls) == 1
    assert dialogs.calls[0].kind == "critical"
    assert dialogs.calls[0].title == "Reload failed"


def test_asset_load_failure_keeps_editor_state(
    qapp, monkeypatch, preview_clock: PreviewClock
) -> None:
    """Failed selection keeps the draft and cannot silently authorize overwrite."""
    from qtpy.QtWidgets import QMessageBox
    from zcu_tools.resources.waveform_assets import ArbWaveformError

    ctrl = _FakeController()
    ctrl.set_formula(
        "broken",
        {"segments": [{"duration": 0.002, "formula": "0.25"}], "normalize": "none"},
    )
    original_data = ctrl.assets["broken"]
    save_observer = MagicMock(wraps=ctrl.set_formula)
    monkeypatch.setattr(ctrl, "set_formula", save_observer)
    questions = MagicMock(return_value=QMessageBox.StandardButton.No)
    dialogs = RecordingDialogPresenter()
    monkeypatch.setattr(QMessageBox, "question", questions)
    dlg = ArbWaveformDialog(ctrl, dialog_presenter=dialogs)
    editor = _Editor(dlg)
    preview_clock.drain()
    draft_key = editor.key.text()

    # The real table lists the asset, but the public load boundary fails.
    load_calls: list[str] = []

    def _raise_load(data_key: str) -> ArbWaveformData:
        load_calls.append(data_key)
        raise ArbWaveformError("corrupt asset", reason="load_failed")

    monkeypatch.setattr(ctrl, "load_data", _raise_load)
    assert editor.assets.rowCount() == 1
    asset_item = editor.assets.item(0, 0)
    assert asset_item is not None
    assert asset_item.text() == "broken"
    editor.assets.selectRow(0)

    assert load_calls == ["broken"]
    assert editor.key.text() == draft_key
    # Warning is shown (the dialog is never .show()n in tests, so check the
    # explicit hidden flag rather than isVisible(), which needs a shown ancestry).
    assert not editor.warning.isHidden()
    assert "corrupt asset" in editor.warning.text()

    # Remembering the failed selection as the current asset would skip this
    # confirmation. Check the warning first: editing the key revalidates it.
    editor.key.setText("broken")
    editor.save.click()
    questions.assert_called_once_with(
        dlg,
        "Overwrite waveform",
        "Overwrite existing arbitrary waveform 'broken'?",
    )
    save_observer.assert_not_called()
    assert ctrl.assets["broken"] is original_data
    assert load_calls == ["broken"]
    dialogs.assert_no_unexpected_messages()


# ---------------------------------------------------------------------------
# E. Typed recipe
# ---------------------------------------------------------------------------


def test_E12_recipe_from_ui_returns_formula_recipe(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Save delivers a typed FormulaRecipe, never a plain dict."""
    ctrl = _FakeController()
    observer = MagicMock(wraps=ctrl.set_formula)
    monkeypatch.setattr(ctrl, "set_formula", observer)
    dlg = ArbWaveformDialog(ctrl)
    editor = _Editor(dlg)

    assert editor.save.isEnabled()
    editor.save.click()
    observer.assert_called_once()
    assert observer.call_args is not None
    result = observer.call_args.args[1]
    assert isinstance(result, FormulaRecipe)
    assert len(result.segments) == 3
    assert result.normalize in ("peak", "none")


def test_save_button_persists_without_remote_png(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qtpy.QtWidgets import QPushButton

    state = SessionState(
        SessionEnv(
            parameters=None,
            soc=None,
            soccfg=None,
            database_path=str(tmp_path),
        )
    )
    assets = arb_waveform.ArbWaveformService(state)
    dialogs = RecordingDialogPresenter()

    def unavailable(*_args: object, **_kwargs: object) -> str:
        raise OSError("PNG export unavailable")

    monkeypatch.setattr(arb_waveform, "render_preview_png", unavailable)
    dialog = ArbWaveformDialog(assets, dialog_presenter=dialogs)
    try:
        save = next(
            button
            for button in dialog.findChildren(QPushButton)
            if button.text() == "Save"
        )
        assert save.isEnabled()
        save.click()
        assert assets.list_data_keys() == ["arb_data1"]
        assert assets.load_data("arb_data1").recipe is not None
        assert state.version.get("arb_waveforms") == 1
        dialogs.assert_no_unexpected_messages()
    finally:
        dialog.close()


def test_preview_and_close_remain_visible_when_window_shrinks(tmp_path: Path) -> None:
    dialog = ArbWaveformDialog(_FakeController())
    dialog.show()
    driver = ComponentDriver(dialog)
    try:
        driver.screenshot(tmp_path / "01-open.png")
        driver.click("Refresh")
        driver.screenshot(tmp_path / "02-refresh.png")
        qt_until(
            lambda: bool(driver.plots().axes[0].visible_curves),
            label="waveform preview",
        )

        for step, (width, height) in enumerate(((1100, 700), (980, 560)), 3):
            driver.resize_window(width, height)
            driver.screenshot(tmp_path / f"{step:02}-{width}x{height}.png")
            size = driver.window_size()
            assert size.width() <= width
            assert size.height() <= height
            close = driver.observe("Close")
            assert close.enabled and close.fully_visible
            assert driver.plots().axes[0].visible_curves >= 3
            assert all(
                bounds.canvas_left
                <= bounds.ylabel_left
                < bounds.ylabel_right
                <= bounds.axes_left
                < bounds.canvas_right
                for bounds in driver.plot_label_bounds()
            )

        driver.click("Close")
        assert driver.dialogs() == ()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_empty_assets_hint_tracks_refresh(entry_waveform_dialog) -> None:
    from qtpy.QtWidgets import QLabel

    dialog, assets = entry_waveform_dialog
    hint = next(
        (
            label
            for label in dialog.findChildren(QLabel)
            if label.text() == "No waveforms saved yet"
        ),
        None,
    )
    assert hint is not None
    assert hint.isVisible()
    recipe = FormulaRecipe.from_raw(
        {"segments": [{"duration": 0.1, "formula": "1"}], "normalize": "peak"}
    )
    assets.set_formula("wave", recipe, overwrite=False)
    dialog.refresh()
    assert not hint.isVisible()
    assets.delete("wave")
    dialog.refresh()
    assert hint.isVisible()


def test_close_is_in_bottom_right_footer(entry_waveform_dialog, qapp) -> None:
    from qtpy.QtCore import QPoint
    from qtpy.QtWidgets import QPushButton, QSplitter

    dialog, _ = entry_waveform_dialog
    close = next(b for b in dialog.findChildren(QPushButton) if b.text() == "Close")
    splitter = dialog.findChild(QSplitter)
    assert splitter is not None
    close_top = close.mapTo(dialog, QPoint(0, 0))
    splitter_bottom = splitter.mapTo(dialog, QPoint(0, splitter.height()))
    assert close_top.y() >= splitter_bottom.y()
    assert close_top.x() > dialog.width() / 2
    dialog.resize(dialog.width() + 100, dialog.height() + 100)
    qapp.processEvents()
    assert (
        close.mapTo(dialog, QPoint(0, 0)).y()
        >= splitter.mapTo(dialog, QPoint(0, splitter.height())).y()
    )
    close.click()
    assert not dialog.isVisible()


def test_arb_error_uses_theme_foreground(qapp, window_palette):
    from qtpy.QtWidgets import QLabel, QTableWidget

    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    dialog = ArbWaveformDialog(arb_waveform.ArbWaveformService(state))
    try:
        dialog.setPalette(window_palette)
        segments = next(
            table
            for table in dialog.findChildren(QTableWidget)
            if table.columnCount() == 2
        )
        duration = segments.item(0, 0)
        assert duration is not None
        duration.setText("not a duration")
        label = next(
            label
            for label in dialog.findChildren(QLabel)
            if not label.isHidden() and "duration" in label.text().lower()
        )
        label.ensurePolished()
        assert label.palette().color(label.foregroundRole()) == error_colour(label)
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_arb_feedback_switches_between_error_and_info(entry_waveform_dialog, qapp):
    from qtpy.QtGui import QColor, QPalette
    from qtpy.QtWidgets import QLabel, QTableWidget
    from zcu_tools.gui.widgets import warning_colour
    from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

    dialog, assets = entry_waveform_dialog
    palette = dialog.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#353535"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#353535"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#eeeeee"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#eeeeee"))
    dialog.setPalette(palette)
    ArbWaveformDatabase(assets.root_path()).save("raw", [0.0, 1.0], [0.0, 1.0])
    dialog.refresh()
    segments = next(
        table for table in dialog.findChildren(QTableWidget) if table.columnCount() == 2
    )
    duration = segments.item(0, 0)
    assert duration is not None
    duration.setText("not a duration")
    label = next(
        label
        for label in dialog.findChildren(QLabel)
        if not label.isHidden() and "duration" in label.text().lower()
    )
    label.ensurePolished()
    assert label.palette().color(label.foregroundRole()) == error_colour(dialog)

    listing = next(
        table for table in dialog.findChildren(QTableWidget) if table.columnCount() == 3
    )
    listing.selectRow(0)
    assert "no formula recipe" in label.text()
    assert label.palette().color(label.foregroundRole()) == warning_colour(dialog)

    duration = segments.item(0, 0)
    assert duration is not None
    duration.setText("not a duration")
    assert label.palette().color(label.foregroundRole()) == error_colour(dialog)
