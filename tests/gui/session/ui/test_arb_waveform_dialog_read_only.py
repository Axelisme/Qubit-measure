"""Read-only editing contract of the shared arbitrary waveform dialog."""

from pathlib import Path

import pytest
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QLineEdit,
    QPushButton,
    QTableWidget,
)
from zcu_tools.gui.session.services import arb_waveform
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.arb_waveform_dialog import ArbWaveformDialog
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase, FormulaRecipe


@pytest.fixture
def entry_waveform_dialog(qapp, tmp_path: Path):
    """Bind the waveform service to an entry installed by the real context owner."""
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.services.context import ContextService
    from zcu_tools.resources.entry import ResultEntry

    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    context = ContextService(
        state, BaseEventBus(), bind_assets=ArbWaveformDatabase.info_at
    )
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )
    context.install_entry(entry)
    assets = arb_waveform.ArbWaveformService(state)
    dialog = ArbWaveformDialog(assets)
    dialog.show()
    qapp.processEvents()
    try:
        yield dialog, assets
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def _dialog_button(dialog: ArbWaveformDialog, text: str) -> QPushButton:
    return next(b for b in dialog.findChildren(QPushButton) if b.text() == text)


def _recipe_table(dialog: ArbWaveformDialog) -> QTableWidget:
    return next(t for t in dialog.findChildren(QTableWidget) if t.columnCount() == 2)


def test_read_only_preserves_draft_and_restores_editing(entry_waveform_dialog) -> None:
    dialog, assets = entry_waveform_dialog
    key = dialog.findChild(QLineEdit)
    normalize = dialog.findChild(QCheckBox)
    assert key is not None and normalize is not None
    segments = _recipe_table(dialog)
    formula = segments.item(0, 1)
    assert formula is not None
    key.setText("unsaved_draft")
    formula.setText("0.25")
    normalize.setChecked(False)
    triggers = segments.editTriggers()
    save = _dialog_button(dialog, "Save")
    assert save.isEnabled()

    dialog.set_read_only(read_only=True)
    dialog.set_read_only(read_only=True)
    assert key.isReadOnly()
    assert segments.isEnabled()
    assert segments.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    assert not normalize.isEnabled()
    for text in (
        "Rename",
        "Delete",
        "Insert Before",
        "Insert After",
        "Delete Segment",
        "Save",
    ):
        assert not _dialog_button(dialog, text).isEnabled(), text
    save.click()
    assert assets.list_data_keys() == []
    assert key.text() == "unsaved_draft"
    assert formula.text() == "0.25"
    assert not normalize.isChecked()
    segments.selectRow(0)
    assert segments.selectedItems()

    dialog.set_read_only(read_only=False)
    dialog.set_read_only(read_only=False)
    assert not key.isReadOnly()
    assert segments.editTriggers() == triggers
    assert normalize.isEnabled()
    for text in ("Insert Before", "Insert After", "Delete Segment", "Save"):
        assert _dialog_button(dialog, text).isEnabled(), text
    save.click()
    stored = assets.load_data("unsaved_draft").recipe
    assert stored is not None
    assert stored.segments[0].formula == "0.25"
    assert stored.normalize == "none"


def test_read_only_allows_asset_selection_preview_refresh_and_close(
    entry_waveform_dialog,
) -> None:
    dialog, assets = entry_waveform_dialog
    recipe = FormulaRecipe.from_raw(
        {"segments": [{"duration": 0.1, "formula": "0.5"}], "normalize": "none"}
    )
    assets.set_formula("wave", recipe, overwrite=False)
    dialog.set_read_only(read_only=True)
    refresh = _dialog_button(dialog, "Refresh")
    assert refresh.isEnabled()
    refresh.click()
    table = next(t for t in dialog.findChildren(QTableWidget) if t.columnCount() == 3)
    assert table.isEnabled()
    table.selectRow(0)
    key = dialog.findChild(QLineEdit)
    assert key is not None and key.text() == "wave"
    formula = _recipe_table(dialog).item(0, 1)
    assert formula is not None and formula.text() == "0.5"
    canvas = dialog.findChild(FigureCanvasQTAgg)
    assert canvas is not None and canvas.isEnabled()
    assert canvas.figure.axes[0].get_title() == "wave"
    assert len(canvas.figure.axes[0].lines) >= 3
    for text in ("Rename", "Delete", "Save"):
        assert not _dialog_button(dialog, text).isEnabled(), text
    dialog.set_read_only(read_only=False)
    for text in ("Rename", "Delete", "Save"):
        assert _dialog_button(dialog, text).isEnabled(), text
    close = _dialog_button(dialog, "Close")
    assert close.isEnabled()
    close.click()
    assert not dialog.isVisible()


def test_unlock_keeps_invalid_draft_unsaveable(entry_waveform_dialog) -> None:
    dialog, _ = entry_waveform_dialog
    duration = _recipe_table(dialog).item(0, 0)
    assert duration is not None
    duration.setText("not a duration")
    dialog.set_read_only(read_only=True)
    dialog.set_read_only(read_only=False)
    assert duration.text() == "not a duration"
    assert not _dialog_button(dialog, "Save").isEnabled()
