from __future__ import annotations

import time
from typing import Any

import pytest
from zcu_tools.gui.app.main.ui.arb_waveform_dialog import ArbWaveformDialog
from zcu_tools.meta_tool import ArbWaveformData, ArbWaveformInfo, render_formula_recipe


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
    formula_item.setText("unknown_symbol")
    assert not dlg._save_btn.isEnabled()
    assert not dlg._warning.isHidden()
    assert "unsupported" in dlg._warning.text()

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
