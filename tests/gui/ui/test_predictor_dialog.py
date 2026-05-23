"""Smoke tests for PredictorDialog."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from zcu_tools.gui.ui.predictor_dialog import PredictorDialog


def test_predictor_dialog_init_and_load(qapp):
    ctrl = MagicMock()
    ctrl.get_predictor_info.return_value = {"path": "/fake/path.json", "flux_bias": 0.5}

    dialog = PredictorDialog(ctrl)

    assert dialog._path_edit.text() == "/fake/path.json"
    assert dialog._flux_bias_spin.value() == 0.5

    with patch(
        "zcu_tools.simulate.fluxonium.predict.FluxoniumPredictor"
    ) as mock_predictor:
        mock_instance = MagicMock()
        mock_predictor.from_file.return_value = mock_instance

        dialog._path_edit.setText("new.json")
        # trigger load
        dialog._on_accepted()

        mock_predictor.from_file.assert_called_with("new.json", flux_bias=0.5)
        ctrl.set_predictor.assert_called_with(mock_instance, path="new.json")


def test_predictor_dialog_predict(qapp):
    ctrl = MagicMock()
    ctrl.get_predictor_info.return_value = None
    mock_predictor = MagicMock()
    mock_predictor.predict_freq.return_value = 123.456
    ctrl.get_predictor.return_value = mock_predictor

    dialog = PredictorDialog(ctrl)

    dialog._predict_value_spin.setValue(1.5)
    dialog._from_spin.setValue(1)
    dialog._to_spin.setValue(2)

    dialog._on_predict_clicked()

    mock_predictor.predict_freq.assert_called_with(1.5, transition=(1, 2))
    assert dialog._predict_result_label.text() == "123.4560 MHz"


def test_predictor_dialog_clear(qapp):
    ctrl = MagicMock()
    ctrl.get_predictor_info.return_value = None
    dialog = PredictorDialog(ctrl)

    dialog._on_clear()
    ctrl.set_predictor.assert_called_with(None)
    assert dialog._predict_result_label.text() == "—"
