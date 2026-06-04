"""Smoke tests for PredictorDialog."""

from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.gui.app.main.services.connection import (
    LoadPredictorRequest,
    PredictFreqRequest,
    PredictorLoadError,
    PredictorNotLoaded,
)
from zcu_tools.gui.app.main.ui.predictor_dialog import PredictorDialog


def test_predictor_dialog_init_and_load_dispatches_request(qapp):
    ctrl = MagicMock()
    ctrl.get_predictor_info.return_value = {"path": "/fake/path.json", "flux_bias": 0.5}

    dialog = PredictorDialog(ctrl)

    assert dialog._path_edit.text() == "/fake/path.json"
    assert dialog._flux_bias_spin.value() == 0.5

    dialog._path_edit.setText("new.json")
    dialog._on_accepted()

    ctrl.load_predictor.assert_called_once()
    (req,) = ctrl.load_predictor.call_args.args
    assert isinstance(req, LoadPredictorRequest)
    assert req.path == "new.json"
    assert req.flux_bias == 0.5


def test_predictor_dialog_load_failure_shown_in_status(qapp):
    ctrl = MagicMock()
    ctrl.get_predictor_info.return_value = None
    ctrl.load_predictor.side_effect = PredictorLoadError("bad file")

    dialog = PredictorDialog(ctrl)
    dialog._path_edit.setText("missing.json")
    dialog._on_accepted()

    assert "bad file" in dialog._status_label.text()


def test_predictor_dialog_predict_dispatches_request(qapp):
    ctrl = MagicMock()
    ctrl.get_predictor_info.return_value = None
    ctrl.predict_freq.return_value = 123.456

    dialog = PredictorDialog(ctrl)
    dialog._predict_value_spin.setValue(1.5)
    dialog._from_spin.setValue(1)
    dialog._to_spin.setValue(2)
    dialog._on_predict_clicked()

    ctrl.predict_freq.assert_called_once()
    (req,) = ctrl.predict_freq.call_args.args
    assert isinstance(req, PredictFreqRequest)
    assert req.value == 1.5
    assert req.transition == (1, 2)
    assert dialog._predict_result_label.text() == "123.4560 MHz"


def test_predictor_dialog_predict_without_predictor_shows_status(qapp):
    ctrl = MagicMock()
    ctrl.get_predictor_info.return_value = None
    ctrl.predict_freq.side_effect = PredictorNotLoaded("nothing loaded")

    dialog = PredictorDialog(ctrl)
    dialog._on_predict_clicked()

    assert "nothing loaded" in dialog._status_label.text()


def test_predictor_dialog_clear(qapp):
    ctrl = MagicMock()
    ctrl.get_predictor_info.return_value = None
    dialog = PredictorDialog(ctrl)

    dialog._on_clear()
    ctrl.clear_predictor.assert_called_once()
    assert dialog._predict_result_label.text() == "—"
