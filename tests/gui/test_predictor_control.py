"""PredictorControlFacet delegation and event-subscription contract."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.events import PredictorChangedPayload
from zcu_tools.gui.session.predictor_control import PredictorControlFacet
from zcu_tools.gui.session.services.predictor import (
    LoadPredictorRequest,
    PredictCurveRequest,
    PredictFreqRequest,
    PredictMatrixCurveRequest,
    SetModelParamsRequest,
)


def _facet() -> tuple[PredictorControlFacet, MagicMock, BaseEventBus]:
    bus = BaseEventBus()
    predictor = MagicMock()
    return (
        PredictorControlFacet(bus=bus, predictor=cast(Any, predictor)),
        predictor,
        bus,
    )


def test_predictor_control_facet_delegates_predictor_calls() -> None:
    facet, predictor, _bus = _facet()

    load_req = LoadPredictorRequest(path="/tmp/params.json", flux_bias=0.1)
    model_req = SetModelParamsRequest(
        EJ=4.0,
        EC=1.0,
        EL=1.0,
        flux_half=0.0,
        flux_period=1.0,
        flux_bias=0.0,
    )
    freq_req = PredictFreqRequest(value=0.5, transition=(0, 1))
    curve_req = PredictCurveRequest(
        values=np.array([0.0, 0.5], dtype=np.float64),
        transitions=((0, 1),),
    )
    matrix_req = PredictMatrixCurveRequest(
        values=np.array([0.0, 0.5], dtype=np.float64),
        transitions=((0, 1),),
        operator="n",
    )

    predictor.predict_freq.return_value = 1234.5
    predictor.predict_freq_curve.return_value = "freq-curve"
    predictor.predict_matrix_element_curve.return_value = "matrix-curve"
    predictor.get_predictor_info.return_value = {"loaded": True}

    facet.load_predictor(load_req)
    facet.set_predictor_model_params(model_req)
    facet.clear_predictor()
    assert facet.predict_freq(freq_req) == 1234.5
    assert facet.predict_freq_curve(curve_req) == "freq-curve"
    assert facet.predict_matrix_element_curve(matrix_req) == "matrix-curve"
    assert facet.get_predictor_info() == {"loaded": True}

    predictor.load_predictor.assert_called_once_with(load_req)
    predictor.set_model_params.assert_called_once_with(model_req)
    predictor.clear_predictor.assert_called_once_with()
    predictor.predict_freq.assert_called_once_with(freq_req)
    predictor.predict_freq_curve.assert_called_once_with(curve_req)
    predictor.predict_matrix_element_curve.assert_called_once_with(matrix_req)
    predictor.get_predictor_info.assert_called_once_with()


def test_predictor_control_facet_event_disposer_unsubscribes() -> None:
    facet, _predictor, bus = _facet()
    changed: list[str] = []

    unsubscribe = facet.on_predictor_changed(lambda _payload: changed.append("changed"))

    bus.emit(PredictorChangedPayload())
    assert changed == ["changed"]

    unsubscribe()
    unsubscribe()
    bus.emit(PredictorChangedPayload())
    assert changed == ["changed"]
