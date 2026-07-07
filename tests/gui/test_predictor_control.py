"""PredictorControlFacet public contract tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.events import PredictorChangedPayload
from zcu_tools.gui.session.predictor_control import PredictorControlFacet
from zcu_tools.gui.session.services.predictor import (
    CalibrateFluxBiasRequest,
    CalibrateFluxBiasResult,
    LoadPredictorRequest,
    PredictCurveRequest,
    PredictFreqRequest,
    PredictMatrixCurveRequest,
    SetModelParamsRequest,
)

from tests.gui._control_fakes import CallLog, RecordedCall, call, same


class RecordingPredictor:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.freq_curve = object()
        self.matrix_curve = object()
        self.calibration = CalibrateFluxBiasResult(flux_bias=0.125)
        self.info: dict[str, bool] = {"loaded": True}

    def load_predictor(self, req: LoadPredictorRequest) -> None:
        self._log.add("predictor", "load_predictor", req)

    def set_model_params(self, req: SetModelParamsRequest) -> None:
        self._log.add("predictor", "set_model_params", req)

    def calibrate_flux_bias(
        self, req: CalibrateFluxBiasRequest
    ) -> CalibrateFluxBiasResult:
        self._log.add("predictor", "calibrate_flux_bias", req)
        return self.calibration

    def clear_predictor(self) -> None:
        self._log.add("predictor", "clear_predictor")

    def predict_freq(self, req: PredictFreqRequest) -> float:
        self._log.add("predictor", "predict_freq", req)
        return 1234.5

    def predict_freq_curve(self, req: PredictCurveRequest) -> object:
        self._log.add("predictor", "predict_freq_curve", req)
        return self.freq_curve

    def predict_matrix_element_curve(self, req: PredictMatrixCurveRequest) -> object:
        self._log.add("predictor", "predict_matrix_element_curve", req)
        return self.matrix_curve

    def get_predictor_info(self) -> dict[str, bool]:
        self._log.add("predictor", "get_predictor_info")
        return self.info


def _facet() -> tuple[PredictorControlFacet, CallLog, RecordingPredictor, BaseEventBus]:
    log = CallLog()
    bus = BaseEventBus()
    predictor = RecordingPredictor(log)
    return (
        PredictorControlFacet(bus=bus, predictor=cast(Any, predictor)),
        log,
        predictor,
        bus,
    )


def test_predictor_control_facet_forwards_deliberate_predictor_contract() -> None:
    facet, log, predictor, _bus = _facet()
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
    calibrate_req = CalibrateFluxBiasRequest(
        value=0.5,
        frequency_mhz=4567.0,
        transition=(0, 1),
    )
    curve_req = PredictCurveRequest(
        values=np.array([0.0, 0.5], dtype=np.float64),
        transitions=((0, 1),),
    )
    matrix_req = PredictMatrixCurveRequest(
        values=np.array([0.0, 0.5], dtype=np.float64),
        transitions=((0, 1),),
        operator="n",
    )

    cases: tuple[tuple[str, Callable[[], object], object, RecordedCall], ...] = (
        (
            "load_predictor",
            lambda: facet.load_predictor(load_req),
            None,
            call("predictor", "load_predictor", same(load_req)),
        ),
        (
            "set_predictor_model_params",
            lambda: facet.set_predictor_model_params(model_req),
            None,
            call("predictor", "set_model_params", same(model_req)),
        ),
        (
            "clear_predictor",
            facet.clear_predictor,
            None,
            call("predictor", "clear_predictor"),
        ),
        (
            "calibrate_flux_bias",
            lambda: facet.calibrate_flux_bias(calibrate_req),
            predictor.calibration,
            call("predictor", "calibrate_flux_bias", same(calibrate_req)),
        ),
        (
            "predict_freq",
            lambda: facet.predict_freq(freq_req),
            1234.5,
            call("predictor", "predict_freq", same(freq_req)),
        ),
        (
            "predict_freq_curve",
            lambda: facet.predict_freq_curve(curve_req),
            predictor.freq_curve,
            call("predictor", "predict_freq_curve", same(curve_req)),
        ),
        (
            "predict_matrix_element_curve",
            lambda: facet.predict_matrix_element_curve(matrix_req),
            predictor.matrix_curve,
            call("predictor", "predict_matrix_element_curve", same(matrix_req)),
        ),
        (
            "get_predictor_info",
            facet.get_predictor_info,
            {"loaded": True},
            call("predictor", "get_predictor_info"),
        ),
    )

    for name, action, expected_result, _expected_call in cases:
        assert action() == expected_result, name

    assert log.calls == [expected_call for *_, expected_call in cases]


def test_predictor_control_facet_event_disposer_unsubscribes() -> None:
    facet, _log, _predictor, bus = _facet()
    changed: list[str] = []

    unsubscribe = facet.on_predictor_changed(lambda _payload: changed.append("changed"))

    bus.emit(PredictorChangedPayload())
    assert changed == ["changed"]

    unsubscribe()
    unsubscribe()
    bus.emit(PredictorChangedPayload())
    assert changed == ["changed"]
