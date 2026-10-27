"""Typed numerical-failure collaborator for predictor RPC contracts."""

from zcu_tools.gui.session.services.predictor import (
    CalibrateOffsetRequest,
    PredictCurveRequest,
    PredictCurveResult,
    PredictFreqRequest,
    PredictorService,
)
from zcu_tools.simulate.predictor import CalibrationResult, PredictorParams


class FailingPredictorService(PredictorService):
    """Record query inputs and raise the supplied unexpected solver failure."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error
        self.requests: list[
            PredictFreqRequest | CalibrateOffsetRequest | PredictCurveRequest
        ] = []

    def predict_freq_curve(
        self, params: PredictorParams, req: PredictCurveRequest
    ) -> PredictCurveResult:
        self.requests.append(req)
        raise self.error

    def predict_freq(self, params: PredictorParams, req: PredictFreqRequest) -> float:
        self.requests.append(req)
        raise self.error

    def calibrate_offset(
        self, params: PredictorParams, req: CalibrateOffsetRequest
    ) -> CalibrationResult:
        self.requests.append(req)
        raise self.error
