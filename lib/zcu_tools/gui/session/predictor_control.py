"""Narrow read-only simulator control shared by dialog and remote adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from zcu_tools.gui.session.services.predictor import (
        CalibrateFluxBiasRequest,
        CalibrateFluxBiasResult,
        PredictCurveRequest,
        PredictCurveResult,
        PredictFreqRequest,
        PredictMatrixCurveRequest,
        PredictMatrixCurveResult,
        PredictorService,
    )
    from zcu_tools.simulate.parameters import ModelParams
    from zcu_tools.simulate.predictor import PredictorParams


class PredictorControlPort(Protocol):
    """Side-effect-free queries; explicit parameters are required for computation.

    Units, shapes and failures are defined by the typed requests/results and
    PredictorService. No query installs a model or reads dialog state/cache.
    """

    def params_for_qubit(self, qubit: str) -> ModelParams:
        """Read current concrete component inputs; see PredictorService.params_for_qubit."""
        ...

    def predict_freq(self, params: PredictorParams, req: PredictFreqRequest) -> float:
        """Query explicit parameters for MHz; see PredictorService.predict_freq."""
        ...

    def calibrate_flux_bias(
        self, params: PredictorParams, req: CalibrateFluxBiasRequest
    ) -> CalibrateFluxBiasResult:
        """Return candidate native bias only; see PredictorService.calibrate_flux_bias."""
        ...

    def predict_freq_curve(
        self, params: PredictorParams, req: PredictCurveRequest
    ) -> PredictCurveResult:
        """Query explicit parameters for MHz rows; see PredictorService.predict_freq_curve."""
        ...

    def predict_matrix_element_curve(
        self, params: PredictorParams, req: PredictMatrixCurveRequest
    ) -> PredictMatrixCurveResult:
        """Query optional matrix rows; see PredictorService.predict_matrix_element_curve."""
        ...


class PredictorControlFacet:
    """Forward typed queries to PredictorService; no state or event subscription."""

    def __init__(self, predictor: PredictorService) -> None:
        self._predictor = predictor

    def params_for_qubit(self, qubit: str) -> ModelParams:
        """Read current concrete inputs; see PredictorService.params_for_qubit."""
        return self._predictor.params_for_qubit(qubit)

    def predict_freq(self, params: PredictorParams, req: PredictFreqRequest) -> float:
        """Return MHz; see PredictorService.predict_freq."""
        return self._predictor.predict_freq(params, req)

    def calibrate_flux_bias(
        self, params: PredictorParams, req: CalibrateFluxBiasRequest
    ) -> CalibrateFluxBiasResult:
        """Return candidate native bias only; see PredictorService.calibrate_flux_bias."""
        return self._predictor.calibrate_flux_bias(params, req)

    def predict_freq_curve(
        self, params: PredictorParams, req: PredictCurveRequest
    ) -> PredictCurveResult:
        """Return MHz rows; see PredictorService.predict_freq_curve."""
        return self._predictor.predict_freq_curve(params, req)

    def predict_matrix_element_curve(
        self, params: PredictorParams, req: PredictMatrixCurveRequest
    ) -> PredictMatrixCurveResult:
        """Return matrix rows; see PredictorService.predict_matrix_element_curve."""
        return self._predictor.predict_matrix_element_curve(params, req)
