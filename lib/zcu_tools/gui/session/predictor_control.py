"""Narrow read-only simulator control shared by dialog and remote adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from zcu_tools.gui.session.services.predictor import (
        CalibrateOffsetRequest,
        PredictCurveRequest,
        PredictCurveResult,
        PredictFreqRequest,
        PredictMatrixCurveRequest,
        PredictMatrixCurveResult,
    )
    from zcu_tools.resources.predictor_file import PredictorFile
    from zcu_tools.simulate.predictor import CalibrationResult, PredictorParams


class PredictorControlPort(Protocol):
    """Side-effect-free queries; explicit parameters are required for computation.

    Units, shapes and failures are defined by the typed requests/results and
    PredictorService. No query installs a model or reads dialog state/cache.
    """

    def list_files(self) -> tuple[str, ...]:
        """List current entry predictor filenames; empty without an entry."""
        ...

    def load_file(self, filename: str) -> PredictorFile:
        """Read one entry predictor filename; see PredictorService.load_file."""
        ...

    def predict_freq(self, params: PredictorParams, req: PredictFreqRequest) -> float:
        """Query explicit parameters for MHz; see PredictorService.predict_freq."""
        ...

    def calibrate_offset(
        self, params: PredictorParams, req: CalibrateOffsetRequest
    ) -> CalibrationResult:
        """Return additive Phi0 offset correction, actual MHz and clamped status; see PredictorService.calibrate_offset."""
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
