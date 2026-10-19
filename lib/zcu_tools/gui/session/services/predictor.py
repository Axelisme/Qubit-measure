"""Independent simulator queries and read-only predictor file loading.

No installed model, dialog state, hardware access, context mutation or events.
Each computation builds a fresh model from the explicit named parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.resources.predictor_file import PredictorFile, load_predictor_file
from zcu_tools.simulate.predictor import CalibrationResult as CalibrateFluxBiasResult
from zcu_tools.simulate.predictor import (
    MatrixPredictor,
    PredictorParams,
)

if TYPE_CHECKING:
    from zcu_tools.gui.session.context_control import ContextControlPort


@dataclass(frozen=True)
class CalibrateFluxBiasRequest:
    """Measured native device value, MHz frequency and increasing level pair."""

    value: float
    frequency_mhz: float
    transition: tuple[int, int] = (0, 1)


@dataclass(frozen=True)
class PredictFreqRequest:
    """Finite native device value and nonnegative integer (from, to) level pair."""

    value: float
    transition: tuple[int, int]


@dataclass(frozen=True)
class PredictCurveRequest:
    """Finite 1-D native values and nonempty ordered increasing/equal level pairs."""

    values: NDArray[np.float64]
    transitions: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PredictCurveResult:
    """Frequency curves; labels follow request transitions (e.g. 0→1).

    values and fluxs are (n_values,) native device values and flux quanta.
    freqs_mhz is (n_transitions, n_values), in MHz, in request row/column order.
    """

    labels: tuple[str, ...]
    values: NDArray[np.float64]
    fluxs: NDArray[np.float64]
    freqs_mhz: NDArray[np.float64]


@dataclass(frozen=True)
class PredictMatrixCurveRequest:
    """Finite 1-D native values, ordered level pairs and charge n or phase phi."""

    values: NDArray[np.float64]
    transitions: tuple[tuple[int, int], ...]
    operator: Literal["n", "phi"]


@dataclass(frozen=True)
class PredictMatrixCurveResult:
    """Matrix curves with labels in transition order (e.g. 0→3).

    values and fluxs are (n_values,) native values and flux quanta.
    mags is (n_transitions, n_values) dimensionless absolute matrix elements.
    """

    labels: tuple[str, ...]
    values: NDArray[np.float64]
    fluxs: NDArray[np.float64]
    mags: NDArray[np.float64]


class PredictorService:
    """Read-only computations, independent of dialog and session model state.

    context supplies the current entry's predictor directory on its owner thread.
    It may be None for standalone explicit-parameter queries. Every query
    leaves parameters, components, files and session versions unchanged.
    """

    def __init__(
        self,
        context: ContextControlPort | None = None,
    ) -> None:
        self._context = context

    def list_files(self) -> tuple[str, ...]:
        """Return sorted *.json filenames in the current entry predictor directory.

        No opened entry or missing directory returns an empty tuple. No point
        selection is required. I/O failures propagate; no directories are created.
        Listing does not read or validate file content.
        """
        root = (
            self._context.get_predictors_path() if self._context is not None else None
        )
        if root is None:
            return ()
        try:
            paths = tuple(root.iterdir())
        except FileNotFoundError:
            return ()
        return tuple(
            sorted(
                path.name for path in paths if path.match("*.json") and path.is_file()
            )
        )

    def load_file(self, filename: str) -> PredictorFile:
        """Read one *.json filename from the current entry predictor directory.

        Accept a filename including .json, not absolute paths or path segments.
        Invalid filenames raise InvalidInputError; no opened entry raises
        FailedPreconditionError. File and content errors from resources propagate.
        No point selection is required, and no file or session state is changed.
        """
        if (
            not filename
            or "/" in filename
            or "\\" in filename
            or Path(filename).is_absolute()
            or Path(filename).suffix != ".json"
        ):
            raise InvalidInputError("Expected a predictor filename ending in .json")
        root = (
            self._context.get_predictors_path() if self._context is not None else None
        )
        if root is None:
            raise FailedPreconditionError("Open an entry to load predictor files")
        return load_predictor_file(root / filename)

    def predict_freq(self, params: PredictorParams, req: PredictFreqRequest) -> float:
        """Return MHz for the explicit params/request; PredictorInputError rejects missing/invalid inputs."""
        return params.build_predictor().predict_freq(req.value, req.transition)

    def calibrate_flux_bias(
        self, params: PredictorParams, req: CalibrateFluxBiasRequest
    ) -> CalibrateFluxBiasResult:
        """Return the model's native bias, actual MHz and clamped status, without writes.

        Missing/invalid params raise PredictorInputError. Non-increasing/negative or
        noninteger levels raise InvalidInputError. The model owns root selection
        and clamping; numerical failures propagate. No caller recomputes the fit.
        """
        frm, to = req.transition
        if type(frm) is not int or type(to) is not int or frm < 0 or frm >= to:
            raise InvalidInputError(
                f"Transition from-level must be < to-level and >= 0, got {req.transition}",
                reason_code="invalid_transition",
            )
        return params.build_predictor().calculate_bias(
            req.value, req.frequency_mhz, req.transition
        )

    def predict_freq_curve(
        self, params: PredictorParams, req: PredictCurveRequest
    ) -> PredictCurveResult:
        """Return labeled MHz curves using the neutral batch model contract.

        Missing/invalid params or batch inputs raise PredictorInputError. Engine failures
        propagate. Returned values are copied; the request arrays are not changed.
        """
        values = np.array(req.values, dtype=np.float64, copy=True)
        fluxs, freqs = params.build_predictor().predict_frequencies_mhz(
            values, req.transitions
        )
        return PredictCurveResult(
            tuple(f"{frm}→{to}" for frm, to in req.transitions), values, fluxs, freqs
        )

    def predict_matrix_element_curve(
        self, params: PredictorParams, req: PredictMatrixCurveRequest
    ) -> PredictMatrixCurveResult:
        """Return labeled dimensionless magnitudes for n/phi, including high levels.

        PredictorInputError rejects missing/invalid inputs. A frequency-only model raises
        FailedPreconditionError. Engine errors propagate; inputs are not changed.
        """
        model = params.build_predictor()
        if not isinstance(model, MatrixPredictor):
            raise FailedPreconditionError("This model does not provide matrix elements")
        values = np.array(req.values, dtype=np.float64, copy=True)
        fluxs, mags = model.predict_matrix_elements(
            values, req.transitions, req.operator
        )
        return PredictMatrixCurveResult(
            tuple(f"{frm}→{to}" for frm, to in req.transitions), values, fluxs, mags
        )
