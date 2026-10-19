"""Independent simulator queries and read-only component/file parameter prefill.

No installed model, dialog state, hardware access, context mutation or events.
Each computation builds a fresh model from the explicit named parameters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.resources.entry.schema import ComponentSchema
from zcu_tools.resources.entry.views import ComponentView
from zcu_tools.resources.predictor_file import PredictorFile, load_predictor_file
from zcu_tools.resources.qubit_params import QubitParams, QubitParamsError
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulate.parameters import ModelParams
from zcu_tools.simulate.predictor import CalibrationResult as CalibrateFluxBiasResult
from zcu_tools.simulate.predictor import (
    MatrixPredictor,
    PredictorInputError,
    PredictorParams,
)

if TYPE_CHECKING:
    from zcu_tools.gui.session.context_control import ContextControlPort

# The user package owns component field meanings. The framework only passes the
# captured component; no factory metadata, write paths or schema guessing.
ParamsFor = Callable[[ComponentSchema | ComponentView], ModelParams]


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


class PredictorLoadError(FailedPreconditionError):
    """Expected failure reading the legacy params.json prefill source."""


def read_fluxdep_fit_params(path: str) -> FluxoniumParams:
    """Read fluxdep_fit from an existing params.json; never write the file.

    Return GHz energies and native affine anchors. The legacy file has no bias,
    so its documented prefill bias is 0.0. flux_int is alignment-only, not an
    input to the simulator. I/O/missing-fit errors become PredictorLoadError.
    """
    try:
        model = QubitParams(path, readonly=True).require_fluxonium_model()
    except (OSError, QubitParamsError) as exc:
        raise PredictorLoadError(f"Failed to read params file: {exc}") from exc
    return FluxoniumParams(
        model.EJ, model.EC, model.EL, model.flux_half, model.flux_period, 0.0
    )


class PredictorService:
    """Read-only computations, independent of dialog and session model state.

    context supplies a current point capture on its owner thread; params_for is
    injected by app composition and owns component→named-input conversion.
    Either may be None for standalone explicit-parameter queries. Every query
    leaves parameters, components, files and session versions unchanged.
    """

    def __init__(
        self,
        context: ContextControlPort | None = None,
        params_for: ParamsFor | None = None,
    ) -> None:
        self._context = context
        self._params_for = params_for

    def list_files(self) -> tuple[str, ...]:
        """Return sorted *.json filenames in the current entry predictor directory.

        No opened entry or missing directory returns an empty tuple. No point
        selection is required. I/O failures propagate; no directories are created.
        Listing does not read or validate file content.
        """
        root = (
            self._context.get_predictors_path() if self._context is not None else None
        )
        if root is None or not root.exists():
            return ()
        return tuple(
            sorted(path.name for path in root.glob("*.json") if path.is_file())
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

    def params_for_qubit(self, qubit: str) -> ModelParams:
        """Read named inputs for one concrete component at the current point.

        Names must be public Python identifiers (e.g. Q2), not aliases/paths.
        PredictorInputError names invalid/unknown components or unsupported physics kinds.
        Missing provider/context raises FailedPreconditionError. EMPTY/source
        faults from ContextService propagate. Missing fields remain None.
        No read refreshes a source, changes aliases or binds a dialog to a qubit.
        """
        if not qubit.isidentifier() or qubit.startswith("_"):
            raise PredictorInputError(f"Invalid concrete qubit name: {qubit}")
        if self._context is None or self._params_for is None:
            raise FailedPreconditionError(
                "Component parameter prefill is not configured"
            )
        document = self._context.read_context().document()
        if qubit not in document.components:
            raise PredictorInputError(f"Unknown qubit component: {qubit}")
        return self._params_for(document.components[qubit])

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
