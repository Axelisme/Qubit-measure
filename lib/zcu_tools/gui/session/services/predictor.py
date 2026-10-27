"""Independent simulator queries and read-only predictor file loading.

No installed model, dialog state, hardware access, context mutation or events.
Each computation builds a fresh model from the explicit named parameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.flux import FluxFrame, flux_from_device_values
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.resources.predictor_file import PredictorFile, load_predictor_file
from zcu_tools.simulate.predictor import (
    CalibrationResult,
    MatrixPredictor,
    PredictorInputError,
    PredictorParams,
)

if TYPE_CHECKING:
    from zcu_tools.gui.session.context_control import ContextControlPort


@dataclass(frozen=True)
class CalibrateOffsetRequest:
    """A measured transition for additive offset calibration without writes.

    frame defines offset in Phi0 and signed native periods/units. device_values
    supplies every frame device in its native unit; extras are ignored.
    frequency_mhz is finite measured MHz, including unattainable targets.
    transition is a nonnegative integer (from, to) pair with from < to.
    """

    frame: FluxFrame
    device_values: Mapping[str, float]
    frequency_mhz: float
    transition: tuple[int, int] = (0, 1)


@dataclass(frozen=True)
class PredictFreqRequest:
    """One prediction from explicit flux coordinates; no hardware reads.

    frame defines zero-output offset (Phi0) and signed native periods/units.
    device_values maps every frame device to a finite native value; extra devices
    are ignored. transition is a nonnegative integer (from, to) pair, to >= from.
    """

    frame: FluxFrame
    device_values: Mapping[str, float]
    transition: tuple[int, int] = (0, 1)


@dataclass(frozen=True)
class PredictCurveRequest:
    """Scan one device while holding the other flux lines fixed.

    frame defines offset (Phi0) and signed native periods/units. axis_device is
    a key in frame.devices. values is a finite 1-D array in that device's unit.
    device_values supplies every other frame device; the axis value and extras
    are ignored. transitions is a nonempty ordered tuple of nonnegative integer
    (from, to) pairs with to >= from. Input arrays/mappings remain unchanged.
    """

    frame: FluxFrame
    axis_device: str
    values: NDArray[np.float64]
    device_values: Mapping[str, float]
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
    """Matrix scan with the same coordinates as PredictCurveRequest.

    frame defines offset (Phi0) and signed native periods/units. axis_device is
    a frame device; values is its finite 1-D native scan. device_values supplies
    every other frame device, ignoring axis/extras. transitions is a nonempty
    ordered tuple of nonnegative integer (from, to) pairs, to >= from.
    operator selects dimensionless charge n or phase phi matrix magnitudes.
    """

    frame: FluxFrame
    axis_device: str
    values: NDArray[np.float64]
    device_values: Mapping[str, float]
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


def _required_flux(frame: FluxFrame, device_values: Mapping[str, float]) -> float:
    missing = frame.devices.keys() - device_values.keys()
    if missing:
        raise PredictorInputError(
            "Missing device values: " + ", ".join(sorted(missing))
        )
    try:
        return flux_from_device_values(frame, device_values)
    except ValueError as exc:
        raise PredictorInputError(str(exc)) from exc


def _curve_fluxs(
    req: PredictCurveRequest | PredictMatrixCurveRequest,
) -> NDArray[np.float64]:
    if req.axis_device not in req.frame.devices:
        raise PredictorInputError(f"Unknown axis device {req.axis_device!r}")
    if req.values.ndim != 1:
        raise PredictorInputError("values must be a finite 1-D array")
    _required_flux(req.frame, {**req.device_values, req.axis_device: 0.0})
    return np.array(
        [
            _required_flux(req.frame, {**req.device_values, req.axis_device: value})
            for value in req.values
        ],
        dtype=np.float64,
    )


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
        """Project the request's native device values and return transition MHz.

        params holds the selected model's energy/asymmetry inputs. req supplies
        frame, every required device value and level pair. Missing device values
        or invalid inputs raise PredictorInputError naming the field/device.
        Extra devices are ignored. Numerical failures propagate; no state changes.
        """
        flux = _required_flux(req.frame, req.device_values)
        return params.build_predictor().predict_freq(flux, req.transition)

    def calibrate_offset(
        self, params: PredictorParams, req: CalibrateOffsetRequest
    ) -> CalibrationResult:
        """Return additive Phi0 offset_correction, actual MHz and clamped, without writes.

        params holds energy/asymmetry inputs; req supplies frame, all device
        values, measured MHz and increasing level pair. Add offset_correction
        to the caller's current frame.offset. Missing devices/invalid parameters
        raise PredictorInputError. Invalid levels raise InvalidInputError.
        The model selects the smallest absolute correction across all roots;
        unattainable targets clamp to nearest MHz. Numerical failures propagate.
        """
        frm, to = req.transition
        if type(frm) is not int or type(to) is not int or frm < 0 or frm >= to:
            raise InvalidInputError(
                f"Transition from-level must be < to-level and >= 0, got {req.transition}",
                reason_code="invalid_transition",
            )
        flux = _required_flux(req.frame, req.device_values)
        return params.build_predictor().calculate_offset_correction(
            flux, req.frequency_mhz, req.transition
        )

    def predict_freq_curve(
        self, params: PredictorParams, req: PredictCurveRequest
    ) -> PredictCurveResult:
        """Scan req.axis_device with other native values fixed; return labeled MHz rows.

        params holds energy/asymmetry inputs. req defines frame, scan values,
        fixed device values and ordered level pairs. Missing fixed devices,
        unknown axis or invalid inputs raise PredictorInputError. Engine failures
        propagate. Returned values are copied; request arrays remain unchanged.
        """
        values = np.array(req.values, dtype=np.float64, copy=True)
        fluxs, freqs = params.build_predictor().predict_frequencies_mhz(
            _curve_fluxs(req), req.transitions
        )
        return PredictCurveResult(
            tuple(f"{frm}→{to}" for frm, to in req.transitions), values, fluxs, freqs
        )

    def predict_matrix_element_curve(
        self, params: PredictorParams, req: PredictMatrixCurveRequest
    ) -> PredictMatrixCurveResult:
        """Scan req.axis_device and return labeled dimensionless n/phi magnitudes.

        params holds energy/asymmetry inputs; req defines frame, scan values,
        fixed device values, ordered level pairs and operator, including high levels.
        PredictorInputError names missing devices, unknown axis or invalid inputs.
        A frequency-only model raises FailedPreconditionError. Engine errors
        propagate; inputs remain unchanged.
        """
        model = params.build_predictor()
        if not isinstance(model, MatrixPredictor):
            raise FailedPreconditionError("This model does not provide matrix elements")
        values = np.array(req.values, dtype=np.float64, copy=True)
        fluxs, mags = model.predict_matrix_elements(
            _curve_fluxs(req), req.transitions, req.operator
        )
        return PredictMatrixCurveResult(
            tuple(f"{frm}→{to}" for frm, to in req.transitions), values, fluxs, mags
        )
