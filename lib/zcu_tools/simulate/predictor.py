"""Storage-independent simulator contracts; all queries are side-effect free."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


class PredictorInputError(ValueError):
    """Caller-correctable simulator input failure, with ValueError ancestry.

    Named-parameter decoding/building rejects unknown fields, missing values,
    nonnumeric/bool/nonfinite inputs and invalid energies/asymmetry.
    Query validators reject invalid values, shapes, operators and transitions.
    The message names the invalid input. No GUI or wire metadata is attached.
    Ordinary ValueError, nonfinite computed spectra and solver exceptions are
    not input failures; frontends must preserve their diagnostic traceback.
    """

    def __init__(self, message: str) -> None:
        """Describe the caller-correctable input failure, naming its field or constraint."""
        super().__init__(message)


@dataclass(frozen=True)
class CalibrationResult:
    """Read-only spectrum inversion result, not installed in any model or source.

    offset_correction is the additive Phi0 correction with minimum absolute value.
    frequency_mhz is the actual transition frequency after correction. clamped is true
    only for an unreachable target; then frequency_mhz is globally closest to it,
    with equal-frequency choices resolved by smallest absolute correction.
    """

    offset_correction: float
    frequency_mhz: float
    clamped: bool


class Predictor(Protocol):
    """Read-only model queries with reduced flux (Phi0) and MHz frequencies.

    Flux coordinates are in flux quanta, not folded into a single period.
    Batch inputs are finite one-dimensional arrays; transitions are a nonempty
    ordered tuple of nonnegative integer (from, to) levels with to >= from.
    Invalid transitions raise PredictorInputError; numerical and array-shape failures
    propagate from the selected engine.
    No query changes parameters or inputs, reads hardware or writes storage.
    """

    def predict_freq(
        self, cur_flux: float, transition: tuple[int, int] = (0, 1)
    ) -> float:
        """Return E_to - E_from in MHz at a finite reduced flux in Phi0.

        Levels are nonnegative integers; equal levels return zero. Negative
        levels raise PredictorInputError. Engine failures propagate without state changes.
        """
        ...

    def predict_frequencies_mhz(
        self,
        fluxs: NDArray[np.float64],
        transitions: tuple[tuple[int, int], ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (fluxs, frequencies) for reduced flux and ordered transitions.

        fluxs has shape (n_fluxs,) in flux quanta; frequencies has shape
        (n_transitions, n_fluxs) in MHz. Rows preserve transition order, columns
        preserve input order including duplicates. See class input/error contract.
        """
        ...

    def calculate_offset_correction(
        self, cur_flux: float, cur_freq: float, transition: tuple[int, int] = (0, 1)
    ) -> CalibrationResult:
        """Invert measured MHz at current reduced flux; return an offset correction.

        cur_flux is finite Phi0 including the caller's current offset.
        cur_freq is finite MHz; transition has integer levels with from < to.
        Select the smallest absolute correction across all roots and periodic/
        mirror equivalents. Unreachable targets return globally nearest MHz and
        clamped=true, with ties choosing the smallest absolute correction.
        Add the result to the caller's offset; this query installs nothing.
        Invalid inputs raise PredictorInputError; numerical failures propagate.
        Frequency equality uses 1e-7 MHz, branch discovery a 401-point half-period
        grid, and exact-spectrum root/extremum refinement without monotonicity.
        """
        ...


@runtime_checkable
class MatrixPredictor(Predictor, Protocol):
    """Optional matrix curves; frequency-only models need not implement this."""

    def predict_matrix_elements(
        self,
        fluxs: NDArray[np.float64],
        transitions: tuple[tuple[int, int], ...],
        operator: Literal["n", "phi"],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (fluxs, magnitudes) for reduced flux and ordered transitions.

        fluxs has shape (n_fluxs,) in flux quanta; dimensionless absolute
        |<from|operator|to>| rows have shape (n_transitions, n_fluxs).
        n is the charge operator, phi the phase operator. High levels are allowed.
        Unknown operators or invalid batch inputs raise PredictorInputError; see Predictor.
        """
        ...


class PredictorParams(Protocol):
    """Partial named physical inputs shared by notebook, service and dialog.

    Implementations document their fields and units. None is unknown. Building
    does not change the inputs or their source; every build returns a fresh model.
    """

    def missing(self) -> tuple[str, ...]:
        """Return unknown field names in declaration order, without side effects."""
        ...

    def build_predictor(self) -> Predictor:
        """Build a fresh query model; PredictorInputError names missing or invalid inputs."""
        ...


def parameter_number(values: Mapping[str, object], name: str) -> float | None:
    """Decode one nullable finite numeric field; reject bool/non-numbers with PredictorInputError.

    A missing key or null is unknown. Does not mutate the mapping.
    """
    value = values.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictorInputError(f"{name} must be a number or null")
    try:
        number = float(value)
    except OverflowError as exc:
        raise PredictorInputError(f"{name} must be finite") from exc
    if not isfinite(number):
        raise PredictorInputError(f"{name} must be finite")
    return number
