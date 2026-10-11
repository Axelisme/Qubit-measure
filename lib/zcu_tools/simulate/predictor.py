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
    nonnumeric/bool/nonfinite inputs, invalid energies/asymmetry and zero period.
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

    flux_bias is the additive native device correction with minimum absolute value.
    frequency_mhz is the actual transition frequency at that bias. clamped is true
    only for an unreachable target; then frequency_mhz is globally closest to it,
    with equal-frequency choices resolved by smallest absolute bias.
    """

    flux_bias: float
    frequency_mhz: float
    clamped: bool


class Predictor(Protocol):
    """Read-only model queries with native device values and MHz frequencies.

    The affine is flux = (value + flux_bias - flux_half)/flux_period + 0.5.
    Flux coordinates are in flux quanta, not folded into a single period.
    Batch inputs are finite one-dimensional arrays; transitions are a nonempty
    ordered tuple of nonnegative integer (from, to) levels with to >= from.
    Invalid transitions raise PredictorInputError; numerical and array-shape failures
    propagate from the selected engine.
    No query changes parameters or inputs, reads hardware or writes storage.
    """

    @property
    def flux_half(self) -> float:
        """Zero-bias device value at half a flux quantum; native device units."""
        ...

    @property
    def flux_period(self) -> float:
        """Signed nonzero device-value change for one flux quantum."""
        ...

    @property
    def flux_bias(self) -> float:
        """Additive correction to a device value, in the same native units."""
        ...

    def value_to_flux(self, cur_value: float) -> float:
        """Map a finite native device value to flux quanta, without folding."""
        ...

    def flux_to_value(self, cur_flux: float) -> float:
        """Invert the affine for finite flux quanta; return native device units."""
        ...

    def predict_freq(
        self, cur_value: float, transition: tuple[int, int] = (0, 1)
    ) -> float:
        """Return E_to - E_from in MHz at a finite native device value.

        Levels are nonnegative integers; equal levels return zero. Negative
        levels raise PredictorInputError. Engine failures propagate without state changes.
        """
        ...

    def predict_frequencies_mhz(
        self,
        values: NDArray[np.float64],
        transitions: tuple[tuple[int, int], ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (fluxs, frequencies) for native values and ordered transitions.

        fluxs has shape (n_values,) in flux quanta; frequencies has shape
        (n_transitions, n_values) in MHz. Rows preserve transition order, columns
        preserve input order including duplicates. See class input/error contract.
        """
        ...

    def calculate_bias(
        self, cur_value: float, cur_freq: float, transition: tuple[int, int] = (0, 1)
    ) -> CalibrationResult:
        """Invert the model spectrum; return bias, actual MHz and clamped status.

        Finite measured native value and MHz frequency are required; transition
        has nonnegative integer levels with from < to. All roots participate in
        smallest-|bias| selection, irrespective of the model's current bias.
        Unreachable targets return the global nearest frequency (clamped=true);
        ties choose smallest |bias|. Reachable targets have clamped=false.
        Invalid input raises ValueError; numerical failures propagate. No query
        installs the result or changes any input. Frequency equality tolerance
        is 1e-7 MHz; branch discovery uses a 401-point half-period sweep and
        exact-spectrum root/extremum refinement, without assuming monotonicity.
        """
        ...


@runtime_checkable
class MatrixPredictor(Predictor, Protocol):
    """Optional matrix curves; frequency-only models need not implement this."""

    def predict_matrix_elements(
        self,
        values: NDArray[np.float64],
        transitions: tuple[tuple[int, int], ...],
        operator: Literal["n", "phi"],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (fluxs, magnitudes) for native values and ordered transitions.

        fluxs has shape (n_values,) in flux quanta; dimensionless absolute
        |<from|operator|to>| rows have shape (n_transitions, n_values).
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
