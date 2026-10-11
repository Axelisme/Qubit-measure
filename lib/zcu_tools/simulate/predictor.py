"""Storage-independent simulator contracts; all queries are side-effect free."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


class Predictor(Protocol):
    """Read-only model queries with native device values and MHz frequencies.

    The affine is flux = (value + flux_bias - flux_half)/flux_period + 0.5.
    Flux coordinates are in flux quanta, not folded into a single period.
    Batch inputs are finite one-dimensional arrays; transitions are a nonempty
    ordered tuple of nonnegative integer (from, to) levels with to >= from.
    Invalid shapes/transitions raise ValueError; numerical engine errors propagate.
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
        levels raise ValueError. Engine failures propagate without state changes.
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
    ) -> float:
        """Return a candidate additive bias in native device units; never install it.

        cur_value is the finite measured device value; cur_freq is the finite
        measured MHz frequency. transition has nonnegative integer levels with
        from < to. Fluxonium searches a local quarter-period window, then picks
        the smallest absolute periodic/mirror-equivalent bias among nearby
        periods. Failed/nonlocal root searches warn with RuntimeWarning and
        use cur_value as the fallback root, not as proof of a successful fit.
        Numerical failures during the root search also warn and fall back.
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
        Unknown operators or invalid batch inputs raise ValueError; see Predictor.
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
        """Build a fresh query model; ValueError names missing or invalid inputs."""
        ...
