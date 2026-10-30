"""Named fitter outputs with legacy tuple layouts and non-positional quality."""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple, Self

import numpy as np
from numpy.typing import NDArray

from .base import FitParameters, FitResult
from .quality import FitQuality


class _FrozenQuality:
    """Keep diagnostics outside tuple positions without allowing reassignment."""

    _quality: FitQuality

    @property
    def quality(self) -> FitQuality:
        """Dimensionless diagnostics computed from the actual fit observations."""
        return self._quality

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is frozen")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is frozen")

    @classmethod
    def _make(cls, iterable: Iterable[object]) -> Self:
        # Retain NamedTuple's keyword name, but discard values without observations.
        del iterable
        raise TypeError(
            "rerun the fitter to change values and obtain consistent quality"
        )

    def _replace(self, **_changes: object) -> Self:
        raise TypeError(
            "rerun the fitter to change values and obtain consistent quality"
        )


class _DecayFitTuple(NamedTuple):
    decay_time: float
    decay_time_err: float
    curve: NDArray[np.float64]
    fit: FitResult


class DecayFit(_FrozenQuality, _DecayFitTuple):
    """Exponential fit; construct from the legacy four-tuple and quality.

    Copy, deepcopy and pickle preserve all tuple fields and quality. _make and
    _replace raise TypeError: rerun the fitter to change values and obtain
    consistent quality.

    Attributes:
        decay_time: Exponential 1/e time, in input coordinate units.
        decay_time_err: Standard error of decay_time, in coordinate units.
        curve: Model values at input coordinates, in observation units.
        fit: Legacy (parameters, covariance) pair; parameter order is
            (y0, yscale, decay_time). y0/yscale use observation units.
            Covariance entries use the product of the corresponding units.
        params: The parameters in fit, without copying.
        cov: The covariance in fit, without copying.
        quality: Dimensionless FitQuality from actual observations, with
            fixed-parameter and decay-bound diagnostics.

    Fields cannot be reassigned or deleted. Arrays retain their existing
    mutability. Four-value tuple unpacking is transitional compatibility and
    will be removed when callers migrate to named access.
    """

    def __new__(
        cls,
        values: tuple[float, float, NDArray[np.float64], FitResult],
        *,
        quality: FitQuality,
    ) -> Self:
        result = tuple.__new__(cls, values)
        # NamedTuple slots cannot hold quality without changing unpacking arity.
        object.__setattr__(result, "_quality", quality)
        return result

    def __getnewargs_ex__(
        self,
    ) -> tuple[tuple[tuple[object, ...]], dict[str, FitQuality]]:
        return (tuple(self),), {"quality": self.quality}

    @property
    def params(self) -> FitParameters:
        """Offset, amplitude and 1/e time; see fit for units."""
        return self.fit[0]

    @property
    def cov(self) -> NDArray[np.float64]:
        """Parameter covariance in params order, with product units."""
        return self.fit[1]


type DualDecayParameters = FitParameters | tuple[float, float, float, float, float]
type DualDecayValues = tuple[
    float,
    float,
    float,
    float,
    NDArray[np.float64],
    tuple[DualDecayParameters, NDArray[np.float64]],
]


class _DualDecayFitTuple(NamedTuple):
    decay_time1: float
    decay_time1_err: float
    decay_time2: float
    decay_time2_err: float
    curve: NDArray[np.float64]
    fit: tuple[DualDecayParameters, NDArray[np.float64]]


class DualDecayFit(_FrozenQuality, _DualDecayFitTuple):
    """Two-exponential fit; construct from the legacy six-tuple and quality.

    Copy, deepcopy and pickle preserve all tuple fields and quality. _make and
    _replace raise TypeError: rerun the fitter to change values and obtain
    consistent quality.

    Attributes:
        decay_time1: Longer exponential 1/e time, in coordinate units.
        decay_time1_err: Standard error of decay_time1, in coordinate units.
        decay_time2: Shorter exponential 1/e time, in coordinate units.
        decay_time2_err: Standard error of decay_time2, in coordinate units.
        curve: Model values at input coordinates, in observation units.
        fit: Legacy (parameters, covariance) pair, ordered as
            (y0, yscale1, decay_time1, yscale2, decay_time2). y0/yscale1/yscale2
            use observation units; times use coordinate units. Covariance
            entries use products of the corresponding parameter units.
        params: The parameters in fit, without copying.
        cov: The covariance in fit, without copying.
        quality: Dimensionless FitQuality with fixed flags and decay bounds
            following the reordered parameters.

    Fields cannot be reassigned or deleted; arrays retain existing mutability.
    Six-value tuple unpacking is transitional compatibility and will be removed
    when callers migrate to named access.
    """

    def __new__(cls, values: DualDecayValues, *, quality: FitQuality) -> Self:
        result = tuple.__new__(cls, values)
        object.__setattr__(result, "_quality", quality)
        return result

    def __getnewargs_ex__(
        self,
    ) -> tuple[tuple[tuple[object, ...]], dict[str, FitQuality]]:
        return (tuple(self),), {"quality": self.quality}

    @property
    def params(self) -> DualDecayParameters:
        """Offset, two amplitudes and ordered 1/e times; see fit for units."""
        return self.fit[0]

    @property
    def cov(self) -> NDArray[np.float64]:
        """Parameter covariance in params order, with product units."""
        return self.fit[1]


type FringeParameters = tuple[float, float, float, float, float]
type FringeValues = tuple[
    float,
    float,
    float,
    float,
    NDArray[np.float64],
    tuple[FringeParameters, NDArray[np.float64]],
]


class _DecayFringeFitTuple(NamedTuple):
    decay_time: float
    decay_time_err: float
    freq: float
    freq_err: float
    curve: NDArray[np.float64]
    fit: tuple[FringeParameters, NDArray[np.float64]]


class DecayFringeFit(_FrozenQuality, _DecayFringeFitTuple):
    """Damped-cosine fit; construct from the legacy six-tuple and quality.

    Copy, deepcopy and pickle preserve all tuple fields and quality. _make and
    _replace raise TypeError: rerun the fitter to change values and obtain
    consistent quality.

    Attributes:
        decay_time: Envelope 1/e time, in input coordinate units.
        decay_time_err: Standard error of decay_time, in coordinate units.
        freq: Oscillation frequency, in inverse coordinate units.
        freq_err: Standard error of freq, in inverse coordinate units.
        curve: Model values at input coordinates, in observation units.
        fit: Legacy (parameters, covariance) pair, ordered as
            (y0, yscale, freq, phase, decay_time). y0/yscale use observation
            units, phase uses degrees, and time/frequency use coordinate /
            inverse coordinate units. Covariance entries use products of the
            corresponding parameter units.
        params: The parameter tuple in fit, without copying.
        cov: The covariance in fit, without copying.
        quality: Dimensionless FitQuality with fixed-parameter and
            decay-bound diagnostics.

    Fields cannot be reassigned or deleted; arrays retain existing mutability.
    Six-value tuple unpacking is transitional compatibility and will be removed
    when callers migrate to named access.
    """

    def __new__(cls, values: FringeValues, *, quality: FitQuality) -> Self:
        result = tuple.__new__(cls, values)
        object.__setattr__(result, "_quality", quality)
        return result

    def __getnewargs_ex__(
        self,
    ) -> tuple[tuple[tuple[object, ...]], dict[str, FitQuality]]:
        return (tuple(self),), {"quality": self.quality}

    @property
    def params(self) -> FringeParameters:
        """Offset, amplitude, frequency, phase and 1/e time; see fit for units."""
        return self.fit[0]

    @property
    def cov(self) -> NDArray[np.float64]:
        """Parameter covariance in params order, with product units."""
        return self.fit[1]


type RabiValues = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    NDArray[np.float64],
    tuple[tuple[float, ...], NDArray[np.float64]],
]


class _RabiFitTuple(NamedTuple):
    pi_x: float
    pi_x_err: float
    pi2_x: float
    pi2_x_err: float
    freq: float
    freq_err: float
    curve: NDArray[np.float64]
    fit: tuple[tuple[float, ...], NDArray[np.float64]]


class RabiFit(_FrozenQuality, _RabiFitTuple):
    """Rabi cosine fit; construct from the legacy eight-tuple and quality.

    Copy, deepcopy and pickle preserve all tuple fields and quality. _make and
    _replace raise TypeError: rerun the fitter to change values and obtain
    consistent quality.

    Attributes:
        pi_x: Derived pi-pulse coordinate, in input coordinate units.
        pi_x_err: Propagated standard error of pi_x, in coordinate units.
        pi2_x: Derived pi/2-pulse coordinate, in input coordinate units.
        pi2_x_err: Propagated standard error of pi2_x, in coordinate units.
        freq: Oscillation frequency, in inverse coordinate units.
        freq_err: Standard error of freq, in inverse coordinate units.
        curve: Model values at input coordinates, in observation units.
        fit: Legacy (parameters, covariance) pair, with parameter tuple
            (y0, yscale, freq, phase[, decay_time]). y0/yscale use observation
            units, phase uses degrees, and optional decay_time uses coordinate
            units. Covariance entries use products of the corresponding units.
        params: The parameter tuple in fit, without copying.
        cov: The covariance in fit, without copying.
        quality: Dimensionless FitQuality with explicit fixed-phase diagnostics;
            no decay bounds are applied to Rabi fits.

    Fields cannot be reassigned or deleted; arrays retain existing mutability.
    Eight-value tuple unpacking is transitional compatibility and will be removed
    when callers migrate to named access.
    """

    def __new__(cls, values: RabiValues, *, quality: FitQuality) -> Self:
        result = tuple.__new__(cls, values)
        object.__setattr__(result, "_quality", quality)
        return result

    def __getnewargs_ex__(
        self,
    ) -> tuple[tuple[tuple[object, ...]], dict[str, FitQuality]]:
        return (tuple(self),), {"quality": self.quality}

    @property
    def params(self) -> tuple[float, ...]:
        """Offset, amplitude, frequency, phase and optional time; see fit for units."""
        return self.fit[0]

    @property
    def cov(self) -> NDArray[np.float64]:
        """Parameter covariance in params order, with product units."""
        return self.fit[1]
