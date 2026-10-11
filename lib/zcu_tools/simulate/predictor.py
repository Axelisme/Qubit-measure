"""Neutral qubit prediction queries shared by simulation-backed models.

Values use native device units, returned flux coordinates use flux quanta,
frequencies are MHz, and batch rows follow the supplied transition order.
Queries do not change the model.
Matrix-element curves are an optional capability; frequency and calibration
clients need only Predictor.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


class Predictor(Protocol):
    """Read-only frequency and bias-calibration surface, independent of storage."""

    @property
    def flux_half(self) -> float: ...
    @property
    def flux_period(self) -> float: ...
    @property
    def flux_bias(self) -> float: ...

    def value_to_flux(self, cur_value: float) -> float: ...
    def flux_to_value(self, cur_flux: float) -> float: ...
    def predict_freq(
        self, cur_value: float, transition: tuple[int, int] = (0, 1)
    ) -> float: ...
    def predict_frequencies_mhz(
        self,
        values: NDArray[np.float64],
        transitions: tuple[tuple[int, int], ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def calculate_bias(
        self, cur_value: float, cur_freq: float, transition: tuple[int, int] = (0, 1)
    ) -> float: ...


@runtime_checkable
class MatrixPredictor(Predictor, Protocol):
    """Optional dimensionless matrix-element magnitude curves."""

    def predict_matrix_elements(
        self,
        values: NDArray[np.float64],
        transitions: tuple[tuple[int, int], ...],
        operator: Literal["n", "phi"],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
