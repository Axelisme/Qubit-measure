from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import overload

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar

from zcu_tools.simulate.fluxonium.prediction import FluxoniumPrediction, MatrixOperator


class FluxoniumPredictor:
    """
    Provide some methods to predict hyper-parameters of fluxonium measurement.
    """

    def __init__(
        self,
        params: tuple[float, float, float],
        flux_half: float,
        flux_period: float,
        flux_bias: float,
    ) -> None:
        self._engine = FluxoniumPrediction(
            params,
            flux_half=flux_half,
            flux_period=flux_period,
            flux_bias=flux_bias,
        )
        self.params = self._engine.params
        self.flux_half = self._engine.affine.flux_half
        self.flux_period = self._engine.affine.flux_period
        self.flux_bias = self._engine.affine.flux_bias
        self._affine = self._engine.affine

    @classmethod
    def from_file(cls, result_path: str, flux_bias: float = 0.0) -> FluxoniumPredictor:
        from zcu_tools.notebook.persistance import load_result

        result_dict = load_result(result_path)
        fluxdepfit_dict = result_dict.get("fluxdep_fit")
        if fluxdepfit_dict is None:
            raise ValueError(
                "fluxdep_fit result is required to create FluxoniumPredictor"
            )
        params = (
            fluxdepfit_dict["params"]["EJ"],
            fluxdepfit_dict["params"]["EC"],
            fluxdepfit_dict["params"]["EL"],
        )
        flux_half = fluxdepfit_dict["flux_half"]
        flux_period = fluxdepfit_dict["flux_period"]

        return cls(params, flux_half, flux_period, flux_bias)

    def clone(self) -> FluxoniumPredictor:
        return FluxoniumPredictor(
            self.params, self.flux_half, self.flux_period, self.flux_bias
        )

    def value_to_flux(self, cur_value: float) -> float:
        return self._engine.value_to_flux(cur_value)

    def _value_to_flux_array(
        self, cur_values: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self._engine.values_to_flux(cur_values)

    def flux_to_value(self, cur_flux: float) -> float:
        return self._engine.flux_to_value(cur_flux)

    def calculate_bias(
        self, cur_value: float, cur_freq: float, transition: tuple[int, int] = (0, 1)
    ) -> float:
        """
        Calibrate the flux_half of the fluxonium qubit by a given current and frequency.

        This method finds a bias such that the predicted frequency matches cur_freq,
        and among all equivalent solutions (periodic and mirror symmetry), returns
        the one with the smallest |bias|.

        Args:
            cur_value (float): Current value.
            cur_freq (float): Frequency in MHz.
            transition (tuple[int, int]): transition between which level
        Returns:
            float: fitting bias value with minimum absolute value.
        """

        def freq_diff_func(value: float) -> float:
            return self._predict_freq(value, transition) - cur_freq

        # Step 1: Use root_scalar to find one valid fit_A
        bracket = [
            cur_value - 0.25 * self.flux_period,
            cur_value + 0.25 * self.flux_period,
        ]
        fit_value = cur_value
        try:
            result = root_scalar(
                freq_diff_func,
                x0=cur_value,
                x1=cur_value + 0.1 * self.flux_period,
                method="secant",
                xtol=1e-5 * self.flux_period,
                maxiter=100,
            )
            if result.converged and bracket[0] <= result.root <= bracket[1]:
                fit_value = result.root
            else:
                warnings.warn(
                    "Bias calibration did not converge inside the local search window; "
                    "using the current value as the fallback.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        except Exception as exc:
            warnings.warn(
                "Bias calibration failed; using the current value as the fallback: "
                f"{exc}",
                RuntimeWarning,
                stacklevel=2,
            )

        # Step 2: Compute initial bias and corresponding flux
        bias0 = fit_value - cur_value + self.flux_bias
        phi0 = (cur_value + bias0 - self.flux_half) / self.flux_period + 0.5

        # Step 3: Enumerate equivalent flux candidates (periodic + mirror symmetry)
        # Periodic: phi0 + n
        # Mirror:   1 - phi0 + n
        N = 2  # number of periods to consider in each direction
        candidate_fluxes = []
        for n in range(-N, N + 1):
            candidate_fluxes.append(phi0 + n)  # periodic equivalents
            candidate_fluxes.append(1 - phi0 + n)  # mirror equivalents

        # Step 4: Convert each candidate flux to candidate bias
        # From value_to_flux: flux = (cur_value + bias - flux_half) / flux_period + 0.5
        # Solve for bias: bias = (flux - 0.5) * flux_period + flux_half - cur_value
        candidate_biases = [
            (phi - 0.5) * self.flux_period + self.flux_half - cur_value
            for phi in candidate_fluxes
        ]

        # Step 5: Pick the candidate with minimum |bias|
        best_bias = min(candidate_biases, key=lambda b: abs(b))

        return best_bias

    def update_bias(self, flux_bias: float) -> None:
        self._engine = FluxoniumPrediction(
            self.params,
            flux_half=self.flux_half,
            flux_period=self.flux_period,
            flux_bias=flux_bias,
        )
        self.flux_bias = self._engine.affine.flux_bias
        self._affine = self._engine.affine

    def _predict_freq(self, cur_value: float, transition: tuple[int, int]) -> float:
        values = np.array([cur_value], dtype=np.float64)
        freq = self._predict_freq_array(values, transition)
        return float(freq[0])

    @overload
    def predict_freq(
        self, cur_value: float, transition: tuple[int, int] = (0, 1)
    ) -> float: ...

    @overload
    def predict_freq(
        self, cur_value: NDArray[np.float64], transition: tuple[int, int] = (0, 1)
    ) -> NDArray[np.float64]: ...

    def predict_freq(
        self,
        cur_value: float | NDArray[np.float64],
        transition: tuple[int, int] = (0, 1),
    ) -> float | NDArray[np.float64]:
        """
        Predict the transition frequency of a fluxonium qubit.
        Args:
            cur_value (float): Current value.
            transition (tuple[from, to]): transition between which level
        Returns:
            float: transition frequency in MHz.
        """
        if isinstance(cur_value, Iterable):
            return self._predict_freq_array(np.asarray(cur_value), transition)
        return self._predict_freq(cur_value, transition)

    def _predict_freq_array(
        self,
        cur_values: NDArray[np.float64],
        transition: tuple[int, int],
    ) -> NDArray[np.float64]:
        """Predict one transition over a value array through the shared engine."""

        engine_transition, sign = _frequency_engine_transition(transition)
        _, freqs_mhz = self._engine.predict_frequencies_mhz(
            np.asarray(cur_values, dtype=np.float64),
            (engine_transition,),
            cutoff=40,
        )
        return np.asarray(sign * freqs_mhz[0], dtype=np.float64)

    def _predict_matrix_element(
        self,
        cur_value: float,
        transition: tuple[int, int],
        operator: MatrixOperator,
    ) -> float:
        values = np.array([cur_value], dtype=np.float64)
        elems = self._predict_matrix_element_array(values, transition, operator)
        return float(elems[0])

    @overload
    def predict_matrix_element(
        self,
        cur_value: float,
        transition: tuple[int, int] = (0, 1),
        operator: MatrixOperator = "n",
    ) -> float: ...

    @overload
    def predict_matrix_element(
        self,
        cur_value: NDArray[np.float64],
        transition: tuple[int, int] = (0, 1),
        operator: MatrixOperator = "n",
    ) -> NDArray[np.float64]: ...

    def predict_matrix_element(
        self,
        cur_value: float | NDArray[np.float64],
        transition: tuple[int, int] = (0, 1),
        operator: MatrixOperator = "n",
    ) -> float | NDArray[np.float64]:
        """
        Predict the matrix element of operator between two levels of a fluxonium qubit.
        Args:
            cur_value (float): Current value.
            transition (tuple[from, to]): transition between which level
            operator (str): 'phi' or 'n'
        Returns:
            float: matrix element of operator between two levels.
        """
        if isinstance(cur_value, Iterable):
            return self._predict_matrix_element_array(
                np.asarray(cur_value), transition, operator
            )
        return self._predict_matrix_element(cur_value, transition, operator)

    def _predict_matrix_element_array(
        self,
        cur_values: NDArray[np.float64],
        transition: tuple[int, int],
        operator: MatrixOperator,
    ) -> NDArray[np.float64]:
        """Predict one 0/1 matrix-element transition through the shared engine."""

        engine_transition = _matrix_engine_transition(transition)
        _, elems = self._engine.predict_matrix_elements(
            np.asarray(cur_values, dtype=np.float64),
            (engine_transition,),
            operator,
        )
        return np.asarray(elems[0], dtype=np.float64)


def _frequency_engine_transition(
    transition: tuple[int, int],
) -> tuple[tuple[int, int], float]:
    frm, to = transition
    if frm < 0 or to < 0:
        raise ValueError(f"Transition levels must be >= 0, got {transition}")
    if to >= frm:
        return transition, 1.0
    return (to, frm), -1.0


def _matrix_engine_transition(transition: tuple[int, int]) -> tuple[int, int]:
    frm, to = transition
    if frm < 0 or to < 0:
        raise ValueError(f"Transition levels must be >= 0, got {transition}")
    if frm > 1 or to > 1:
        raise ValueError(
            "FluxoniumPredictor matrix elements support only levels 0 and 1; "
            f"got {transition}"
        )
    if to >= frm:
        return transition
    return (to, frm)
