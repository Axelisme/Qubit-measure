from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar
from typing_extensions import Literal, Union, overload


class FluxoniumPredictor:
    """
    Provide some methods to predict hyper-parameters of fluxonium measurement.
    """

    def __init__(
        self,
        params: tuple[float, float, float],
        flx_half: float,
        flx_period: float,
        flx_bias: float,
    ) -> None:
        self.params = params
        self.flx_half = flx_half
        self.flx_period = flx_period

        self.flx_bias = flx_bias

        from scqubits.core.fluxonium import Fluxonium  # lazy import

        self.fluxonium = Fluxonium(*self.params, flux=0.5, cutoff=40, truncated_dim=2)

    @classmethod
    def from_file(cls, result_path: str, flx_bias: float = 0.0) -> FluxoniumPredictor:
        from zcu_tools.notebook.persistance import load_result

        result_dict = load_result(result_path)
        fluxdepfit_dict = result_dict.get("fluxdep_fit")
        assert fluxdepfit_dict is not None, (
            "fluxdep_fit result is required to create FluxoniumPredictor"
        )
        params = (
            fluxdepfit_dict["params"]["EJ"],
            fluxdepfit_dict["params"]["EC"],
            fluxdepfit_dict["params"]["EL"],
        )
        flx_half = fluxdepfit_dict["flx_half"]
        flx_period = fluxdepfit_dict["flx_period"]

        return cls(params, flx_half, flx_period, flx_bias)

    def clone(self) -> FluxoniumPredictor:
        return FluxoniumPredictor(
            self.params, self.flx_half, self.flx_period, self.flx_bias
        )

    def value_to_flx(self, cur_value: float) -> float:
        return (cur_value + self.flx_bias - self.flx_half) / self.flx_period + 0.5

    def flx_to_value(self, cur_flx: float) -> float:
        return (cur_flx - 0.5) * self.flx_period + self.flx_half - self.flx_bias

    def calculate_bias(
        self, cur_value: float, cur_freq: float, transition: tuple[int, int] = (0, 1)
    ) -> float:
        """
        Calibrate the flx_half of the fluxonium qubit by a given current and frequency.

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
        try:
            result = root_scalar(
                freq_diff_func,
                x0=cur_value,
                x1=cur_value + 0.1 * self.flx_period,
                method="secant",
                bracket=[
                    cur_value - 0.25 * self.flx_period,
                    cur_value + 0.25 * self.flx_period,
                ],
                xtol=1e-5 * self.flx_period,
                maxiter=100,
            )
            if result.converged:
                fit_value = result.root
            else:
                fit_value = cur_value
        except Exception:
            fit_value = cur_value

        # Step 2: Compute initial bias and corresponding flux
        bias0 = fit_value - cur_value + self.flx_bias
        phi0 = (cur_value + bias0 - self.flx_half) / self.flx_period + 0.5

        # Step 3: Enumerate equivalent flux candidates (periodic + mirror symmetry)
        # Periodic: phi0 + n
        # Mirror:   1 - phi0 + n
        N = 2  # number of periods to consider in each direction
        candidate_fluxes = []
        for n in range(-N, N + 1):
            candidate_fluxes.append(phi0 + n)  # periodic equivalents
            candidate_fluxes.append(1 - phi0 + n)  # mirror equivalents

        # Step 4: Convert each candidate flux to candidate bias
        # From value_to_flx: flx = (cur_value + bias - flx_half) / flx_period + 0.5
        # Solve for bias: bias = (flx - 0.5) * flx_period + flx_half - cur_value
        candidate_biases = [
            (phi - 0.5) * self.flx_period + self.flx_half - cur_value
            for phi in candidate_fluxes
        ]

        # Step 5: Pick the candidate with minimum |bias|
        best_bias = min(candidate_biases, key=lambda b: abs(b))

        return best_bias

    def update_bias(self, flx_bias: float) -> None:
        self.flx_bias = flx_bias

    def _predict_freq(self, cur_value: float, transition: tuple[int, int]) -> float:
        flux = self.value_to_flx(cur_value)

        self.fluxonium.flux = flux
        energies = self.fluxonium.eigenvals(evals_count=max(*transition) + 5)

        return float(energies[transition[1]] - energies[transition[0]]) * 1e3  # MHz

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
        cur_value: Union[float, NDArray[np.float64]],
        transition: tuple[int, int] = (0, 1),
    ) -> Union[float, NDArray[np.float64]]:
        """
        Predict the transition frequency of a fluxonium qubit.
        Args:
            cur_value (float): Current value.
            transition (tuple[from, to]): transition between which level
        Returns:
            float: transition frequency in MHz.
        """
        if isinstance(cur_value, Iterable):
            return np.array([self._predict_freq(ca, transition) for ca in cur_value])
        return self._predict_freq(cur_value, transition)

    def _predict_matrix_element(
        self,
        cur_value: float,
        transition: tuple[int, int],
        operator: Literal["phi", "n"],
    ) -> float:
        flux = self.value_to_flx(cur_value)

        self.fluxonium.flux = flux
        if operator == "n":
            oper = self.fluxonium.n_operator(energy_esys=True)
        elif operator == "phi":
            oper = self.fluxonium.phi_operator(energy_esys=True)

        return float(np.abs(oper[transition[0], transition[1]]))

    @overload
    def predict_matrix_element(
        self,
        cur_value: float,
        transition: tuple[int, int] = (0, 1),
        operator: Literal["phi", "n"] = "n",
    ) -> float: ...

    @overload
    def predict_matrix_element(
        self,
        cur_value: NDArray[np.float64],
        transition: tuple[int, int] = (0, 1),
        operator: Literal["phi", "n"] = "n",
    ) -> NDArray[np.float64]: ...

    def predict_matrix_element(
        self,
        cur_value: Union[float, NDArray[np.float64]],
        transition: tuple[int, int] = (0, 1),
        operator: Literal["phi", "n"] = "n",
    ) -> Union[float, NDArray[np.float64]]:
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
            return np.array(
                [
                    self._predict_matrix_element(ca, transition, operator)
                    for ca in cur_value
                ],
                dtype=np.float64,
            )
        return self._predict_matrix_element(cur_value, transition, operator)
