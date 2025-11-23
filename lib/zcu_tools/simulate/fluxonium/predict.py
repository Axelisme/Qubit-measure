from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Literal, Tuple

import numpy as np
from scipy.optimize import root_scalar


class FluxoniumPredictor:
    """
    Provide some methods to predict hyper-parameters of fluxonium measurement.
    """

    def __init__(self, result_path: str, bias: float = 0.0) -> None:
        with open(result_path, "r") as f:
            data = json.load(f)

        self.result_path = result_path
        self.params = np.array(
            [data["params"]["EJ"], data["params"]["EC"], data["params"]["EL"]]
        )
        self.A_c = data["half flux"]
        self.period = data["period"]

        self.bias = bias

        from scqubits import Fluxonium  # lazy import

        self.fluxonium = Fluxonium(*self.params, flux=0.5, cutoff=40, truncated_dim=2)

    def clone(self) -> FluxoniumPredictor:
        return FluxoniumPredictor(self.result_path, bias=self.bias)

    def A_to_flx(self, cur_A: float) -> float:
        return (cur_A + self.bias - self.A_c) / self.period + 0.5

    def flx_to_A(self, cur_flx: float) -> float:
        return (cur_flx - 0.5) * self.period + self.A_c - self.bias

    def calculate_bias(
        self, cur_A: float, cur_freq: float, transition: Tuple[int, int] = (0, 1)
    ) -> float:
        """
        Calibrate the mA_c of the fluxonium qubit by a given current and frequency.
        Args:
            cur_A (float): Current in A.
            cur_freq (float): Frequency in MHz.
            transition (Tuple[int, int]): transition between which level
        Returns:
            float: fitting bias current in A (fit_A - cur_A).
        """

        def freq_diff_func(test_A: float) -> float:
            return self._predict_freq(test_A, transition) - cur_freq

        try:
            result = root_scalar(
                freq_diff_func,
                x0=cur_A,
                x1=cur_A + 1e-4,
                method="secant",
                bracket=[cur_A - 1e-3, cur_A + 1e-3],
                xtol=1e-9,
                maxiter=50,
            )
            if result.converged:
                fit_A = result.root
            else:
                fit_A = cur_A
        except Exception:
            fit_A = cur_A

        bias = fit_A - cur_A + self.bias
        return bias

    def update_bias(self, bias: float) -> None:
        self.bias = bias

    def _predict_freq(self, cur_A: float, transition: Tuple[int, int]) -> float:
        flx = self.A_to_flx(cur_A)

        self.fluxonium.flux = flx
        energies = self.fluxonium.eigenvals(evals_count=max(*transition) + 5)

        return float(energies[transition[1]] - energies[transition[0]]) * 1e3  # MHz

    def predict_freq(self, cur_A: float, transition: Tuple[int, int] = (0, 1)) -> float:
        """
        Predict the transition frequency of a fluxonium qubit.
        Args:
            cur_A (float): Current in A.
            transition (Tuple[from, to]): transition between which level
        Returns:
            float: transition frequency in MHz.
        """
        if isinstance(cur_A, Iterable):
            return np.array([self._predict_freq(ca, transition) for ca in cur_A])
        return self._predict_freq(cur_A, transition)

    def _predict_matrix_element(
        self, cur_A: float, transition: Tuple[int, int], operator: Literal["phi", "n"]
    ) -> float:
        flx = self.A_to_flx(cur_A)

        self.fluxonium.flux = flx
        if operator == "n":
            oper = self.fluxonium.n_operator(energy_esys=True)
        elif operator == "phi":
            oper = self.fluxonium.phi_operator(energy_esys=True)

        return float(np.abs(oper[transition[0], transition[1]]))

    def predict_matrix_element(
        self,
        cur_A: float,
        transition: Tuple[int, int] = (0, 1),
        operator: Literal["phi", "n"] = "n",
    ) -> float:
        """
        Predict the matrix element of operator between two levels of a fluxonium qubit.
        Args:
            cur_A (float): Current in A.
            transition (Tuple[from, to]): transition between which level
            operator (str): 'phi' or 'n'
        Returns:
            float: matrix element of operator between two levels.
        """
        if isinstance(cur_A, Iterable):
            return np.array(
                [self._predict_matrix_element(ca, transition, operator) for ca in cur_A]
            )
        return self._predict_matrix_element(cur_A, transition, operator)
