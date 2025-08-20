import json
from typing import Tuple

import numpy as np
from scipy.optimize import root_scalar


class FluxoniumPredictor:
    """
    Provide some methods to predict hyper-parameters of fluxonium measurement.
    """

    def __init__(self, result_path: str, bias: float = 0.0) -> None:
        with open(result_path, "r") as f:
            data = json.load(f)
        self.params = np.array(
            [data["params"]["EJ"], data["params"]["EC"], data["params"]["EL"]]
        )
        self.A_c = data["half flux"]
        self.period = data["period"]

        self.bias = bias

        from scqubits import Fluxonium  # lazy import

        self.fluxonium = Fluxonium(*self.params, flux=0.5, cutoff=40, truncated_dim=2)

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

        def freq_diff_func(test_A):
            return self.predict_freq(test_A, transition) - cur_freq

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
        return round(bias, 6)  # 1e-3mA/mV precision

    def update_bias(self, bias: float) -> None:
        self.bias = bias

    def predict_freq(self, cur_A: float, transition: Tuple[int, int] = (0, 1)) -> float:
        """
        Predict the transition frequency of a fluxonium qubit.
        Args:
            cur_A (float): Current in A.
            transition (Tuple[from, to]): transition between which level
        Returns:
            float: transition frequency in MHz.
        """

        flx = self.A_to_flx(cur_A)

        self.fluxonium.flux = flx
        energies = self.fluxonium.eigenvals(evals_count=max(*transition) + 1)

        return float(energies[transition[1]] - energies[transition[0]]) * 1e3  # MHz

    def predict_lenrabi(self, cur_A: float, ref_A: float, ref_pilen: float) -> float:
        """
        Predict the length of Pi pulse for a given current.
        Args:
            cur_A (float): Current in A.
            ref_A (float): Reference current in A.
            ref_len (float): Reference length of Pi pulse in ns.
        Returns:
            float: Length of Pi pulse in ns.
        """
        flx, ref_flx = self.A_to_flx(cur_A), self.A_to_flx(ref_A)

        self.fluxonium.flux = flx
        n_oper = self.fluxonium.n_operator(energy_esys=True)
        m01 = np.abs(n_oper[0, 1])

        self.fluxonium.flux = ref_flx
        ref_n_oper = self.fluxonium.n_operator(energy_esys=True)
        ref_m01 = np.abs(ref_n_oper[0, 1])

        # the length of pi pulse is inversely proportional to matrix element of capacitance operator
        # so we need to calculate the ratio of matrix element at cur_A and ref_A
        return ref_pilen * ref_m01 / m01
