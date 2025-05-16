import json
from typing import Optional

import numpy as np
from scqubits import Fluxonium


class FluxoniumPredictor:
    """
    Provide some methods to predict hyper-parameters of fluxonium measurement.
    """

    def __init__(self, result_path: str, mA_c: Optional[float] = None) -> None:
        with open(result_path, "r") as f:
            data = json.load(f)
        self.params = np.array(
            [data["params"]["EJ"], data["params"]["EC"], data["params"]["EL"]]
        )
        self.mA_c = data["half flux"]
        self.period = data["period"]

        if mA_c is not None:
            self.mA_c = mA_c

        self.fluxonium = Fluxonium(*self.params, flux=0.5, cutoff=40, truncated_dim=2)

    @staticmethod
    def _A_to_flx(cur_A: float, mA_c: float, period: float) -> float:
        return (1e3 * cur_A - mA_c) / period + 0.5

    def predict_f01(self, cur_A: float) -> float:
        """
        Predict the 0-1 transition frequency of a fluxonium qubit.
        Args:
            cur_A (float): Current in A.
        Returns:
            float: 0-1 transition frequency in MHz.
        """

        flx = self._A_to_flx(cur_A, self.mA_c, self.period)

        self.fluxonium.flux = flx
        energies = self.fluxonium.eigenvals(evals_count=2)

        return float(energies[1] - energies[0]) * 1e3  # MHz

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
        flx = self._A_to_flx(cur_A, self.mA_c, self.period)
        ref_flx = self._A_to_flx(ref_A, self.mA_c, self.period)

        self.fluxonium.flux = flx
        n_oper = self.fluxonium.n_operator(energy_esys=True)
        m01 = np.abs(n_oper[0, 1])

        self.fluxonium.flux = ref_flx
        ref_n_oper = self.fluxonium.n_operator(energy_esys=True)
        ref_m01 = np.abs(ref_n_oper[0, 1])

        # the length of pi pulse is inversely proportional to matrix element of capacitance operator
        # so we need to calculate the ratio of matrix element at cur_A and ref_A
        return ref_pilen * ref_m01 / m01
