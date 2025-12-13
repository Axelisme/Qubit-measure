from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Literal, Tuple, Union, overload

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar


class FluxoniumPredictor:
    """
    Provide some methods to predict hyper-parameters of fluxonium measurement.
    """

    def __init__(
        self, params: Tuple[float, float, float], A_c: float, period: float, bias: float
    ) -> None:
        self.params = params
        self.A_c = A_c
        self.period = period

        self.bias = bias

        from scqubits.core.fluxonium import Fluxonium  # lazy import

        self.fluxonium = Fluxonium(*self.params, flux=0.5, cutoff=40, truncated_dim=2)

    @classmethod
    def from_file(cls, result_path: str, bias: float = 0.0) -> FluxoniumPredictor:
        with open(result_path, "r") as f:
            data = json.load(f)

            params = (
                data["params"]["EJ"],
                data["params"]["EC"],
                data["params"]["EL"],
            )
            A_c = data["half flux"]
            period = data["period"]

        return cls(params, A_c, period, bias)

    def clone(self) -> FluxoniumPredictor:
        return FluxoniumPredictor(self.params, self.A_c, self.period, self.bias)

    def A_to_flx(self, cur_A: float) -> float:
        return (cur_A + self.bias - self.A_c) / self.period + 0.5

    def flx_to_A(self, cur_flx: float) -> float:
        return (cur_flx - 0.5) * self.period + self.A_c - self.bias

    def calculate_bias(
        self, cur_A: float, cur_freq: float, transition: Tuple[int, int] = (0, 1)
    ) -> float:
        """
        Calibrate the mA_c of the fluxonium qubit by a given current and frequency.

        This method finds a bias such that the predicted frequency matches cur_freq,
        and among all equivalent solutions (periodic and mirror symmetry), returns
        the one with the smallest |bias|.

        Args:
            cur_A (float): Current in A.
            cur_freq (float): Frequency in MHz.
            transition (Tuple[int, int]): transition between which level
        Returns:
            float: fitting bias current in A with minimum absolute value.
        """

        def freq_diff_func(test_A: float) -> float:
            return self._predict_freq(test_A, transition) - cur_freq

        # Step 1: Use root_scalar to find one valid fit_A
        try:
            result = root_scalar(
                freq_diff_func,
                x0=cur_A,
                x1=cur_A + 0.1 * self.period,
                method="secant",
                bracket=[cur_A - 0.25 * self.period, cur_A + 0.25 * self.period],
                xtol=1e-4 * self.period,
                maxiter=100,
            )
            if result.converged:
                fit_A = result.root
            else:
                fit_A = cur_A
        except Exception:
            fit_A = cur_A

        # Step 2: Compute initial bias and corresponding flux
        bias0 = fit_A - cur_A + self.bias
        phi0 = (cur_A + bias0 - self.A_c) / self.period + 0.5

        # Step 3: Enumerate equivalent flux candidates (periodic + mirror symmetry)
        # Periodic: phi0 + n
        # Mirror:   1 - phi0 + n
        N = 2  # number of periods to consider in each direction
        candidate_fluxes = []
        for n in range(-N, N + 1):
            candidate_fluxes.append(phi0 + n)  # periodic equivalents
            candidate_fluxes.append(1 - phi0 + n)  # mirror equivalents

        # Step 4: Convert each candidate flux to candidate bias
        # From A_to_flx: flx = (cur_A + bias - A_c) / period + 0.5
        # Solve for bias: bias = (flx - 0.5) * period + A_c - cur_A
        candidate_biases = [
            (phi - 0.5) * self.period + self.A_c - cur_A for phi in candidate_fluxes
        ]

        # Step 5: Pick the candidate with minimum |bias|
        best_bias = min(candidate_biases, key=lambda b: abs(b))

        return best_bias

    def update_bias(self, bias: float) -> None:
        self.bias = bias

    def _predict_freq(self, cur_A: float, transition: Tuple[int, int]) -> float:
        flx = self.A_to_flx(cur_A)

        self.fluxonium.flux = flx
        energies = self.fluxonium.eigenvals(evals_count=max(*transition) + 5)

        return float(energies[transition[1]] - energies[transition[0]]) * 1e3  # MHz

    @overload
    def predict_freq(
        self, cur_A: float, transition: Tuple[int, int] = (0, 1)
    ) -> float: ...

    @overload
    def predict_freq(
        self, cur_A: NDArray[np.float64], transition: Tuple[int, int] = (0, 1)
    ) -> NDArray[np.float64]: ...

    def predict_freq(
        self,
        cur_A: Union[float, NDArray[np.float64]],
        transition: Tuple[int, int] = (0, 1),
    ) -> Union[float, NDArray[np.float64]]:
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

    @overload
    def predict_matrix_element(
        self,
        cur_A: float,
        transition: Tuple[int, int] = (0, 1),
        operator: Literal["phi", "n"] = "n",
    ) -> float: ...

    @overload
    def predict_matrix_element(
        self,
        cur_A: NDArray[np.float64],
        transition: Tuple[int, int] = (0, 1),
        operator: Literal["phi", "n"] = "n",
    ) -> NDArray[np.float64]: ...

    def predict_matrix_element(
        self,
        cur_A: Union[float, NDArray[np.float64]],
        transition: Tuple[int, int] = (0, 1),
        operator: Literal["phi", "n"] = "n",
    ) -> Union[float, NDArray[np.float64]]:
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
                [
                    self._predict_matrix_element(ca, transition, operator)
                    for ca in cur_A
                ],
                dtype=np.float64,
            )
        return self._predict_matrix_element(cur_A, transition, operator)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # =========================================================================
    # Simulation: Demonstrate that calculate_bias picks the minimum |bias|
    # even when root_scalar might converge to a symmetric branch.
    # =========================================================================

    # Typical fluxonium parameters (EJ, EC, EL) in GHz
    params = (4.0, 1.0, 0.5)  # example values
    A_c = 0.0  # half-flux point in A
    period = 1e-3  # 1 mA period
    true_bias = 0.15e-3  # 0.15 mA true bias to simulate drift

    print("Creating FluxoniumPredictor...")
    print(f"  params (EJ, EC, EL) = {params} GHz")
    print(f"  A_c = {A_c * 1e3:.3f} mA, period = {period * 1e3:.3f} mA")
    print(f"  True bias = {true_bias * 1e3:.3f} mA")

    # Create predictor with bias=0 (uncalibrated)
    predictor = FluxoniumPredictor(params, A_c, period, bias=0.0)

    # Create a "truth" predictor with the true bias (simulates real qubit)
    truth_predictor = FluxoniumPredictor(params, A_c, period, bias=true_bias)

    # Sweep cur_A values around half-flux point
    cur_A_values = np.linspace(-0.4e-3, 0.4e-3, 21)

    calculated_biases = []
    calculated_fluxes = []
    true_fluxes = []

    print("\nRunning bias calculation sweep...")
    for cur_A in cur_A_values:
        # Get the "measured" frequency from the truth predictor
        cur_freq = truth_predictor.predict_freq(cur_A)

        # Calculate bias using the uncalibrated predictor
        calc_bias = predictor.calculate_bias(cur_A, cur_freq)
        calculated_biases.append(calc_bias)

        # Compute the flux that would result from this calculated bias
        calc_flux = (cur_A + calc_bias - A_c) / period + 0.5
        calculated_fluxes.append(calc_flux)

        # True flux (what the measurement actually corresponds to)
        true_flux = (cur_A + true_bias - A_c) / period + 0.5
        true_fluxes.append(true_flux)

    calculated_biases = np.array(calculated_biases)
    calculated_fluxes = np.array(calculated_fluxes)
    true_fluxes = np.array(true_fluxes)

    # =========================================================================
    # Plotting
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Plot 1: Calculated bias vs cur_A
    ax1 = axes[0, 0]
    ax1.plot(cur_A_values * 1e3, calculated_biases * 1e3, "o-", label="Calculated bias")
    ax1.axhline(
        true_bias * 1e3,
        color="r",
        linestyle="--",
        label=f"True bias = {true_bias * 1e3:.3f} mA",
    )
    ax1.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("cur_A (mA)")
    ax1.set_ylabel("Calculated bias (mA)")
    ax1.set_title("Calculated Bias vs Current")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bias error (calculated - true)
    ax2 = axes[0, 1]
    bias_error = (calculated_biases - true_bias) * 1e3
    ax2.plot(cur_A_values * 1e3, bias_error, "o-", color="orange")
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_xlabel("cur_A (mA)")
    ax2.set_ylabel("Bias error (mA)")
    ax2.set_title("Bias Error (Calculated - True)")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Calculated flux vs True flux
    ax3 = axes[1, 0]
    ax3.plot(cur_A_values * 1e3, true_fluxes, "s-", label="True flux", alpha=0.7)
    ax3.plot(cur_A_values * 1e3, calculated_fluxes, "o--", label="Calc flux", alpha=0.7)
    ax3.axhline(0.5, color="r", linestyle=":", label="Half-flux (0.5)")
    ax3.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax3.axhline(1, color="gray", linestyle=":", alpha=0.5)
    ax3.set_xlabel("cur_A (mA)")
    ax3.set_ylabel("Flux (Φ/Φ₀)")
    ax3.set_title("Flux Comparison")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: |bias| histogram to show minimum selection works
    ax4 = axes[1, 1]
    ax4.hist(np.abs(calculated_biases) * 1e3, bins=15, edgecolor="black", alpha=0.7)
    ax4.axvline(
        np.abs(true_bias) * 1e3,
        color="r",
        linestyle="--",
        label=f"|True bias| = {np.abs(true_bias) * 1e3:.3f} mA",
    )
    ax4.set_xlabel("|Calculated bias| (mA)")
    ax4.set_ylabel("Count")
    ax4.set_title("Distribution of |Calculated Bias|")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("bias_simulation_result.png", dpi=150)
    plt.show()

    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  True bias:           {true_bias * 1e3:.4f} mA")
    print(f"  Mean calculated:     {np.mean(calculated_biases) * 1e3:.4f} mA")
    print(f"  Std calculated:      {np.std(calculated_biases) * 1e3:.4f} mA")
    print(f"  Mean |error|:        {np.mean(np.abs(bias_error)):.4f} mA")
    print(f"  Max |error|:         {np.max(np.abs(bias_error)):.4f} mA")
    print("=" * 60)
    print("\nPlot saved to: bias_simulation_result.png")
