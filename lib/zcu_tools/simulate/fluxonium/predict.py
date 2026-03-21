from __future__ import annotations

import json
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # =========================================================================
    # Simulation: Demonstrate that calculate_bias picks the minimum |bias|
    # even when root_scalar might converge to a symmetric branch.
    # =========================================================================

    # Typical fluxonium parameters (EJ, EC, EL) in GHz
    params = (4.0, 1.0, 0.5)  # example values
    flx_half = 0.0  # half-flux point in A
    flx_period = 1e-3  # 1 mA period
    true_bias = 0.15e-3  # 0.15 mA true bias to simulate drift

    print("Creating FluxoniumPredictor...")
    print(f"  params (EJ, EC, EL) = {params} GHz")
    print(
        f"  flx_half = {flx_half * 1e3:.3f} mA, flx_period = {flx_period * 1e3:.3f} mA"
    )
    print(f"  True bias = {true_bias * 1e3:.3f} mA")

    # Create predictor with bias=0 (uncalibrated)
    predictor = FluxoniumPredictor(params, flx_half, flx_period, flx_bias=0.0)

    # Create a "truth" predictor with the true bias (simulates real qubit)
    truth_predictor = FluxoniumPredictor(
        params, flx_half, flx_period, flx_bias=true_bias
    )

    # Sweep cur_A values around half-flux point
    cur_values = np.linspace(-0.4e-3, 0.4e-3, 21)

    calculated_biases = []
    calculated_fluxes = []
    true_fluxes = []

    print("\nRunning bias calculation sweep...")
    for cur_value in cur_values:
        # Get the "measured" frequency from the truth predictor
        cur_freq = truth_predictor.predict_freq(cur_value)

        # Calculate bias using the uncalibrated predictor
        calc_bias = predictor.calculate_bias(cur_value, cur_freq)
        calculated_biases.append(calc_bias)

        # Compute the flux that would result from this calculated bias
        calc_flux = (cur_value + calc_bias - flx_half) / flx_period + 0.5
        calculated_fluxes.append(calc_flux)

        # True flux (what the measurement actually corresponds to)
        true_flux = (cur_value + true_bias - flx_half) / flx_period + 0.5
        true_fluxes.append(true_flux)

    calculated_biases = np.array(calculated_biases)
    calculated_fluxes = np.array(calculated_fluxes)
    true_fluxes = np.array(true_fluxes)

    # =========================================================================
    # Plotting
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Plot 1: Calculated bias vs cur_value
    ax1 = axes[0, 0]
    ax1.plot(cur_values * 1e3, calculated_biases * 1e3, "o-", label="Calculated bias")
    ax1.axhline(
        true_bias * 1e3,
        color="r",
        linestyle="--",
        label=f"True bias = {true_bias * 1e3:.3f} mA",
    )
    ax1.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("cur_value (mA)")
    ax1.set_ylabel("Calculated bias (mA)")
    ax1.set_title("Calculated Bias vs Current")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bias error (calculated - true)
    ax2 = axes[0, 1]
    bias_error = (calculated_biases - true_bias) * 1e3
    ax2.plot(cur_values * 1e3, bias_error, "o-", color="orange")
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_xlabel("cur_value (mA)")
    ax2.set_ylabel("Bias error (mA)")
    ax2.set_title("Bias Error (Calculated - True)")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Calculated flux vs True flux
    ax3 = axes[1, 0]
    ax3.plot(cur_values * 1e3, true_fluxes, "s-", label="True flux", alpha=0.7)
    ax3.plot(cur_values * 1e3, calculated_fluxes, "o--", label="Calc flux", alpha=0.7)
    ax3.axhline(0.5, color="r", linestyle=":", label="Half-flux (0.5)")
    ax3.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax3.axhline(1, color="gray", linestyle=":", alpha=0.5)
    ax3.set_xlabel("cur_value (mA)")
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
