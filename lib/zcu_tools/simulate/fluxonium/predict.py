from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Union, overload

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar


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
        self.params = params
        self.flux_half = flux_half
        self.flux_period = flux_period

        self.flux_bias = flux_bias

        from scqubits.core.fluxonium import Fluxonium  # lazy import

        self.fluxonium = Fluxonium(*self.params, flux=0.5, cutoff=40, truncated_dim=2)

    @classmethod
    def from_file(cls, result_path: str, flux_bias: float = 0.0) -> FluxoniumPredictor:
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
        flux_half = fluxdepfit_dict["flux_half"]
        flux_period = fluxdepfit_dict["flux_period"]

        return cls(params, flux_half, flux_period, flux_bias)

    def clone(self) -> FluxoniumPredictor:
        return FluxoniumPredictor(
            self.params, self.flux_half, self.flux_period, self.flux_bias
        )

    def value_to_flux(self, cur_value: float) -> float:
        return (cur_value + self.flux_bias - self.flux_half) / self.flux_period + 0.5

    def _value_to_flux_array(
        self, cur_values: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Array form of value_to_flux (same affine, kept in lockstep): a separate
        # method so the scalar value_to_flux keeps its float-only public signature
        # while the batched paths get a typed array transform.
        flux = (cur_values + self.flux_bias - self.flux_half) / self.flux_period + 0.5
        return np.asarray(flux, dtype=np.float64)

    def flux_to_value(self, cur_flux: float) -> float:
        return (cur_flux - 0.5) * self.flux_period + self.flux_half - self.flux_bias

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
        try:
            bracket = [
                cur_value - 0.25 * self.flux_period,
                cur_value + 0.25 * self.flux_period,
            ]
            result = root_scalar(
                freq_diff_func,
                x0=cur_value,
                x1=cur_value + 0.1 * self.flux_period,
                method="secant",
                bracket=bracket,
                xtol=1e-5 * self.flux_period,
                maxiter=100,
            )
            fit_value = result.root
            if not result.converged or fit_value > bracket[1] or fit_value < bracket[0]:
                fit_value = cur_value
        except Exception:
            fit_value = cur_value

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
        self.flux_bias = flux_bias

    def _predict_freq(self, cur_value: float, transition: tuple[int, int]) -> float:
        flux = self.value_to_flux(cur_value)

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
        """Batched ``_predict_freq``: one vectorised flux sweep instead of N
        scalar diagonalisations.

        Reuses ``calculate_energy_vs_flux`` (precomputes the flux-independent
        cos/sin(phi) operators once, then combines per flux), so this does not
        touch ``self.fluxonium`` — the shared scqubits object whose ``.flux`` the
        scalar path mutates. Building a fresh local engine inside the helper keeps
        the batched path free of cross-call shared mutable state (thread-safe),
        unlike the scalar path. ``evals_count`` mirrors the scalar path's
        ``max(*transition) + 5`` so the two stay numerically aligned.
        """
        from zcu_tools.simulate.fluxonium.energies import calculate_energy_vs_flux

        fluxs = self._value_to_flux_array(cur_values)
        evals_count = max(*transition) + 5
        _, energies = calculate_energy_vs_flux(
            self.params, fluxs, cutoff=40, evals_count=evals_count
        )
        diff = energies[:, transition[1]] - energies[:, transition[0]]
        return np.asarray(diff * 1e3, dtype=np.float64)  # MHz

    def _predict_matrix_element(
        self,
        cur_value: float,
        transition: tuple[int, int],
        operator: Literal["phi", "n"],
    ) -> float:
        flux = self.value_to_flux(cur_value)

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
        cur_value: float | NDArray[np.float64],
        transition: tuple[int, int] = (0, 1),
        operator: Literal["phi", "n"] = "n",
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
        operator: Literal["phi", "n"],
    ) -> NDArray[np.float64]:
        """Batched ``_predict_matrix_element``: one vectorised operator sweep.

        Reuses ``calculate_{n,phi}_oper_vs_flux``, which build a local scqubits
        engine and run ``get_matelements_vs_paramvals`` over the whole flux grid,
        so this avoids mutating the shared ``self.fluxonium`` (thread-safe, unlike
        the scalar path).

        Like the scalar path (whose ``self.fluxonium`` has ``truncated_dim=2``)
        this only takes the [0, 1] sub-block via ``return_dim=2``; a ``transition``
        with a level >= 2 indexes past that block and raises, matching the scalar
        path's behaviour. Supporting higher levels is a separate feature.
        """
        from zcu_tools.simulate.fluxonium.matrix_element import (
            calculate_n_oper_vs_flux,
            calculate_phi_oper_vs_flux,
        )

        fluxs = self._value_to_flux_array(cur_values)
        if operator == "n":
            _, opers = calculate_n_oper_vs_flux(self.params, fluxs, return_dim=2)
        else:
            _, opers = calculate_phi_oper_vs_flux(self.params, fluxs, return_dim=2)

        elems = opers[:, transition[0], transition[1]]
        return np.abs(elems).astype(np.float64)
