from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, fields
from math import isfinite
from typing import overload

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar

from zcu_tools.simulate.fluxonium.prediction import FluxoniumPrediction, MatrixOperator


@dataclass(frozen=True)
class FluxoniumParams:
    """Independent fluxonium model inputs; None means unknown, never a default.

    EJ, EC and EL are positive finite Josephson, charging and inductive energies
    in GHz. flux_half is the zero-bias half-flux device value; flux_period is
    the finite nonzero device-value change per flux quantum (either sign).
    flux_bias is the finite additive correction to the device value. All three
    affine inputs use the source's native device units. Instances are immutable;
    creating or completing one never changes its component or file source.
    """

    EJ: float | None = None
    EC: float | None = None
    EL: float | None = None
    flux_half: float | None = None
    flux_period: float | None = None
    flux_bias: float | None = None

    def missing(self) -> tuple[str, ...]:
        """Return unknown field names in declaration order, without side effects."""
        return tuple(field.name for field in fields(self) if getattr(self, field.name) is None)

    def build_predictor(self) -> FluxoniumPredictor:
        """Build a fresh MHz-query model, without modifying these parameters.

        ValueError names missing fields, nonfinite inputs, nonpositive energies
        or a zero period. No physical or affine value is filled implicitly.
        """
        missing = self.missing()
        if missing:
            raise ValueError("Missing parameters: " + ", ".join(missing))
        ej, ec, el = self.EJ, self.EC, self.EL
        half, period, bias = self.flux_half, self.flux_period, self.flux_bias
        # Explicit narrowing also keeps the runtime contract valid under python -O.
        if ej is None or ec is None or el is None or half is None or period is None or bias is None:
            raise ValueError("Missing parameters")
        for name, value in (("EJ", ej), ("EC", ec), ("EL", el), ("flux_half", half), ("flux_period", period), ("flux_bias", bias)):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name in ("EJ", "EC", "EL") and value <= 0:
                raise ValueError(f"{name} must be positive")
        if period == 0:
            raise ValueError("flux_period must be nonzero")
        return FluxoniumPredictor((ej, ec, el), half, period, bias)


class FluxoniumPredictor:
    """Fluxonium queries with GHz energies and native device-value affine inputs.

    params is (EJ, EC, EL); flux_half is the zero-bias half-flux value,
    flux_period is the nonzero signed device-value period and flux_bias is an
    additive device-value correction. Queries do not mutate the model. Legacy
    setters/update_bias change only this instance; clone builds an independent one.
    Frequencies are MHz and matrix magnitudes are dimensionless.
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

    @property
    def params(self) -> tuple[float, float, float]:
        return self._engine.params

    @params.setter
    def params(self, new_params: tuple[float, float, float]) -> None:
        self._engine.params = new_params

    @property
    def flux_half(self) -> float:
        return self._engine.flux_half

    @flux_half.setter
    def flux_half(self, new_flux_half: float) -> None:
        self._engine.flux_half = new_flux_half

    @property
    def flux_period(self) -> float:
        return self._engine.flux_period

    @flux_period.setter
    def flux_period(self, new_flux_period: float) -> None:
        self._engine.flux_period = new_flux_period

    @property
    def flux_bias(self) -> float:
        return self._engine.flux_bias

    @flux_bias.setter
    def flux_bias(self, new_flux_bias: float) -> None:
        self._engine.flux_bias = new_flux_bias

    @classmethod
    def from_file(cls, result_path: str, flux_bias: float = 0.0) -> FluxoniumPredictor:
        from zcu_tools.resources.qubit_params import QubitParams

        model = QubitParams(result_path, readonly=True).require_fluxonium_model(
            flux_bias=flux_bias
        )
        return cls(model.params, model.flux_half, model.flux_period, model.flux_bias)

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
        # Periodic equivalents shift phi0 by an integer n, giving phi0 + n.
        # Mirror equivalents reflect phi0 and shift by n, giving 1 - phi0 + n.
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
        return min(candidate_biases, key=lambda b: abs(b))

    def update_bias(self, flux_bias: float) -> None:
        self._engine = FluxoniumPrediction(
            self.params,
            flux_half=self.flux_half,
            flux_period=self.flux_period,
            flux_bias=flux_bias,
        )

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

    def predict_frequencies_mhz(
        self,
        values: NDArray[np.float64],
        transitions: tuple[tuple[int, int], ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return flux coordinates and MHz rows for an ordered batch of transitions."""
        return self._engine.predict_frequencies_mhz(values, transitions, cutoff=40)

    def predict_matrix_elements(
        self,
        values: NDArray[np.float64],
        transitions: tuple[tuple[int, int], ...],
        operator: MatrixOperator,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return magnitude rows, including transitions above level one."""
        return self._engine.predict_matrix_elements(values, transitions, operator)

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
