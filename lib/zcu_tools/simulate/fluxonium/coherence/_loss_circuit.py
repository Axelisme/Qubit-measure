"""Native-operator reuse for one fixed directional-loss circuit.

Imported only inside the directional-rate call, keeping scqubits lazy.
"""

from functools import cached_property
from typing import override

import numpy as np
from numpy.typing import NDArray
from scqubits.core.fluxonium import Fluxonium

Operator = NDArray[np.float64] | NDArray[np.complex128]
Eigenpairs = tuple[NDArray[np.float64], NDArray[np.complex128]]


class DirectionalLossCircuit(Fluxonium):
    """Reuse native operators for a single fixed-circuit directional-rate call.

    Construct as Fluxonium. The coherence owner keeps EJ/EC/EL/cutoff/flux fixed
    and never shares this instance or its arrays with callers. Sines use exact
    alpha/beta keys, without flux folding or trigonometric approximations.
    Energy-basis requests and all T1 formulas still belong to scqubits; parent
    construction and numerical failures propagate unchanged.
    """

    @cached_property
    def _native_n(self) -> Operator:
        return super().n_operator()

    @cached_property
    def _native_phi(self) -> Operator:
        return super().phi_operator()

    @cached_property
    def _native_sines(self) -> dict[tuple[float, float], Operator]:
        return {}

    @override
    def n_operator(self, energy_esys: bool | Eigenpairs = False) -> Operator:
        if energy_esys is False:
            return self._native_n
        return super().n_operator(energy_esys=energy_esys)

    @override
    def phi_operator(self, energy_esys: bool | Eigenpairs = False) -> Operator:
        if energy_esys is False:
            return self._native_phi
        return super().phi_operator(energy_esys=energy_esys)

    @override
    def sin_phi_operator(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        energy_esys: bool | Eigenpairs = False,
    ) -> Operator:
        if energy_esys is not False:
            return super().sin_phi_operator(
                alpha=alpha, beta=beta, energy_esys=energy_esys
            )
        key = (alpha, beta)
        if key not in self._native_sines:
            self._native_sines[key] = super().sin_phi_operator(alpha=alpha, beta=beta)
        return self._native_sines[key]
