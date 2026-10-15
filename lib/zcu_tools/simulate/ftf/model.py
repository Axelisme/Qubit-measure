"""Coupled F1-SQUID-F2 circuit projected once at the round's DC point."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.baths import (
    H_OVER_K,
    CapacitiveLoss,
    ChargeImpedance,
    FluxLineLoss,
    InductiveLoss,
    QuasiparticleLoss,
    directional_rates,
    gibbs_weights,
    jump_operators,
)
from zcu_tools.simulate.dephasing import (
    dephasing_operators,
    flux_dephasing,
    photon_dephasing,
)
from zcu_tools.simulate.evolution import Basis, DensityState, DriveSource, NumericModel

from .local import Matrix, local_circuit, tensor_operator
from .parameters import FluxoniumCircuit, JointParameters, Resonator, SquidCircuit

RealArray = NDArray[np.float64]


class JointModel:
    """Fixed-DC local projection and lowest-M coupled evolution basis.

    parameters owns the native/local/global truncations and full signed charge
    couplings. dc_flux is finite shape (3,), reduced flux F1/C/F2. temperature_k
    is the common finite nonnegative bath temperature. Construction diagonalizes
    local circuits then the coupled product H0 once. No basis changes along a
    pulse. basis is the square M-coordinate identity; parent_vectors maps these
    coordinates into the fixed local product space for author diagnostics.
    energies_ghz and local_energies_ghz are DC spectra in GHz. These author-only
    numeric objects must not be included in ordinary GUI/MCP Run metadata.
    """

    def __init__(
        self, parameters: JointParameters, dc_flux: RealArray, *, temperature_k: float
    ) -> None:
        self.parameters = parameters
        self.dc_flux = self._flux(dc_flux).copy()
        self.temperature_k = temperature_k
        gibbs_weights(np.zeros(1), temperature_k=temperature_k)
        parts = (parameters.f1, parameters.coupler, parameters.f2)
        local = tuple(
            local_circuit(p, float(f)) for p, f in zip(parts, self.dc_flux, strict=True)
        )
        self._locals = tuple(item[0] for item in local)
        self.local_energies_ghz = tuple(item[1] for item in local)
        self._dimensions = tuple(p.levels for p in parts)
        self._charges = tuple(
            self._lift(c.charge, i) for i, c in enumerate(self._locals)
        )
        n1, nc, n2 = self._charges
        self._coupling = (
            parameters.g1c_ghz * n1 @ nc
            + parameters.g2c_ghz * n2 @ nc
            + parameters.g12_ghz * n1 @ n2
        )
        energies, vectors = np.linalg.eigh(self._parent_hamiltonian(self.dc_flux))
        m = parameters.retained_levels
        self.energies_ghz = energies[:m]
        self.parent_vectors = vectors[:, :m]
        self.basis = Basis(
            "ftf-fixed-dc",
            tuple(f"joint-{i}" for i in range(m)),
            np.eye(m, dtype=complex),
        )

    @staticmethod
    def _flux(flux: RealArray) -> RealArray:
        result = np.asarray(flux, dtype=float)
        if result.shape != (3,) or not np.isfinite(result).all():
            raise ValueError("flux must be finite F1/C/F2 shape (3,)")
        return result

    def _lift(self, op: Matrix, index: int) -> Matrix:
        return tensor_operator(op, index, self._dimensions)

    def _project(self, op: Matrix) -> Matrix:
        return self.parent_vectors.conj().T @ op @ self.parent_vectors

    def _parent_hamiltonian(self, flux: RealArray) -> Matrix:
        h = self._coupling.copy()
        for i, (local, value) in enumerate(zip(self._locals, flux, strict=True)):
            h += self._lift(local.hamiltonian(float(value)), i)
        return h

    def hamiltonian(self, flux: RealArray) -> Matrix:
        """Return H/h GHz in the fixed M basis at finite reduced flux shape (3,).

        Includes all signed g1c/g2c/g12 terms, including for dynamic controls.
        Does not diagonalize into a moving representation or reset a density.
        """
        return self._project(self._parent_hamiltonian(self._flux(flux)))

    def charge_operator(self, subsystem: int) -> Matrix:
        """Return full dimensionless n projected into M; subsystem 0=F1,1=C,2=F2.

        Index outside 0..2 raises ValueError. No 01 normalization is applied.
        """
        if subsystem not in (0, 1, 2):
            raise ValueError("subsystem must be F1=0, C=1 or F2=2")
        return self._project(self._charges[subsystem])

    def charge_squared_operator(self, subsystem: int) -> Matrix:
        """Return dimensionless native n² projected local then joint, not (P n P)².

        subsystem is 0=F1,1=C,2=F2; invalid indices raise ValueError.
        """
        if subsystem not in (0, 1, 2):
            raise ValueError("subsystem must be F1=0, C=1 or F2=2")
        return self._project(
            self._lift(self._locals[subsystem].charge_squared, subsystem)
        )

    def initial_state(self) -> DensityState:
        """Return the joint DC Gibbs density, each shot's independent initial state.

        T=0 uses a uniform mixture over exactly degenerate ground states. No
        density is retained between calls; finite temperature is common to baths.
        """
        return DensityState(
            self.basis,
            np.diag(
                gibbs_weights(self.energies_ghz, temperature_k=self.temperature_k)
            ).astype(complex),
        )

    def eigensystem(self, flux: RealArray) -> tuple[RealArray, Matrix]:
        """Return energies GHz and energy columns in the fixed M basis at flux (3,)."""
        return np.linalg.eigh(self.hamiltonian(flux))

    def _bath_rates(
        self, flux: RealArray, energies: RealArray, vectors: Matrix
    ) -> RealArray:
        rates = np.zeros((len(energies), len(energies)))
        for i, local in enumerate(self._locals):
            p = local.parameters
            channels = []
            if p.q_cap is not None:
                channels.append((local.charge, CapacitiveLoss(p.ec_ghz, p.q_cap)))
            if isinstance(p, FluxoniumCircuit):
                if p.q_ind is not None:
                    channels.append((local.phase, InductiveLoss(p.el_ghz, p.q_ind)))
                if p.xqp:
                    channels.append(
                        (
                            local.quasiparticle(float(flux[i])),
                            QuasiparticleLoss(p.ej_ghz, p.xqp),
                        )
                    )
            else:
                channels.extend(
                    self._coupler_baths(
                        p, local.charge, local.derivative(float(flux[i]))
                    )
                )
            for op, channel in channels:
                joint = vectors.conj().T @ self._project(self._lift(op, i)) @ vectors
                rates += directional_rates(
                    energies, joint, channel, temperature_k=self.temperature_k
                )
        return rates

    @staticmethod
    def _coupler_baths(p: SquidCircuit, charge: Matrix, derivative: Matrix):
        channels = []
        if p.resistance_ohm is not None:
            channels.append((charge, ChargeImpedance(p.resistance_ohm)))
        if p.flux_resistance_ohm is not None and p.mutual_phi0_per_a is not None:
            channels.append(
                (derivative, FluxLineLoss(p.flux_resistance_ohm, p.mutual_phi0_per_a))
            )
        return channels

    def rates(self, flux: RealArray) -> RealArray:
        """Return all directional rates [target,source] in 1/ns in H(flux) order.

        Bath eigensystem includes g12 but no microwave. Uses instantaneous dressed
        matrix elements for every retained multilevel transition; exact active
        degeneracies fail under the documented secular convention.
        """
        flux = self._flux(flux)
        energies, vectors = self.eigensystem(flux)
        return self._bath_rates(flux, energies, vectors)

    def readout_frequencies_ghz(
        self, flux: RealArray, resonators: tuple[Resonator, Resonator]
    ) -> RealArray:
        """Return (M,2) dressed GHz frequencies using full local-product virtual sums.

        flux is reduced (3,). Resonator coefficients multiply full parent charges.
        All parent eigenstates (not only lowest M) appear in counter-rotating
        denominators. State order is the fixed-M H(flux) eigensystem. No sink pull
        is substituted for retained leakage/coupler states. Exact singular
        denominators with nonzero matrix elements fail numerically; there is no
        runtime dispersive/mixing/quality threshold or convergence certification.
        """
        flux = self._flux(flux)
        parent_energy, parent_v = np.linalg.eigh(self._parent_hamiltonian(flux))
        energy, v = self.eigensystem(flux)
        embedded = self.parent_vectors @ v
        gap = energy[:, None] - parent_energy[None, :]
        results = []
        for r in resonators:
            op = sum(
                (
                    g * n
                    for g, n in zip(
                        r.charge_coefficients_ghz, self._charges, strict=True
                    )
                ),
                np.zeros_like(self._charges[0]),
            )
            strength = np.abs(embedded.conj().T @ op @ parent_v) ** 2
            denominator = gap**2 - r.frequency_ghz**2
            if np.any((denominator == 0) & (strength != 0)):
                raise RuntimeError("singular parent virtual transition denominator")
            fraction = np.divide(
                2 * gap, denominator, out=np.zeros_like(gap), where=denominator != 0
            )
            results.append(r.frequency_ghz + np.sum(strength * fraction, axis=1))
        result = np.column_stack(results)
        if not np.isfinite(result).all():
            raise RuntimeError("nonfinite parent virtual transition response")
        return result

    def collapse_operators(
        self, flux: RealArray, resonators: tuple[Resonator, Resonator]
    ) -> tuple[Matrix, ...]:
        """Return instantaneous dressed relaxation + CP pure dephasing in fixed M.

        flux: finite reduced (3,); resonators supplies residual photon pulls/nth.
        Slopes use rad/us and rates convert explicitly to 1/ns. No added T1/2.
        Operators include sqrt(1/ns); the evolution tool owns integration.
        """
        flux = self._flux(flux)
        energy, v = self.eigensystem(flux)
        rates = self._bath_rates(flux, energy, v)
        slopes = np.array(
            [
                np.diag(
                    v.conj().T
                    @ self._project(self._lift(local.derivative(float(flux[i])), i))
                    @ v
                ).real
                * 2
                * np.pi
                * 1000
                for i, local in enumerate(self._locals)
            ]
        )
        aphi = np.array([c.parameters.aphi for c in self._locals])
        gamma = flux_dephasing(slopes, aphi)
        if any(r.thermal_occupation != 0 for r in resonators):
            pulls = (
                (
                    self.readout_frequencies_ghz(flux, resonators)
                    - np.array([r.frequency_ghz for r in resonators])
                )
                * 2
                * np.pi
                * 1000
            )
            linewidth = (
                np.array(
                    [
                        r.kappa_external_per_ns + r.kappa_internal_per_ns
                        for r in resonators
                    ]
                )
                * 1000
            )
            nth = np.array([self._occupation(r) for r in resonators])
            gamma += photon_dephasing(pulls.T, linewidth, nth)
        pure = tuple(v @ op @ v.conj().T for op in dephasing_operators(gamma))
        return jump_operators(rates, v) + pure

    def numeric_model(
        self,
        flux_at: Callable[[float], RealArray],
        sources: tuple[DriveSource, ...],
        *,
        end_ns: float,
        bath_segment_ns: float,
        resonators: tuple[Resonator, Resonator],
    ) -> NumericModel:
        """Build fixed-basis continuous H with piecewise instantaneous baths.

        flux_at returns actual reduced (3,) flux at absolute ns; sources holds
        lab-frame microwave quadratures and zero operators for flux-event sources.
        end_ns is finite nonnegative evolution end. bath_segment_ns is the user's
        positive update interval, rounded upward to at least one 2 ns event tick.
        Baths use segment-start flux, remain fixed during each segment, and are
        rotated back into this fixed basis. Evolution owns event partitioning and
        integration; states are never reset. resonators supplies residual photons.
        Invalid scales raise ValueError. This is not a convergence certification.
        """
        if (
            not np.isfinite([end_ns, bath_segment_ns]).all()
            or end_ns < 0
            or bath_segment_ns <= 0
        ):
            raise ValueError("finite evolution end and positive bath segment required")
        step = max(2.0, 2 * float(np.rint(bath_segment_ns / 2)))
        boundaries = tuple(float(t) for t in np.arange(0.0, end_ns, step))
        cached: dict[int, tuple[Matrix, ...]] = {}
        h0 = self.hamiltonian(self.dc_flux)

        def hamiltonian_at(t: float) -> Matrix:
            return self.hamiltonian(flux_at(t)) - h0

        def collapse_at(t: float) -> tuple[Matrix, ...]:
            index = int(np.floor(max(t, 0.0) / step))
            if index not in cached:
                cached[index] = self.collapse_operators(
                    flux_at(index * step), resonators
                )
            return cached[index]

        return NumericModel(
            "ftf-joint-v1",
            self.basis,
            h0,
            sources,
            hamiltonian_at=hamiltonian_at,
            collapse_at=collapse_at,
            event_times_ns=boundaries,
        )

    def _occupation(self, resonator: Resonator) -> float:
        if resonator.thermal_occupation is not None:
            return resonator.thermal_occupation
        if self.temperature_k == 0:
            return 0.0
        x = H_OVER_K * resonator.frequency_ghz / self.temperature_k
        return 0.0 if x > 700 else float(1 / np.expm1(x))
