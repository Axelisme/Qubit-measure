"""Coupled F1-SQUID-F2 circuit projected once at the round's DC point."""

from dataclasses import dataclass
from functools import partial
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.adc import ReadoutModel
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
from zcu_tools.simulate.evolution import (
    Basis,
    DensityState,
    DriveSource,
    FluxExpansion,
    FluxLevel,
    FluxTerm,
    ModelSource,
    NumericModel,
    SecularDrive,
    SecularModel,
    quantize_ns,
)

from .local import Matrix, local_circuit, tensor_operator
from .parameters import FluxoniumCircuit, JointParameters, Resonator, SquidCircuit

RealArray = NDArray[np.float64]


@dataclass(frozen=True)
class FluxDynamics:
    """Caller metadata for absolute-ns flux events.

    identity: optional caller-assigned dependency descriptor; not read by Engine
        or included in NumericModel.
    event_times_ns: finite absolute events, including delayed pulse support and
        envelope/phase changes. Empty means no additional events.
        JointModel.numeric_model quantizes and unions them with bath ticks.
    Invalid event coordinates raise ValueError. This record performs no dynamics.
    """

    identity: str | None = None
    event_times_ns: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not np.isfinite(self.event_times_ns).all():
            raise ValueError("flux event times must be finite ns")


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
    Circuit parameters, temperature and fixed-DC projection must remain unchanged
    for this model's lifetime; exact memo results are read-only.
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
        # Each cache holds only a few exact points in this fixed projection.
        self._eigensystems: dict[bytes, tuple[RealArray, Matrix]] = {}
        self._rates: dict[tuple[bytes, bytes, bytes], RealArray] = {}
        self.dc_flux.setflags(write=False)
        self.energies_ghz.setflags(write=False)
        self.parent_vectors.setflags(write=False)

    def native_embedding(self) -> Matrix:
        """Return an owned dimensionless map from M to physical native coordinates.

        Shape is (F1 native_cutoff, 2*C charge_cutoff+1, F2 native_cutoff, M).
        The first/third axes are oscillator number 0..cutoff-1 with the circuit's
        fixed EC/EL scale. The middle axis is Cooper-pair charge -ncut..+ncut,
        not oscillator number. Contract the final axis with a state vector in
        ``basis`` coordinates, or both final axes with a density matrix.
        Columns are orthonormal; no state populations are clipped/renormalized.
        This author diagnostic includes local and joint projections and their
        complex phases. Across native cutoffs, pad oscillator axes at the high
        end and center the charge axis. Comparing different EC/EL requires an
        additional oscillator-coordinate transformation, not zero padding.
        Allocates the returned tensor only; does not change runtime state.
        """
        v1, vc, v2 = (c.vectors for c in self._locals)
        parent = self.parent_vectors.reshape(*self._dimensions, -1)
        return np.einsum("ai,bj,ck,ijkm->abcm", v1, vc, v2, parent, optimize=True)

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

    def flux_operator(self, subsystem: int) -> Matrix:
        """Return d(H/h)/df at DC in retained coordinates, in GHz/Phi0.

        f = Phi/Phi0 is external reduced flux; subsystem is F1=0, C=1 or F2=2.
        Uses the analytic derivative of this model's fixed-coordinate Hamiltonian,
        projected through the DC local and joint bases, without rediagonalizing.
        This is a fixed-DC small-signal operator, not H(Phi(t)) for fast flux.
        An index outside 0..2 raises ValueError.
        """
        if subsystem not in (0, 1, 2):
            raise ValueError("subsystem must be F1=0, C=1 or F2=2")
        local = self._locals[subsystem]
        derivative = local.derivative(float(self.dc_flux[subsystem]))
        return self._project(self._lift(derivative, subsystem))

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
        """Return read-only energies GHz and energy columns at finite flux (3,).

        Exact float64 flux bytes reuse results within this joint's fixed projection.
        At most eight eigensystems are retained; callers must not mutate results.
        """
        flux = self._flux(flux)
        key = flux.tobytes()
        if key not in self._eigensystems:
            energies, vectors = np.linalg.eigh(self.hamiltonian(flux))
            energies.setflags(write=False)
            vectors.setflags(write=False)
            if len(self._eigensystems) == 8:
                self._eigensystems.pop(next(iter(self._eigensystems)))
            self._eigensystems[key] = energies, vectors
        return self._eigensystems[key]

    def _bath_rates(
        self, flux: RealArray, energies: RealArray, vectors: Matrix
    ) -> RealArray:
        key = (flux.tobytes(), energies.tobytes(), vectors.tobytes())
        if key in self._rates:
            return self._rates[key]
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
        rates.setflags(write=False)
        if len(self._rates) == 8:
            self._rates.pop(next(iter(self._rates)))
        self._rates[key] = rates
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
        """Return read-only directional rates [target,source] in 1/ns in H(flux) order.

        Up to eight exact flux/eigensystem inputs reuse rates within this joint.

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
        gamma = self._pure_dephasing(flux, v, resonators)
        pure = tuple(v @ op @ v.conj().T for op in dephasing_operators(gamma))
        return jump_operators(rates, v) + pure

    def fixed_numeric_model(
        self,
        flux: RealArray,
        sources: tuple[DriveSource, ...],
        *,
        resonators: tuple[Resonator, Resonator],
    ) -> NumericModel:
        """Return static H and complete Lindblad baths in this joint fixed basis.

        flux: finite (F1,C,F2) reduced flux in Phi/Phi0, constant throughout the
        requested evolution (caller must not use this for pulses/passive tails).
        sources: fixed-basis numeric quadratures in GHz per complex amplitude.
        resonators: two numeric readout modes supplying residual photon baths.
        Includes full couplings, directional relaxation and effective Markov
        dephasing at this flux, without event metadata. Initial Gibbs state
        remains separately explicit. Invalid shapes,
        active degenerate secular transitions or invalid bath data raise ValueError.
        No physical applicability or basis convergence certification is performed.
        """
        return NumericModel(
            "ftf-joint-v1",
            self.basis,
            self.hamiltonian(flux),
            sources,
            self.collapse_operators(flux, resonators),
        )

    def numeric_model(
        self,
        sources: tuple[DriveSource, ...],
        *,
        end_ns: float,
        bath_segment_ns: float,
        dynamics: FluxDynamics | None = None,
    ) -> NumericModel:
        """Return static DC Hamiltonian, sources and absolute-ns event metadata.

        sources: fixed-basis microwave quadratures in GHz per complex amplitude.
        end_ns: finite nonnegative end of the event window.
        bath_segment_ns: positive event spacing, rounded to the nearest 2 ns
            with a minimum of one tick.
        dynamics: optional extra events; quantized events in [0, end_ns) join
            regular ticks and zero. None adds no extra events. Its identity is
            not copied into the returned record.
        The record has no collapse operators and performs no evolution or bath
        updates. Engine uses ModelSource from model_source instead.
        Invalid scales or event coordinates raise ValueError. This is not a
        physical applicability or convergence certification.
        """
        if (
            not np.isfinite([end_ns, bath_segment_ns]).all()
            or end_ns < 0
            or bath_segment_ns <= 0
        ):
            raise ValueError("finite evolution end and positive bath segment required")
        dynamics = FluxDynamics() if dynamics is None else dynamics
        step = max(2.0, 2 * float(np.rint(bath_segment_ns / 2)))
        boundaries = tuple(
            sorted(
                {
                    0.0,
                    *(float(t) for t in np.arange(0.0, end_ns, step)),
                    *(
                        q
                        for t in dynamics.event_times_ns
                        if 0 <= (q := quantize_ns(t)) < end_ns
                    ),
                }
            )
        )
        return NumericModel(
            "ftf-joint-v1",
            self.basis,
            self.hamiltonian(self.dc_flux),
            sources,
            event_times_ns=boundaries,
        )

    def _occupation(self, resonator: Resonator) -> float:
        if resonator.thermal_occupation is not None:
            return resonator.thermal_occupation
        if self.temperature_k == 0:
            return 0.0
        x = H_OVER_K * resonator.frequency_ghz / self.temperature_k
        return 0.0 if x > 700 else float(1 / np.expm1(x))

    def _pure_dephasing(
        self,
        flux: RealArray,
        vectors: Matrix,
        resonators: tuple[Resonator, Resonator],
    ) -> RealArray:
        """Joint eigenbasis pure rates in 1/ns, shared by both model representations."""
        slopes = np.array(
            [
                np.diag(
                    vectors.conj().T
                    @ self._project(self._lift(local.derivative(float(flux[i])), i))
                    @ vectors
                ).real
                * 2
                * np.pi
                * 1000
                for i, local in enumerate(self._locals)
            ]
        )
        gamma = flux_dephasing(
            slopes, np.array([c.parameters.aphi for c in self._locals])
        )
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
        return gamma

    def secular_model(
        self,
        level: FluxLevel,
        sources: tuple[DriveSource, ...],
        *,
        resonators: tuple[Resonator, Resonator],
    ) -> SecularModel:
        """Build one F1/C/F2 lattice model at self.dc_flux + 1e-3 * level.

        level has three integers. sources supplies microwave quadratures in joint's
        fixed DC coordinates; resonators supplies the residual-photon baths. Returns
        energies, quadratures, directional and pure rates in the instantaneous energy
        basis, with eigenvectors mapping back to fixed coordinates. Active exact bath
        degeneracy raises ValueError naming the level and secular requirement. Other
        invalid shapes or numeric bath inputs retain their existing ValueError.
        """
        if len(level) != 3 or np.asarray(level).dtype.kind not in "iu":
            raise ValueError("FTF level must contain three integers")
        flux = self.dc_flux + 1e-3 * np.array(level)
        energy, vectors = self.eigensystem(flux)
        try:
            rates = self._bath_rates(flux, energy, vectors)
        except ValueError as error:
            if "degenerate secular transition" not in str(error):
                raise
            raise ValueError(
                f"fast simulation requires a secular bath at level {level}"
            ) from error
        drives = tuple(
            SecularDrive(
                s.name,
                vectors.conj().T @ s.in_phase_ghz @ vectors,
                vectors.conj().T
                @ (
                    np.zeros_like(s.in_phase_ghz)
                    if s.quadrature_ghz is None
                    else s.quadrature_ghz
                )
                @ vectors,
            )
            for s in sources
        )
        return SecularModel(
            level,
            _identity(self, level, sources, resonators),
            energy,
            vectors,
            drives,
            rates,
            self._pure_dephasing(flux, vectors, resonators),
        )

    def flux_expansion(self) -> FluxExpansion:
        """Return the exact trigonometric H(flux)/h expansion in joint fixed coordinates.

        Fluxonium uses cos/sin(2*pi*f); the SQUID uses cos/sin(pi*f). All couplings
        and flux-independent kinetic terms enter static, in GHz. No sampling or
        Taylor approximation is used; flux is absolute reduced F1/C/F2 flux.
        """
        static = self._coupling.copy()
        terms: list[FluxTerm] = []
        for i, local in enumerate(self._locals):
            static += self._lift(local.kinetic, i)
            p = local.parameters
            if isinstance(p, FluxoniumCircuit):
                terms.extend(
                    (
                        FluxTerm(
                            i,
                            "cos_2pi",
                            self._project(self._lift(-p.ej_ghz * local.cosine, i)),
                        ),
                        FluxTerm(
                            i,
                            "sin_2pi",
                            self._project(self._lift(p.ej_ghz * local.sine, i)),
                        ),
                    )
                )
            else:
                terms.extend(
                    (
                        FluxTerm(
                            i,
                            "cos_pi",
                            self._project(self._lift(-p.ejmax_ghz * local.cosine, i)),
                        ),
                        FluxTerm(
                            i,
                            "sin_pi",
                            self._project(
                                self._lift(p.asymmetry * p.ejmax_ghz * local.sine, i)
                            ),
                        ),
                    )
                )
        return FluxExpansion(self._project(static), tuple(terms))


def _identity(
    joint: JointModel,
    level: FluxLevel,
    sources: tuple[DriveSource, ...],
    resonators: tuple[Resonator, Resonator],
) -> str:
    digest = sha256(b"ftf-secular-v1")
    digest.update(
        repr((joint.parameters, joint.temperature_k, level, resonators)).encode()
    )
    digest.update(joint.dc_flux.astype("<f8").tobytes())
    # Fixed-coordinate phase conventions affect drives/overlaps even at equal DC.
    digest.update(joint.parent_vectors.astype("<c16").tobytes())
    for source in sources:
        digest.update(source.name.encode())
        digest.update(source.in_phase_ghz.astype("<c16").tobytes())
        if source.quadrature_ghz is not None:
            digest.update(source.quadrature_ghz.astype("<c16").tobytes())
    return digest.hexdigest()


def secular_model(
    joint: JointModel,
    level: FluxLevel,
    sources: tuple[DriveSource, ...],
    *,
    resonators: tuple[Resonator, Resonator],
) -> SecularModel:
    """Build joint's instantaneous secular model; see JointModel.secular_model.

    Inputs use F1/C/F2 integer lattice offsets, fixed-DC drive quadratures in
    GHz and resonator photon baths. Non-secular baths raise ValueError naming
    level. The resulting model uses the instantaneous energy eigenbasis.
    """
    return joint.secular_model(level, sources, resonators=resonators)


def flux_expansion(joint: JointModel) -> FluxExpansion:
    """Return joint's exact trigonometric GHz Hamiltonian in fixed DC coordinates.

    The terms take absolute reduced F1/C/F2 flux. No Taylor approximation or
    sampling enters this expansion; see JointModel.flux_expansion.
    """
    return joint.flux_expansion()


def model_source(
    joint: JointModel,
    sources: tuple[DriveSource, ...],
    *,
    resonators: tuple[Resonator, Resonator],
) -> ModelSource:
    """Return a round-local cached secular builder, DC Gibbs density and exact H.

    sources are fixed-DC microwave ports and resonators supplies photon baths.
    The returned ModelSource owns level caching; joint/source inputs must remain
    unchanged for its lifetime. Initial density is in the DC energy eigenbasis.
    Bath applicability errors occur on model(level), not during construction.
    """
    energy, _ = joint.eigensystem(joint.dc_flux)
    return ModelSource(
        np.diag(gibbs_weights(energy, temperature_k=joint.temperature_k)).astype(
            complex
        ),
        partial(secular_model, joint, sources=sources, resonators=resonators),
        flux_expansion(joint),
    )


def readout_model(
    joint: JointModel, resonators: tuple[Resonator, Resonator]
) -> ReadoutModel:
    """Return ADC's model at DC, ignoring flux tails and readout-overlapping controls.

    Frequencies and directional rates use DC energy-state order, including every
    retained state. Resonator linewidths are in 1/ns. Reference frequency is zero
    GHz and Receiver uses its neutral defaults; composition may replace these
    calibration fields. Invalid bath/parent response data propagate unchanged.
    """
    return ReadoutModel(
        _identity(joint, (0, 0, 0), (), resonators),
        joint.basis.labels,
        joint.readout_frequencies_ghz(joint.dc_flux, resonators),
        0.0,
        np.array([r.kappa_internal_per_ns for r in resonators]),
        np.array([r.kappa_external_per_ns for r in resonators]),
        joint.rates(joint.dc_flux),
    )
