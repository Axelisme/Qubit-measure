"""Fixed-DC single-fluxonium projection and its evolution/readout composition."""

from dataclasses import dataclass
from functools import partial
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.baths import H_OVER_K, gibbs_weights
from zcu_tools.simulate.dephasing import flux_dephasing, photon_dephasing
from zcu_tools.simulate.evolution import (
    Basis,
    DriveSource,
    FluxExpansion,
    FluxLevel,
    FluxTerm,
    ModelSource,
    SecularDrive,
    SecularModel,
)

from .coherence.purcell import calculate_purcell_t1_vs_flux
from .coherence.rates import FluxoniumNoise, calculate_directional_loss_rates
from .dispersive import calculate_dispersive_vs_flux_fast
from .matrix_element import calculate_n_oper
from .operators import calculate_eigensystem, calculate_oscillator_operators

RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
_DISABLED_NOISE = FluxoniumNoise()


@dataclass(frozen=True)
class DispersiveResonator:
    """One weak dispersive charge-coupled resonator, without ADC calibration.

    frequency_ghz: finite positive bare resonator frequency H/h, GHz.
    g_ghz: finite signed coupling multiplying full dimensionless n*(a+a†), GHz.
    kappa_per_ns: finite positive total photon decay rate, ns^-1, not kappa/2pi.
    thermal_occupation: finite nonnegative residual photon occupation for pure
    dephasing; None selects Bose occupation at the model temperature. Purcell
    always uses the model temperature, not this residual occupation.
    Invalid input raises ValueError. No dispersive-validity gate is implied.
    """

    frequency_ghz: float
    g_ghz: float
    kappa_per_ns: float
    thermal_occupation: float | None = None

    def __post_init__(self) -> None:
        if (
            not np.isfinite([self.frequency_ghz, self.g_ghz, self.kappa_per_ns]).all()
            or min(self.frequency_ghz, self.kappa_per_ns) <= 0
        ):
            raise ValueError(
                "resonator frequency/linewidth must be positive and coupling finite"
            )
        if self.thermal_occupation is not None and (
            not np.isfinite(self.thermal_occupation) or self.thermal_occupation < 0
        ):
            raise ValueError("thermal occupation must be nonnegative finite or None")


class SingleFluxoniumModel:
    """Single fluxonium in a fixed retained subspace selected at round DC flux.

    params: (EJ,EC,EL) H/h GHz, EJ >=0 and EC/EL >0. dc_flux: finite absolute
    Phi/Phi0. cutoff: integer native oscillator dimension >=2. retained_levels:
    integer M in [2,cutoff], lowest DC eigenstates used for every operator.
    temperature_k: finite nonnegative K, common to equilibrium populations and
    baths. These five public attributes must not change during this model's life.
    basis: identity Basis in M fixed DC coordinates, labels level-0..level-(M-1).
    energies_ghz: immutable native DC energies (M,), ascending H/h GHz.

    Construction diagonalizes once at DC. Other fluxes diagonalize H projected
    into that SAME subspace, not a newly truncated instantaneous native space.
    Eigenvectors are square/unitary in fixed DC coordinates; large excursions
    require caller convergence checks on retained_levels. No file/device/global
    state is read or changed. Invalid inputs raise ValueError; eigensolver,
    scqubits and dressed-labeling failures propagate, without fallback/repair.
    """

    def __init__(
        self,
        params: tuple[float, float, float],
        dc_flux: float,
        *,
        cutoff: int,
        retained_levels: int,
        temperature_k: float,
    ) -> None:
        if type(retained_levels) is not int or not 2 <= retained_levels <= cutoff:
            raise ValueError("retained_levels must be an integer in [2,cutoff]")
        gibbs_weights(np.zeros(1), temperature_k=temperature_k)
        energy, projection = calculate_eigensystem(
            params, dc_flux, cutoff=cutoff, evals_count=retained_levels
        )
        native = calculate_oscillator_operators(params, cutoff=cutoff)
        self.params, self.dc_flux = params, dc_flux
        self.cutoff, self.retained_levels = cutoff, retained_levels
        self.temperature_k = temperature_k
        self.energies_ghz = energy
        self._projection = projection
        self._expansion = FluxExpansion(
            projection.conj().T @ native.static_ghz @ projection,
            (
                FluxTerm(
                    0, "cos_2pi", projection.conj().T @ native.cosine_ghz @ projection
                ),
                FluxTerm(
                    0, "sin_2pi", projection.conj().T @ native.sine_ghz @ projection
                ),
            ),
        )
        self._charge = calculate_n_oper(
            params, dc_flux, retained_levels, (energy, projection), cutoff=cutoff
        )
        self._charge.setflags(write=False)
        self.basis = Basis(
            "single-fluxonium-fixed-dc",
            tuple(f"level-{i}" for i in range(retained_levels)),
            np.eye(retained_levels, dtype=np.complex128),
        )

    def hamiltonian(self, flux: float) -> ComplexArray:
        """Return owned H(flux)/h GHz (M,M) in fixed DC coordinates.

        flux is finite absolute Phi/Phi0, not an offset. Exact cosine/sine flux
        dependence is retained within the fixed projection. Invalid flux raises
        ValueError; the result does not modify the model or certify convergence.
        """
        if not np.isfinite(flux):
            raise ValueError("flux must be finite Phi/Phi0")
        angle = 2 * np.pi * flux
        return (
            self._expansion.static
            + np.cos(angle) * self._expansion.terms[0].matrix
            + np.sin(angle) * self._expansion.terms[1].matrix
        )

    def eigensystem(self, flux: float) -> tuple[RealArray, ComplexArray]:
        """Return ascending energies (M,) GHz and square fixed-DC columns (M,M).

        flux is finite absolute Phi/Phi0. Arrays are owned read-only; no flux
        folding or extra phase convention is imposed on numpy.linalg.eigh.
        Invalid flux raises ValueError and eigensolver failures propagate.
        """
        energy, vectors = np.linalg.eigh(self.hamiltonian(flux))
        energy.setflags(write=False)
        vectors.setflags(write=False)
        return energy, vectors

    def charge_operator(self) -> ComplexArray:
        """Return immutable dimensionless n (M,M) in fixed DC coordinates.

        Full physical charge is projected, without n01 normalization. For an
        instantaneous charge matrix use V.conj().T @ n @ V from eigensystem.
        """
        return self._charge

    def flux_expansion(self) -> FluxExpansion:
        """Return immutable H/h expansion in fixed DC coordinates, using absolute flux.

        Static is projected H_LC; loop 0 terms are -EJ*cos(phi)*cos(2*pi*flux)
        and +EJ*sin(phi)*sin(2*pi*flux), GHz. This is exact in the retained
        subspace, not a Taylor expansion or a native-convergence claim.
        """
        return self._expansion

    def readout_frequencies_ghz(self, resonator: DispersiveResonator) -> RealArray:
        """Return absolute resonator frequencies (M,) GHz in DC energy-state order.

        resonator defines the bare frequency and full-charge coupling. Reuses
        the existing dressed-spectrum helper with caller cutoff, all native
        qubit levels and four resonator photon levels. Dressed label collisions
        raise DressedLabelingError; diagonalization failures propagate. Returns
        owned data, without populating ADC or modifying any model coordinates.
        """
        values = calculate_dispersive_vs_flux_fast(
            self.params,
            np.array([self.dc_flux]),
            resonator.frequency_ghz,
            resonator.g_ghz,
            res_dim=4,
            qub_dim=self.cutoff,
            qub_cutoff=self.cutoff,
            return_dim=self.retained_levels,
        )
        return np.array([value[0] for value in values])

    def _rates(
        self,
        flux: float,
        energy: RealArray,
        vectors: ComplexArray,
        noise: FluxoniumNoise,
        resonator: DispersiveResonator | None,
    ) -> RealArray:
        rates = calculate_directional_loss_rates(
            self.params,
            flux,
            (energy, self._projection @ vectors),
            noise,
            cutoff=self.cutoff,
            temperature_k=self.temperature_k,
        )
        if resonator is not None and resonator.g_ghz != 0:
            lifetime = calculate_purcell_t1_vs_flux(
                np.array([flux]),
                resonator.frequency_ghz,
                resonator.kappa_per_ns / (2 * np.pi),
                resonator.g_ghz,
                self.temperature_k,
                self.params,
                progress=False,
            )[0]
            ratio = (
                0.0
                if self.temperature_k == 0
                else np.exp(-H_OVER_K * (energy[1] - energy[0]) / self.temperature_k)
            )
            down = 1 / (lifetime * (1 + ratio))
            rates[0, 1] += down
            rates[1, 0] += down * ratio
        return rates

    def _dephasing(
        self,
        flux: float,
        vectors: ComplexArray,
        noise: FluxoniumNoise,
        resonator: DispersiveResonator | None,
    ) -> RealArray:
        angle = 2 * np.pi * flux
        derivative = (
            2
            * np.pi
            * (
                -np.sin(angle) * self._expansion.terms[0].matrix
                + np.cos(angle) * self._expansion.terms[1].matrix
            )
        )
        slopes = np.asarray(
            np.real(np.diag(vectors.conj().T @ derivative @ vectors)), dtype=np.float64
        )
        gamma = flux_dephasing(
            (2 * np.pi * 1000 * slopes)[None, :], np.array([noise.aphi])
        )
        if resonator is not None:
            nth = resonator.thermal_occupation
            if nth is None:
                exponent = (
                    np.inf
                    if self.temperature_k == 0
                    else (H_OVER_K * resonator.frequency_ghz / self.temperature_k)
                )
                nth = float(np.exp(-exponent) / -np.expm1(-exponent))
            if nth:
                # Pure photon dephasing uses instantaneous pulls, not a DC hold.
                values = calculate_dispersive_vs_flux_fast(
                    self.params,
                    np.array([flux]),
                    resonator.frequency_ghz,
                    resonator.g_ghz,
                    qub_dim=self.cutoff,
                    qub_cutoff=self.cutoff,
                    return_dim=self.retained_levels,
                )
                pulls = np.array([[v[0] - resonator.frequency_ghz for v in values]])
                gamma += photon_dephasing(
                    2 * np.pi * 1000 * pulls,
                    np.array([1000 * resonator.kappa_per_ns]),
                    np.array([nth]),
                )
        return gamma

    def secular_model(
        self,
        level: FluxLevel,
        sources: tuple[DriveSource, ...] = (),
        *,
        noise: FluxoniumNoise = _DISABLED_NOISE,
        resonator: DispersiveResonator | None = None,
    ) -> SecularModel:
        """Build one quantized model at dc_flux + level[0]*1e-3 Phi0.

        level contains exactly one integer. sources supplies named Hermitian
        drive quadratures (M,M), GHz per envelope unit, in fixed DC coordinates.
        noise supplies constant Q/qp losses and effective echo pure dephasing.
        Optional resonator adds existing equilibrium 0↔1 Purcell T1, split by
        detailed balance, and residual-photon pure dephasing; no high-level
        Purcell channels are added. Rates are ns^-1, R[target,source], with
        zero diagonal. Returned energies, drives and rates use this flux's own
        eigenbasis; vectors map it into fixed DC coordinates. Invalid level or
        drive dimensions raise ValueError; bath/dressed failures propagate.
        """
        if len(level) != 1 or type(level[0]) is not int:
            raise ValueError("single fluxonium level requires exactly one integer")
        if any(
            s.in_phase_ghz.shape != (self.retained_levels, self.retained_levels)
            for s in sources
        ):
            raise ValueError("drive shape must match retained model coordinates")
        flux = self.dc_flux + 1e-3 * level[0]
        energy, vectors = self.eigensystem(flux)
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
        rates = self._rates(flux, energy, vectors, noise, resonator)
        gamma = self._dephasing(flux, vectors, noise, resonator)
        digest = sha256(
            repr(
                (
                    self.params,
                    self.dc_flux,
                    self.cutoff,
                    self.retained_levels,
                    self.temperature_k,
                    level,
                    noise,
                    resonator,
                )
            ).encode()
        )
        for drive in drives:
            digest.update(repr(drive.name).encode())
            digest.update(drive.in_phase_ghz.tobytes())
            digest.update(drive.quadrature_ghz.tobytes())
        return SecularModel(
            level, digest.hexdigest(), energy, vectors, drives, rates, gamma
        )

    def model_source(
        self,
        sources: tuple[DriveSource, ...] = (),
        *,
        noise: FluxoniumNoise = _DISABLED_NOISE,
        resonator: DispersiveResonator | None = None,
    ) -> ModelSource:
        """Return a round-local engine source with DC Gibbs initial density.

        sources/noise/resonator follow secular_model; the engine requests
        quantized levels and owns the cache. Initial density is diagonal in the
        DC eigenbasis, with Gibbs weights at temperature_k. H along continuous
        flux comes from flux_expansion, while baths retain each row's quantized
        level. The source retains models, not an evolving density between shots.
        Invalid source/bath inputs raise on the first requested level, as in
        secular_model; construction only validates the initial density.
        """
        return ModelSource(
            np.diag(
                gibbs_weights(self.energies_ghz, temperature_k=self.temperature_k)
            ).astype(complex),
            partial(
                self.secular_model, sources=sources, noise=noise, resonator=resonator
            ),
            self.flux_expansion(),
        )
