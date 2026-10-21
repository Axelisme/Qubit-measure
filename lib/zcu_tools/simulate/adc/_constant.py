"""Closed-form shared-line propagation and sampled scattering projections."""

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from ._integrated import IntegratedLine
from ._quadratic import field_product_integral
from ._sampled import SampledLine
from .model import ReadoutModel, RealArray
from .request import ReadoutPulse, SampleGrid


class ConstantLine:
    """Share homogeneous line propagation while keeping carrier shifts scalar.

    q: finite population generator (state,state), indexed [target,source], 1/ns.
    a: finite complex line blocks (state,2,2), 1/ns.
    v: finite drive coupling (2,), sqrt(1/ns).
    scattering: finite complex output coupling (2,), including mismatch phase.
    All arrays come from a validated ReadoutModel and remain caller-owned.
    Methods cache propagators for this model; no shot state is retained.

    The block generator is [[Q, 0], [b*incident, M+i*omega]]. For a
    well-separated Sylvester solve (M+i*omega)X-XQ=b, its driven propagator
    is incident*(exp((M+i*omega)*dt)X-X*exp(Q*dt)). Only X and the scalar
    carrier phase depend on frequency. No diagonalizability assumption is made.
    """

    def __init__(
        self,
        q: RealArray,
        a: NDArray[np.complex128],
        v: RealArray,
        scattering: NDArray[np.complex128],
    ):
        self._q = q
        n = len(q)
        self._m = np.asarray(np.kron(q, np.eye(2)), dtype=np.complex128)
        self._b = np.zeros((2 * n, n), dtype=np.complex128)
        for s in range(n):
            self._m[2 * s : 2 * s + 2, 2 * s : 2 * s + 2] -= a[s]
            self._b[2 * s : 2 * s + 2, s] = v
        self._output = np.tile(scattering, n)
        # A scalar carrier shift leaves both Schur bases unchanged, even for
        # defective generators. Only the triangular Sylvester solve repeats.
        moment_schur = scipy.linalg.schur(self._m, output="complex")
        rate_schur = scipy.linalg.schur(-q.T, output="complex")
        self._r, self._u = moment_schur[0], moment_schur[1]
        self._s, self._w = rate_schur[0], rate_schur[1]
        self._rhs = self._u.conj().T @ self._b @ self._w
        (self._trsyl,) = scipy.linalg.get_lapack_funcs(("trsyl",), (self._r, self._s))
        self._projections: dict[
            bytes, tuple[NDArray[np.complex128], NDArray[np.complex128]]
        ] = {}
        self._homogeneous: dict[
            float, tuple[NDArray[np.complex128], NDArray[np.complex128]]
        ] = {}
        self._integrated: IntegratedLine | None = None

    def coherent_integral(
        self,
        fields: NDArray[np.complex128],
        incidents: NDArray[np.complex128],
        omegas: RealArray,
        duration: float,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Advance fixed branches and integrate their coherent field products.

        This route requires Q=0: fields (state,2) are sqrt(photons), incidents
        (carrier,) are sqrt(photons/ns) at the interval start, and omegas are
        carrier/reference rad/ns. duration is positive ns. Return final fields
        and (state,2,state,2) integrals in photon*ns, retaining all cross terms.
        Singular generators use augmented exponentials, not ODE or inversion.
        Nonfinite propagation/integrals raise RuntimeError.
        """
        if np.any(self._q):
            raise ValueError("coherent field integrals require fixed state branches")
        n = len(fields)
        generators = np.array(
            [self._m[2 * s : 2 * s + 2, 2 * s : 2 * s + 2] for s in range(n)]
        )
        products = field_product_integral(
            generators, self._b[:2, 0].real, fields, incidents, omegas, duration
        )
        return self.coherent_step(fields, incidents, omegas, duration), products

    def coherent_step(
        self,
        fields: NDArray[np.complex128],
        incidents: NDArray[np.complex128],
        omegas: RealArray,
        duration: float,
    ) -> NDArray[np.complex128]:
        """Advance fixed (state,2) fields by positive ns with flat carriers.

        Requires Q=0. incidents are sqrt(photons/ns) at the interval start;
        omegas are carrier/reference rad/ns. Return owned sqrt(photons) fields.
        Singular driven blocks retain the full exponential fallback. Nonzero Q
        raises ValueError; nonfinite propagation raises RuntimeError.
        """
        if np.any(self._q):
            raise ValueError("coherent fields require fixed state branches")
        n = len(fields)
        times = np.array([duration])
        _, homogeneous, _ = self.propagators(0.0, 0j, times)[duration]
        final = homogeneous @ fields.ravel()
        for incident, omega in zip(incidents, omegas, strict=True):
            _, _, driven = self.propagators(float(omega), complex(incident), times)[
                duration
            ]
            final += (driven @ np.ones(n)) * np.exp(-1j * omega * duration)
        if not np.isfinite(final).all():
            raise RuntimeError("constant cavity propagation is nonfinite")
        return final.reshape(n, 2)

    def integrated_sums(
        self, model: ReadoutModel, pulse: ReadoutPulse, grids: tuple[SampleGrid, ...]
    ) -> tuple[RealArray, NDArray[np.complex128]] | None:
        """Calibrated flat sums and final populations, or fallback.

        model is the validated model that constructed this line. Grids specify
        physical ADC times and demodulation. Unsupported or ill-conditioned
        spectral sums return None, leaving the original field route in charge.
        """
        if self._integrated is None:
            self._integrated = IntegratedLine(self._q, self._m, self._b, self._output)
        return self._integrated.sums(model, pulse, grids)

    def integrated_batch_sums(self, model, pulses, grids):
        """Calibrated per-content sums/final populations, or individual fallback.

        Inputs share exact timing and per-read sample coordinates; caller bounds
        tiles to 64 flat single-carrier contents. Demodulation may vary.
        """
        if self._integrated is None:
            self._integrated = IntegratedLine(self._q, self._m, self._b, self._output)
        return self._integrated.batch_sums(model, pulses, grids)

    def sampled_trajectory(self, pulse: ReadoutPulse, times: RealArray, omega: float):
        """Emitted sampled-envelope fields/final carrier state, or ODE fallback.

        times are retarded absolute ns, omega is carrier/reference rad/ns.
        Uses affine convolution blocks without assuming diagonalizable rates.
        """
        return SampledLine(self._q, self._m, self._b, self._output, omega).trajectory(
            pulse, times
        )

    def trajectory(
        self, pulse: ReadoutPulse, times: RealArray, omega: float
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]] | None:
        """Project a flat vacuum-start trajectory, or request full-expm fallback.

        times are retarded absolute ns and omega is carrier/reference rad/ns.
        Return outgoing fields (sample, initial state) in the model frame and
        final (state, population/moment component, initial state) in the carrier
        frame. A poorly separated Sylvester solve returns None without repair.
        """
        coupling = self._coupling(omega)
        if coupling is None:
            return None
        elapsed = np.maximum(times - pulse.start_ns, 0.0)
        duration = pulse.end_ns - pulse.start_ns
        incident = pulse.amplitude * np.exp(1j * pulse.phase_rad)
        ep, projected = self._projected(elapsed)
        phase = np.exp(1j * omega * elapsed)
        scattered = (projected @ coupling) * phase[:, None]
        # While driven: L m(t) = b [L exp(M_w t) X - L X exp(Q t)].
        scattered -= np.einsum("j,tji->ti", self._output @ coupling, ep)
        ringdown = elapsed > duration
        if np.any(ringdown):
            after = elapsed[ringdown] - duration
            _, projected_after = self._projected(after)
            ep_on, _ = self._homogeneous_step(duration)
            # After switch-off, subtract the response of an opposite step.
            scattered[ringdown] = (projected[ringdown] @ coupling) * phase[
                ringdown, None
            ] - (
                (projected_after @ coupling) * np.exp(1j * omega * after)[:, None]
            ) @ ep_on
        active = (times >= pulse.start_ns) & (times < pulse.end_ns)
        outgoing = incident * (active[:, None] - scattered)
        outgoing[times < pulse.start_ns] = 0
        outgoing *= np.exp(-1j * omega * elapsed)[:, None]
        state = self._final_state(
            coupling, omega, incident, float(elapsed[-1]), duration
        )
        return outgoing, state

    def _homogeneous_step(self, dt: float):
        if dt not in self._homogeneous:
            self._homogeneous[dt] = (
                np.asarray(scipy.linalg.expm(self._q * dt), dtype=np.complex128),
                np.asarray(scipy.linalg.expm(self._m * dt), dtype=np.complex128),
            )
        return self._homogeneous[dt]

    def _projected(self, durations: RealArray):
        # Only population matrices and one output row per sample survive.
        # Cache exact relative grids, not carrier-dependent dense moments.
        key = durations.tobytes()
        if key not in self._projections:
            unique, inverse = np.unique(durations, return_inverse=True)
            populations = np.asarray(
                [scipy.linalg.expm(self._q * dt) for dt in unique],
                dtype=np.complex128,
            )
            rows = np.asarray(
                [self._output @ scipy.linalg.expm(self._m * dt) for dt in unique]
            )
            self._projections[key] = (populations[inverse], rows[inverse])
        return self._projections[key]

    def _final_state(self, coupling, omega, incident, elapsed, duration):
        driven = min(elapsed, duration)
        after = elapsed - driven
        ep_on, em_on = self._homogeneous_step(driven)
        ep_off, em_off = self._homogeneous_step(after)
        moments = incident * (
            (em_on * np.exp(1j * omega * driven)) @ coupling - coupling @ ep_on
        )
        moments = (em_off * np.exp(1j * omega * after)) @ moments
        n = len(self._q)
        state = np.empty((n, 3, n), dtype=np.complex128)
        state[:, 0, :] = ep_off @ ep_on
        state[:, 1:, :] = moments.reshape(n, 2, n)
        return state

    def propagators(self, omega: float, incident: complex, durations: RealArray):
        """Return population, moment and driven blocks for positive durations.

        Durations are ns, omega is the carrier/reference angular detuning in
        rad/ns, and incident is the flat drive in its carrier frame. Cache only
        frequency-independent exponentials, not every carrier's dense matrices.
        Return a dict keyed by each distinct positive duration, with complex
        population (n,n), moment (2*n,2*n), and driven (2*n,n) arrays. Near-
        singular driven blocks use a full exponential without regularization.
        """
        shifted = self._m + 1j * omega * np.eye(len(self._m))
        coupling = self._coupling(omega) if incident else None
        result = {}
        for dt in np.unique(durations[durations > 0]):
            dt = float(dt)
            ep, base = self._homogeneous_step(dt)
            em = base * np.exp(1j * omega * dt)
            if not incident:
                driven = np.zeros_like(self._b)
            elif coupling is None:
                result[dt] = self._full_step(shifted, incident, dt)
                continue
            else:
                driven = incident * (em @ coupling - coupling @ ep)
            result[dt] = (ep, em, driven)
        return result

    def _coupling(self, omega):
        shifted = self._m + 1j * omega * np.eye(len(self._m))
        triangular = self._r + 1j * omega * np.eye(len(self._r))
        y, scale, info = self._trsyl(triangular, self._s, self._rhs, tranb="C")
        if info != 0 or scale != 1:
            # Singular or scaled solves use the original full exponential.
            return None
        coupling = self._u @ y @ self._w.conj().T
        # Near coincident spectra can make the difference of two large terms
        # lose precision. Use the original exponential rather than regularizing
        # rates, clipping populations or assuming a diagonalizable generator.
        roundoff = np.finfo(float).eps * np.linalg.norm(coupling, ord=np.inf)
        residual = shifted @ coupling - coupling @ self._q - self._b
        if (
            not np.isfinite(coupling).all()
            or roundoff > 1e-13
            or np.linalg.norm(residual, ord=np.inf) > 1e-13
        ):
            return None
        return coupling

    def _full_step(self, shifted, incident, dt):
        n = len(self._q)
        generator = np.zeros((3 * n, 3 * n), dtype=np.complex128)
        generator[:n, :n] = self._q
        generator[n:, n:] = shifted
        generator[n:, :n] = self._b * incident
        step = np.asarray(scipy.linalg.expm(generator * dt), dtype=np.complex128)
        return step[:n, :n], step[n:, n:], step[n:, :n]
