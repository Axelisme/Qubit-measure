"""Explicit joint qubit/two-resonator prototype, separate from Engine and ADC."""

from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray

from ._joint_fock_diagnostic import diagnostic_row
from ._joint_fock_two_level import two_level_row
from .secular import SecularModel

RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class JointFockDiagnostic:
    """Opt in to a small dynamic projection with numeric preparation.

    model: full public SecularModel, including all virtual intermediate states.
        Rates/pure dephasing and driven coefficients come from this model.
    drive_port: integer index into model.drives, for pi and probe.
    pi_gain, probe_gain: finite real amplitudes of that port.
    retained_indices: exactly (0,2) or (0,1,2,3), in ascending energy order.
        Dynamic levels outside these indices are omitted without a bath sink.
        RWA coefficients are built on the FULL model before selecting indices.

    step_ns: fixed RK4 step upper bound in ns for all numeric preparation and
        probe segments; finite 0 < step_ns <= 1, default 1 ns. Each segment
        uses ceil(duration / step_ns) equal steps. No automatic reduction or
        retry occurs; nonfinite densities raise FloatingPointError.

    Invalid indices, port, gains or step_ns raise ValueError. This is a numerical
    diagnostic, not an acquisition backend or a pure truncation-error control:
    it changes analytic one-jump preparation to numeric vacuum/pi/prefill/probe,
    and includes physical vacuum in the sampled coherent cutoff heuristic.
    """

    model: SecularModel
    drive_port: int
    pi_gain: float
    probe_gain: float
    retained_indices: tuple[int, ...] = (0, 1, 2, 3)
    step_ns: float = 1.0

    def __post_init__(self) -> None:
        if self.retained_indices not in ((0, 2), (0, 1, 2, 3)) or max(
            self.retained_indices
        ) >= len(self.model.energies_ghz):
            raise ValueError("retained_indices must be (0,2) or (0,1,2,3) within model")
        if not 0 <= self.drive_port < len(self.model.drives):
            raise ValueError("drive_port must index model.drives")
        if not np.isfinite([self.pi_gain, self.probe_gain]).all():
            raise ValueError("diagnostic gains must be finite")
        if not np.isfinite(self.step_ns) or not 0 < self.step_ns <= 1:
            raise ValueError("step_ns must be finite and satisfy 0 < step_ns <= 1")


@dataclass(frozen=True)
class JointFockRequest:
    """Numeric two-state spectrum in GHz, ns and radians, with two preparations.

    frequencies_ghz: (2,2) absolute conditional cavity frequencies, rows g/e.
        With diagnostic, shape is (M,2), in the full model's state order.
    kappa_external_per_ns, kappa_internal_per_ns: (2,) energy decay rates.
        The external ports share reciprocal co-located two-port damping.
    resonator_carriers_ghz, qubit_carriers_ghz: nonempty 1-D scan coordinates.
    qubit_frequency_ghz: physical g/e gap (FTF dynamic indices [0,2]).
    incident_per_sqrt_ns: real coherent incident field, constant during prefill
        and probe. Each rf starts in physical vacuum, independently.
    down_per_ns, up_per_ns: direct e->g and g->e rates, not outside-level exit.
    gamma2_per_ns: full intrinsic coherence decay, at least (down+up)/2.
    pi_duration_ns, pi_rabi_rad_per_ns: finite resonant preparation pulse on
        ground before cavity illumination. The other preparation stays ground.
    prefill_duration_ns: illumination before the weak probe; analytic one-jump
        preparation requires up_per_ns * prefill_duration_ns <= 1e-12.
    probe_duration_ns, probe_rabi_rad_per_ns: constant weak qubit probe.
        Carriers reset locally at probe onset, as in the reference protocol.

    Arrays are copied read-only. Nonfinite/negative rates or durations, bad
    shapes, negative pure dephasing and nonnegligible upward prefill raise
    ValueError. There are no acquisition, ADC, noise or environment inputs.
    diagnostic: None selects the two-state analytic preparation. An explicit
        JointFockDiagnostic selects numeric preparation, full-model coefficients
        and its retained indices. In that case down/up/gamma2 and both scalar
        rabi parameters are unused; model baths and diagnostic gains replace
        them. qubit_frequency_ghz must equal the full model's [0,2] gap.
    """

    frequencies_ghz: RealArray
    kappa_external_per_ns: RealArray
    kappa_internal_per_ns: RealArray
    resonator_carriers_ghz: RealArray
    qubit_carriers_ghz: RealArray
    qubit_frequency_ghz: float
    incident_per_sqrt_ns: float
    down_per_ns: float
    up_per_ns: float
    gamma2_per_ns: float
    pi_duration_ns: float
    pi_rabi_rad_per_ns: float
    prefill_duration_ns: float
    probe_duration_ns: float
    probe_rabi_rad_per_ns: float
    diagnostic: JointFockDiagnostic | None = None

    def __post_init__(self) -> None:
        for name, shape in (
            (
                "frequencies_ghz",
                (
                    2
                    if self.diagnostic is None
                    else len(self.diagnostic.model.energies_ghz),
                    2,
                ),
            ),
            ("kappa_external_per_ns", (2,)),
            ("kappa_internal_per_ns", (2,)),
        ):
            value = _owned(getattr(self, name), name)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if name.startswith("kappa") and np.any(value < 0):
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        for name in ("resonator_carriers_ghz", "qubit_carriers_ghz"):
            value = _owned(getattr(self, name), name)
            if value.ndim != 1 or not value.size:
                raise ValueError(f"{name} must be a nonempty vector")
            object.__setattr__(self, name, value)
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray) or field.name == "diagnostic":
                continue
            if not np.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
            if (
                field.name.endswith("_per_ns") or field.name.endswith("_duration_ns")
            ) and value < 0:
                raise ValueError(f"{field.name} must be nonnegative")
        self._validate_preparation()

    def _validate_preparation(self) -> None:
        if self.gamma2_per_ns < (self.down_per_ns + self.up_per_ns) / 2:
            raise ValueError("gamma2_per_ns implies negative pure dephasing")
        if self.diagnostic is not None:
            gap = (
                self.diagnostic.model.energies_ghz[2]
                - self.diagnostic.model.energies_ghz[0]
            )
            if not np.isclose(self.qubit_frequency_ghz, gap, rtol=0, atol=1e-12):
                raise ValueError(
                    "qubit_frequency_ghz must match diagnostic model [0,2] gap"
                )
        elif self.up_per_ns * self.prefill_duration_ns > 1e-12:
            raise ValueError("One-jump prefill requires up * prefill duration <= 1e-12")


@dataclass(frozen=True)
class JointFockResult:
    """Raw populations and solver diagnostics, never clipped or renormalized.

    populations: (rf, preparation, qf, state), with preparation ground then
        finite-pi. State order follows state_indices. Values are cavity partial
        traces, with no normalization or clipping.
    cutoffs: one total cavity excitation N per rf, after conservative N+1.
    trace_error, hermitian_error, min_eigenvalue, edge_population: real arrays
        (rf, preparation, qf) of final joint density trace-minus-one magnitude,
        maximum Hermitian defect, minimum Hermitianized eigenvalue, and
        population on n1+n2=N. These are observations, not error certificates.
    sampled_pointer_means: (rf,3) maximum sampled displaced photon means for
        g/e/no-more-than-one-prefill-jump paths, used only for cutoff selection.
    initial_trace_error, initial_edge_population: (rf,preparation) diagnostics
        at probe onset before any qf propagation.
    vacuum_trace_error, vacuum_edge_population: (rf,) trace defect and total-N
        edge population of the normalized finite coherent vector representing
        physical cavity vacuum. The two-state path does not propagate this
        vacuum; it constructs analytic prefill directly.
    state_indices: physical state labels, (0,2) for the analytic main path or
        diagnostic.retained_indices for the explicit diagnostic.
    """

    populations: RealArray
    cutoffs: tuple[int, ...]
    trace_error: RealArray
    hermitian_error: RealArray
    min_eigenvalue: RealArray
    edge_population: RealArray
    sampled_pointer_means: RealArray
    initial_trace_error: RealArray
    initial_edge_population: RealArray
    vacuum_trace_error: RealArray
    vacuum_edge_population: RealArray
    state_indices: tuple[int, ...]


def _owned(value: RealArray, name: str) -> RealArray:
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    result.flags.writeable = False
    return result


def joint_fock_spectrum(request: JointFockRequest) -> JointFockResult:
    """Compute an explicit joint spectrum without selecting an acquisition backend.

    Each rf starts from physical vacuum and shares its analytic prefill across
    all probes and both preparations. A common displaced total-excitation basis
    uses the sampled Poisson-tail<=1e-3 cutoff plus one. No-flip g/e pointers
    use 129 samples from prefill end through probe end; prefill one-jump paths
    use 65 jump times and 33 subsequent times. This sampled coherent heuristic
    is not a complete-density truncation bound. Cutoffs above 20 raise ValueError.

    The analytic-preparation path uses fixed RK4 <=1 ns. This does not guarantee
    positivity or global accuracy for arbitrary frequency/drive scales. Populations and
    diagnostics are not clipped, normalized or repaired. Singular driven
    steady pointers raise ValueError; nonfinite propagation raises
    FloatingPointError. No state/cache survives a call, no solver fallback
    occurs, and acquisition, ADC and existing Engine routes are unchanged.

    With request.diagnostic, all RWA virtual sums use its full SecularModel
    before projection to at most four dynamic levels. Physical vacuum is a
    normalized finite coherent vector in the same displaced basis. The scan
    starts at t=0, including that vacuum, and rejects an initial Poisson tail
    above 1e-3 or an unrepresentable vector. Numeric finite-pi, prefill and probe
    use the projected baths and equal RK4 steps <= diagnostic.step_ns in each
    segment, with local carrier resets and Schrödinger endpoint restoration.
    The requested step is never reduced automatically and failures are not retried.
    No-flip and one-jump pointers still use physical states [0,2], not every
    multilevel jump history. This is a sampled
    coherent heuristic, not a complete-density error bound. Preparation and
    cutoff differ from the two-state analytic path, so their difference is
    not a pure dynamic-truncation error measurement.
    """
    diagnostic = request.diagnostic
    indices = (0, 2) if diagnostic is None else diagnostic.retained_indices
    rows = [
        two_level_row(request, float(rf))
        if diagnostic is None
        else diagnostic_row(request, float(rf), diagnostic)
        for rf in request.resonator_carriers_ghz
    ]
    populations, traces, hermitian, eigenvalues, edges = [], [], [], [], []
    for row in rows:
        density, size = row.densities, len(row.basis)
        diagonal = density.diagonal(axis1=-2, axis2=-1).real
        populations.append(
            diagonal.reshape(
                2, len(request.qubit_carriers_ghz), len(indices), size
            ).sum(axis=-1)
        )
        traces.append(abs(np.trace(density, axis1=-2, axis2=-1) - 1))
        hermitian.append(
            np.max(abs(density - density.conj().swapaxes(-1, -2)), axis=(-1, -2))
        )
        eigenvalues.append(
            np.linalg.eigvalsh((density + density.conj().swapaxes(-1, -2)) / 2).min(
                axis=-1
            )
        )
        order = max(n + m for n, m in row.basis)
        edge = np.tile([n + m == order for n, m in row.basis], len(indices))
        edges.append(diagonal[..., edge].sum(axis=-1))
    return JointFockResult(
        populations=np.array(populations),
        cutoffs=tuple(max(n + m for n, m in row.basis) for row in rows),
        trace_error=np.array(traces),
        hermitian_error=np.array(hermitian),
        min_eigenvalue=np.array(eigenvalues),
        edge_population=np.array(edges),
        sampled_pointer_means=np.array([row.sampled_means for row in rows]),
        initial_trace_error=np.array([row.initial_trace_error for row in rows]),
        initial_edge_population=np.array([row.initial_edge_population for row in rows]),
        vacuum_trace_error=np.array([row.vacuum_trace_error for row in rows]),
        vacuum_edge_population=np.array([row.vacuum_edge_population for row in rows]),
        state_indices=indices,
    )
