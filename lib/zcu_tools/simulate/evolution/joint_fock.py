"""Explicit joint qubit/two-resonator prototype, separate from Engine and ADC."""

from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray

from ._joint_fock_two_level import two_level_row

RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class JointFockRequest:
    """Numeric two-state spectrum in GHz, ns and radians, with two preparations.

    frequencies_ghz: (2,2) absolute conditional cavity frequencies, rows g/e.
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

    def __post_init__(self) -> None:
        for name, shape in (
            ("frequencies_ghz", (2, 2)),
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
            if isinstance(value, np.ndarray):
                continue
            if not np.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
            if (
                field.name.endswith("_per_ns") or field.name.endswith("_duration_ns")
            ) and value < 0:
                raise ValueError(f"{field.name} must be nonnegative")
        if self.gamma2_per_ns < (self.down_per_ns + self.up_per_ns) / 2:
            raise ValueError("gamma2_per_ns implies negative pure dephasing")
        if self.up_per_ns * self.prefill_duration_ns > 1e-12:
            raise ValueError("One-jump prefill requires up * prefill duration <= 1e-12")


@dataclass(frozen=True)
class JointFockResult:
    """Raw populations and solver diagnostics, never clipped or renormalized.

    populations: (rf, preparation, qf, state), with preparation ground then
        finite-pi and state g then e. Values are cavity partial traces.
    cutoffs: one total cavity excitation N per rf, after conservative N+1.
    trace_error, hermitian_error, min_eigenvalue, edge_population: real arrays
        (rf, preparation, qf) of final joint density trace-minus-one magnitude,
        maximum Hermitian defect, minimum Hermitianized eigenvalue, and
        population on n1+n2=N. These are observations, not error certificates.
    sampled_pointer_means: (rf,3) maximum sampled displaced photon means for
        g/e/no-more-than-one-prefill-jump paths, used only for cutoff selection.
    initial_trace_error, initial_edge_population: (rf,preparation) diagnostics
        at probe onset before any qf propagation.
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

    Constant CSR generators use RK4 <=1 ns. This is not a positivity or global
    accuracy guarantee for arbitrary frequency/drive scales. Populations and
    diagnostics are not clipped, normalized or repaired. Singular driven
    steady pointers raise ValueError; nonfinite propagation raises
    FloatingPointError. No state/cache survives a call, no solver fallback
    occurs, and acquisition, ADC and existing Engine routes are unchanged.
    """
    rows = [two_level_row(request, float(rf)) for rf in request.resonator_carriers_ghz]
    populations, traces, hermitian, eigenvalues, edges = [], [], [], [], []
    for row in rows:
        density, size = row.densities, len(row.basis)
        diagonal = density.diagonal(axis1=-2, axis2=-1).real
        populations.append(
            diagonal.reshape(2, len(request.qubit_carriers_ghz), 2, size).sum(axis=-1)
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
        edge = np.tile([n + m == order for n, m in row.basis], 2)
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
    )
