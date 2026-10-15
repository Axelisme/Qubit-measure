"""Offline interpolation proof against independent complex128 exponentials."""

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fminbound

from .batch import CPUReference, PlateauFamily
from .solver import propagator_power
from .state import ComplexMatrix


@dataclass(frozen=True)
class ValidationEvidence:
    """Measured absolute numerical errors for one published plateau table.

    density_abs: largest density element error, covering populations/coherences
        of basis and Fourier-superposition inputs at 8 independent heldouts.
    long_abs: same after 512 identical propagators. phase_abs: coherence angle
        difference in radians for reference magnitude>1e-6. continuation_abs:
        density element error after two different heldout blocks.
    invariant_abs: max element of U512†U512-I, not gate infidelity.
    trace_abs: largest |trace(rho)-1| over basis/Fourier-superposition inputs
        at heldouts, after two different blocks, and after 512 repeats.
    calibration_scale_abs: recovered dimensionless common amplitude-scale error
        from a heldout gate's full density observables, fitted in the same range.
    gpu_node_abs: complex element error at all nodes versus CPU SciPy expm.
    References share the exact model, basis and quantized controls. These do not
    estimate basis truncation, bath approximation or physical waveform errors.
    """

    density_abs: float
    long_abs: float
    phase_abs: float
    continuation_abs: float
    invariant_abs: float
    calibration_scale_abs: float
    gpu_node_abs: float
    trace_abs: float


@dataclass(frozen=True)
class ValidationLimits:
    """Absolute publication limits, never relaxed automatically.

    density_abs: single-use density/population/coherence max element error.
    long_abs: same after 512 repeats. phase_abs: superposition coherence angle
    error in radians where reference magnitude>1e-6. calibration_abs: recovered
    dimensionless amplitude-scale error. gpu_abs: GPU-vs-CPU node element error.
    Trace/unitarity are independently required <=2e-9 (no output repair).
    These are numerical limits for this family, not physical-model certification.
    """

    density_abs: float = 2e-9
    long_abs: float = 2e-7
    phase_abs: float = 2e-8
    calibration_abs: float = 2e-5
    gpu_abs: float = 2e-11

    def __post_init__(self) -> None:
        if (
            not np.isfinite(list(asdict(self).values())).all()
            or min(asdict(self).values()) <= 0
        ):
            raise ValueError("validation limits must be positive finite")


def interpolate(
    nodes: NDArray[np.float64], values: NDArray[np.complex128], scale: float
) -> ComplexMatrix:
    """Polynomial interpolation at finite in-range scale; never extrapolates.

    Small fixed degree (4/6) uses direct Lagrange weights, with exact node hits.
    Values are not projected to unitaries or otherwise repaired.
    """
    if not np.isfinite(scale) or not nodes[0] <= scale <= nodes[-1]:
        raise ValueError("interpolation scale outside table range")
    weights = np.ones(len(nodes))
    for i, node in enumerate(nodes):
        for j, other in enumerate(nodes):
            if i != j:
                weights[i] *= (scale - other) / (node - other)
    return np.asarray(np.einsum("i,ijk->jk", weights, values), dtype=np.complex128)


def _densities(u: ComplexMatrix) -> NDArray[np.complex128]:
    n = len(u)
    # All basis inputs plus Fourier superpositions exercise populations and phase.
    columns = np.concatenate((np.eye(n), np.fft.fft(np.eye(n)) / np.sqrt(n)), axis=1)
    states = u @ columns
    return np.einsum("ik,jk->kij", states, states.conj())


def _phase_error(
    actual: NDArray[np.complex128], expected: NDArray[np.complex128]
) -> float:
    mask = np.abs(expected) > 1e-6
    return float(np.max(np.abs(np.angle(actual[mask] * expected[mask].conj()))))


def validate_table(
    family: PlateauFamily,
    nodes: NDArray[np.float64],
    values: NDArray[np.complex128],
    limits: ValidationLimits,
) -> tuple[tuple[str, float], ...]:
    """Validate heldouts, superposition phase, composition, 512 repeats and scale fit.

    Reference is scipy expm, independent from the OpenCL Taylor exponential.
    Heldouts are not interpolation nodes. Fits recover a known heldout gate
    amplitude from complete density observables; this is not gate infidelity.
    Trace is checked independently for every basis/Fourier-superposition input
    at heldouts, after two different blocks and after 512 repeats, without repair.
    Raise ValueError without publication on any failed numerical limit.
    """
    lower, upper = float(nodes[0]), float(nodes[-1])
    scales = lower + (upper - lower) * np.array(
        [0.071, 0.193, 0.317, 0.431, 0.587, 0.719, 0.863, 0.947]
    )
    references = CPUReference().compute(family, scales).propagators
    density_error = long_error = phase_error = invariant = continuation = trace = 0.0
    for scale, reference in zip(scales, references, strict=True):
        actual = interpolate(nodes, values, float(scale))
        rho, expected = _densities(actual), _densities(reference)
        density_error = max(density_error, float(np.max(np.abs(rho - expected))))
        phase_error = max(phase_error, _phase_error(rho, expected))
        long_u, long_ref = (
            propagator_power(actual, 512),
            propagator_power(reference, 512),
        )
        long_rho = _densities(long_u)
        long_error = max(
            long_error, float(np.max(np.abs(long_rho - _densities(long_ref))))
        )
        invariant = max(
            invariant,
            float(np.max(np.abs(long_u.conj().T @ long_u - np.eye(len(actual))))),
        )
        # Two different blocks acting on a phase-bearing superposition.
        next_u = interpolate(nodes, values, float(scales[0]))
        continued_rho = _densities(next_u @ actual)
        continuation = max(
            continuation,
            float(
                np.max(np.abs(continued_rho - _densities(references[0] @ reference)))
            ),
        )
        for densities in (rho, continued_rho, long_rho):
            trace = max(
                trace,
                float(np.max(np.abs(np.trace(densities, axis1=1, axis2=2) - 1))),
            )
    target_scale = float(scales[3])
    target = _densities(references[3])

    def objective(scale: float) -> float:
        delta = _densities(interpolate(nodes, values, scale)) - target
        return float(np.sum(np.abs(delta) ** 2))

    fitted, _, status, _ = fminbound(
        objective, lower, upper, xtol=1e-13, full_output=True, disp=0
    )
    calibration = abs(float(fitted) - target_scale)
    metrics = (
        ("density", density_error),
        ("long", long_error),
        ("phase", phase_error),
        ("continuation", continuation),
        ("invariant", invariant),
        ("calibration", calibration),
        ("trace", trace),
    )
    bounds = (
        limits.density_abs,
        limits.long_abs,
        limits.phase_abs,
        limits.density_abs,
        2e-9,
        limits.calibration_abs,
        2e-9,
    )
    if status != 0 or any(
        value > bound for (_, value), bound in zip(metrics, bounds, strict=True)
    ):
        raise ValueError(f"table validation failed: {metrics}")
    return metrics
