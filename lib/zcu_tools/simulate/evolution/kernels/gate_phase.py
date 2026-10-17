"""Model-only phase convergence for short gate-cache Hilbert preparation."""

import numpy as np
from numpy.typing import NDArray

from ..kernel import SegmentTable
from ..secular import SecularModel
from ._gate_endpoints import build_endpoints

# Longer gates retain the original sixteen-node preparation.
MAX_DURATION_NS = 30.0


def _interpolate(
    samples: NDArray[np.complex128], phases: NDArray[np.float64]
) -> NDArray[np.complex128]:
    count = samples.shape[1]
    modes = np.fft.fftfreq(count) * count
    coefficients = np.fft.fft(samples, axis=1) / count
    return np.einsum(
        "pm,smij->spij", np.exp(1j * phases[:, None] * modes), coefficients
    )


def build(
    model: SecularModel, table: SegmentTable, row: int, budget: float
) -> NDArray[np.complex128] | None:
    """Return checked (4,16,M,M) coefficients, or None to request legacy build.

    Require a sampled single tone with duration in (0, MAX_DURATION_NS] ns and
    a finite positive budget. Integrate six equispaced phases and two independent probes
    (pi/6 and sqrt(3) radians), using unchanged temporal tolerances and knot
    restarts. Every Strang endpoint must match both probes in full-unitary
    Frobenius norm within min(1e-4, budget*0.1). A failed phase check returns None;
    the caller must build its original sixteen-node payload instead.

    Retract interpolated unitaries at the original sixteen phase nodes before
    Fourier transformation. No density is accessed or normalized. This empirical
    convergence check is not a rigorous global bound for arbitrary harmonics.
    Invalid rows/budgets raise ValueError; numerical failures propagate.
    """
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("phase budget must be finite and positive")
    if table.tones[row] != 1 or not 0 < table.duration_ns[row] <= MAX_DURATION_NS:
        raise ValueError("phase preparation requires a short single-tone gate")
    envelope = table.envelopes[int(table.tone_envelope[row, 0])]
    if envelope.kind != "sampled":
        raise ValueError("phase preparation requires a sampled envelope")
    phases = np.asarray(np.arange(6) * (2 * np.pi / 6), dtype=np.float64)
    # The irrational second probe avoids sharing a single rational phase grid's
    # harmonic aliases. Both probes share the adaptive integration batch.
    probes = np.array([np.pi / 6, np.sqrt(3)], dtype=np.float64)
    samples = build_endpoints(model, table, row, budget, np.r_[phases, probes])
    predicted = _interpolate(samples[:, :6], probes)
    defect = float(np.max(np.linalg.norm(predicted - samples[:, 6:], axis=(-2, -1))))
    if defect > min(1e-4, budget * 0.1):
        return None
    # Polar projection removes radial interpolation error, not density error.
    full_phases = np.asarray(np.arange(16) * (2 * np.pi / 16), dtype=np.float64)
    expanded = _interpolate(samples[:, :6], full_phases)
    left, _, right = np.linalg.svd(expanded)
    return np.fft.fft(left @ right, axis=1) / 16
