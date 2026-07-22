from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import numpy as np
import scipy as sp
from numpy.typing import NDArray

from ..base import fit_func

_DEFAULT_EDELAY_SEARCH_PERIODS = 2.0
_EDELAY_BRANCH_OVERSAMPLING = 8.0
_EDELAY_BRANCH_CHUNK_SIZE = 512
_MAX_EDELAY_BRANCH_CANDIDATES = 100_001
_EDELAY_AMBIGUITY_MARGIN = 1e-6


def run_complex_refinement(
    residual: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    initial: Sequence[float],
    bounds: tuple[Sequence[float], Sequence[float]],
    *,
    model_name: str,
) -> NDArray[np.float64] | None:
    """Run the shared bounded complex-fit optimizer with explicit fallback."""
    try:
        result = sp.optimize.least_squares(
            residual,
            np.asarray(initial, dtype=np.float64),
            bounds=bounds,
            x_scale=np.ones(len(initial), dtype=np.float64),
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=10_000,
        )
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        warnings.warn(
            f"{model_name} complex refinement failed; using sequential "
            f"initializer ({exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if (
        not result.success
        or not np.all(np.isfinite(result.x))
        or not np.all(np.isfinite(result.fun))
        or np.any(result.active_mask != 0)
    ):
        warnings.warn(
            f"{model_name} complex refinement was non-finite, unsuccessful, or "
            "reached a parameter bound; using sequential initializer",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return np.asarray(result.x, dtype=np.float64)


def get_rough_edelay(
    freqs: NDArray[np.float64], signals: NDArray[np.complex128]
) -> float:
    freq_steps = np.diff(freqs)
    if np.any(freq_steps == 0.0):
        raise ValueError("frequency points must be distinct")
    signal_ratios = signals[1:] / (signals[:-1] + 1e-12)
    local_slopes = np.angle(signal_ratios) / freq_steps
    slope = np.median(local_slopes)

    return -slope / (2 * np.pi)


def _is_uniform_frequency_grid(freq_steps: NDArray[np.float64]) -> bool:
    abs_steps = np.abs(freq_steps)
    return bool(np.allclose(abs_steps, np.median(abs_steps), rtol=1e-8, atol=0.0))


def _aggregate_rough_edelays(
    freq_steps: NDArray[np.float64], rough_edelays: NDArray[np.float64]
) -> float:
    if len(rough_edelays) == 1:
        return float(rough_edelays[0])
    if not _is_uniform_frequency_grid(freq_steps):
        return float(np.median(rough_edelays))

    alias_period = 1.0 / float(np.median(np.abs(freq_steps)))
    alias_angles = 2.0 * np.pi * rough_edelays / alias_period
    resultant = np.mean(np.exp(1j * alias_angles))
    if abs(resultant) < 1e-6:
        warnings.warn(
            "related traces have ambiguous uniform-grid delay aliases; using the "
            "first trace's canonical alias",
            RuntimeWarning,
            stacklevel=2,
        )
        return float(rough_edelays[0])
    return float(np.angle(resultant) * alias_period / (2.0 * np.pi))


def _align_uniform_edelay_aliases(
    freq_steps: NDArray[np.float64],
    edelays: NDArray[np.float64],
    reference: float,
) -> NDArray[np.float64]:
    """Align equivalent uniform-grid delays to the alias nearest ``reference``."""
    if not _is_uniform_frequency_grid(freq_steps):
        return edelays

    alias_period = 1.0 / float(np.median(np.abs(freq_steps)))
    offsets = (edelays - reference + 0.5 * alias_period) % alias_period
    offsets -= 0.5 * alias_period
    return reference + offsets


def _find_edelay_branch(
    freqs: NDArray[np.float64],
    signals: NDArray[np.complex128],
    rough_edelay: float,
    search_radius: float,
) -> float:
    """Find the nonuniform-grid delay branch from adjacent unit-phasor coherence."""
    if not np.isfinite(search_radius) or search_radius <= 0.0:
        raise ValueError(
            "electrical-delay branch search radius must be positive and finite"
        )

    freq_steps = np.diff(freqs)
    if not (np.all(freq_steps > 0.0) or np.all(freq_steps < 0.0)):
        raise ValueError("resonance fit frequencies must be strictly monotonic")
    if signals.ndim == 1:
        signal_rows = signals[None, :]
    elif signals.ndim == 2:
        signal_rows = signals
    else:
        raise ValueError("electrical-delay branch search expects one or two dimensions")
    if signal_rows.shape[-1] != len(freqs):
        raise ValueError(
            "electrical-delay branch search frequency and signal lengths must match"
        )

    amplitudes = np.abs(signal_rows)
    max_amplitudes = np.max(amplitudes, axis=1, keepdims=True)
    if np.any(max_amplitudes <= 0.0):
        raise ValueError("electrical-delay branch search requires non-zero signal rows")
    if _is_uniform_frequency_grid(freq_steps):
        # A uniform grid cannot distinguish delays separated by 1 / |df|. Keep the
        # local alias supplied by get_rough_edelay rather than selecting an arbitrary
        # equivalent branch from a finite global search interval.
        return rough_edelay
    valid = amplitudes > max_amplitudes * 1e-12
    valid_pairs = valid[:, 1:] & valid[:, :-1]
    if np.count_nonzero(valid_pairs) < 3:
        raise ValueError(
            "electrical-delay branch search requires at least three valid signal pairs"
        )

    units = np.zeros_like(signal_rows)
    units[valid] = signal_rows[valid] / amplitudes[valid]
    phase_steps = units[:, 1:] * np.conj(units[:, :-1])
    weighted_phase_steps = np.sum(valid_pairs * phase_steps, axis=0)
    weight_sum = float(np.count_nonzero(valid_pairs))

    span = float(np.ptp(freqs))
    candidate_step = 1.0 / (_EDELAY_BRANCH_OVERSAMPLING * span)
    max_steps_each_side = (_MAX_EDELAY_BRANCH_CANDIDATES - 1) // 2
    if search_radius > max_steps_each_side * candidate_step:
        raise ValueError(
            "electrical-delay branch search exceeds the candidate resource limit; "
            "reduce search_radius or pass an explicit edelay"
        )
    steps_each_side = int(np.floor(search_radius / candidate_step))
    candidate_count = 2 * steps_each_side + 1
    if candidate_count > _MAX_EDELAY_BRANCH_CANDIDATES:
        raise ValueError(
            "electrical-delay branch search would require "
            f"{candidate_count} candidates; reduce search_radius or pass an explicit "
            "edelay"
        )
    offsets = candidate_step * np.arange(
        -steps_each_side, steps_each_side + 1, dtype=np.float64
    )
    candidates = rough_edelay + offsets
    scores = np.empty(candidate_count, dtype=np.float64)
    for start in range(0, candidate_count, _EDELAY_BRANCH_CHUNK_SIZE):
        stop = min(start + _EDELAY_BRANCH_CHUNK_SIZE, candidate_count)
        corrections = np.exp(
            1j * 2.0 * np.pi * candidates[start:stop, None] * freq_steps[None, :]
        )
        scores[start:stop] = np.real(corrections @ weighted_phase_steps) / weight_sum

    best_score = float(np.max(scores))
    tie_atol = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(best_score))
    tied = np.flatnonzero(scores >= best_score - tie_atol)
    best_index = int(tied[np.argmin(np.abs(offsets[tied]))])
    if best_index == 0 or best_index == candidate_count - 1:
        raise ValueError(
            "electrical-delay branch optimum reached the search boundary; increase "
            "search_radius or pass an explicit edelay"
        )

    peak_indices = (
        np.flatnonzero((scores[1:-1] > scores[:-2]) & (scores[1:-1] >= scores[2:])) + 1
    )
    other_peaks = peak_indices[peak_indices != best_index]
    score_margin = (
        best_score - float(np.max(scores[other_peaks]))
        if len(other_peaks) > 0
        else None
    )
    if score_margin is not None and score_margin < _EDELAY_AMBIGUITY_MARGIN:
        warnings.warn(
            "electrical-delay branch search is ambiguous; using the strongest "
            "adjacent-phase-coherence branch",
            RuntimeWarning,
            stacklevel=2,
        )

    return float(candidates[best_index])


def find_edelay_branch(
    freqs: NDArray[np.float64],
    signals: NDArray[np.complex128],
    *,
    search_radius: float | None = None,
) -> float:
    """Return a global electrical-delay branch seed for one or more signal rows.

    ``signals`` may be a single trace or a row stack sharing ``freqs``. With no
    explicit ``search_radius``, the search covers two alias periods of an equivalent
    uniform grid with the same span and sample count. The radius uses the inverse unit
    of ``freqs``. Uniform-grid row aliases are aggregated circularly over ``1/|df|``.
    Invalid, boundary-limited, or oversized searches raise ``ValueError``; an
    ambiguous nonuniform maximum emits ``RuntimeWarning``.
    """
    if freqs.ndim != 1 or signals.ndim not in (1, 2):
        raise ValueError(
            "electrical-delay branch search expects a one-dimensional frequency "
            "axis and one- or two-dimensional signals"
        )
    if signals.shape[-1] != len(freqs):
        raise ValueError(
            "electrical-delay branch search frequency and signal lengths must match"
        )
    if signals.ndim == 2 and signals.shape[0] == 0:
        raise ValueError(
            "electrical-delay branch search requires at least one signal row"
        )
    if len(freqs) < 4:
        raise ValueError(
            "electrical-delay branch search requires at least four samples"
        )
    if not np.all(np.isfinite(freqs)) or not np.all(np.isfinite(signals)):
        raise ValueError("electrical-delay branch search inputs must be finite")
    span = float(np.ptp(freqs))
    if not np.isfinite(span) or span <= 0.0 or len(np.unique(freqs)) != len(freqs):
        raise ValueError(
            "electrical-delay branch search requires distinct frequencies with a "
            "positive span"
        )

    signal_rows = signals[None, :] if signals.ndim == 1 else signals
    rough_edelays = np.asarray(
        [get_rough_edelay(freqs, row) for row in signal_rows],
        dtype=np.float64,
    )
    rough_edelay = _aggregate_rough_edelays(np.diff(freqs), rough_edelays)
    if search_radius is None:
        search_radius = _DEFAULT_EDELAY_SEARCH_PERIODS * (len(freqs) - 1) / span
    return _find_edelay_branch(freqs, signals, rough_edelay, search_radius)


def remove_edelay(
    freqs: NDArray[np.float64], signals: NDArray[np.complex128], edelay: float
) -> NDArray[np.complex128]:
    return np.exp(1j * 2 * np.pi * freqs * edelay) * signals


def calc_M(xs: NDArray[np.float64], ys: NDArray[np.float64]) -> NDArray[np.float64]:
    zs = xs**2 + ys**2
    N = len(zs)
    Mx = np.sum(xs)
    My = np.sum(ys)
    Mz = np.sum(zs)
    Mxx = np.sum(xs**2)
    Myy = np.sum(ys**2)
    Mzz = np.sum(zs**2)
    Mxz = np.sum(xs * zs)
    Myz = np.sum(ys * zs)
    Mxy = np.sum(xs * ys)

    M = np.array(
        [
            [Mzz, Mxz, Myz, Mz],
            [Mxz, Mxx, Mxy, Mx],
            [Myz, Mxy, Myy, My],
            [Mz, Mx, My, N],
        ]
    )
    return M


def fit_circle_params(
    xs: NDArray[np.float64], ys: NDArray[np.float64]
) -> tuple[float, float, float]:
    """[center_x, center_y, radius]"""
    mean_x, mean_y = np.mean(xs), np.mean(ys)
    xs = xs - mean_x
    ys = ys - mean_y

    # calculate M matrix
    M = calc_M(xs, ys)
    B = np.array(
        [
            [0, 0, 0, -2],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-2, 0, 0, 0],
        ]
    )

    eigvals, eigvecs = sp.linalg.eig(M, B)
    eigvals = eigvals.real

    # The exact-circle solution is the eigenvalue nearest zero. Floating-point
    # roundoff can place it just below zero, so filtering for non-negative values
    # selects a different circle even for noiseless data.
    eigvec = eigvecs[:, np.argmin(np.abs(eigvals))]

    A, B, C, D = eigvec
    center_x = mean_x - B / (2 * A)
    center_y = mean_y - C / (2 * A)
    radius = np.sqrt(B**2 + C**2 - 4 * A * D) / abs(2 * A)

    return center_x, center_y, radius


def fit_edelay(
    freqs: NDArray[np.float64],
    signals: NDArray[np.complex128],
    *,
    search_radius: float | None = None,
    branch_seed: float | None = None,
) -> float:
    """Fit electrical delay, resolving nonuniform-grid aliases within a finite radius.

    ``search_radius`` uses the inverse unit of ``freqs`` (microseconds when ``freqs``
    is in MHz). The default covers two average-grid alias periods. Uniform grids retain
    the local canonical alias because their absolute delay branch is not identifiable
    from the sampled data. ``branch_seed`` lets a caller share one previously discovered
    branch across related traces and skips the global search.
    """
    validate_complex_fit_inputs(freqs, signals)
    if branch_seed is not None and search_radius is not None:
        raise ValueError("branch_seed and search_radius are mutually exclusive")
    if branch_seed is None:
        branch_edelay = find_edelay_branch(
            freqs,
            signals,
            search_radius=search_radius,
        )
    elif np.isfinite(branch_seed):
        branch_edelay = branch_seed
    else:
        raise ValueError("electrical-delay branch seed must be finite")
    signals = remove_edelay(freqs, signals, branch_edelay)

    def loss_func(edelay: float) -> float:
        rot_signals = remove_edelay(freqs, signals, edelay)
        xc, yc, r0 = fit_circle_params(rot_signals.real, rot_signals.imag)
        norm_signals = rot_signals - (xc + 1j * yc)

        return np.sum((r0 - np.abs(norm_signals)) ** 2).item()

    fit_range = 5.0 / np.ptp(freqs)
    edelays = np.linspace(-fit_range, fit_range, 1000)
    loss_values = [loss_func(edelay) for edelay in edelays]
    edelay = edelays[np.argmin(loss_values)] + branch_edelay
    edelay = _align_uniform_edelay_aliases(
        np.diff(freqs), np.asarray([edelay]), branch_edelay
    )[0]

    return float(edelay)


def calc_phase(
    signals: NDArray[np.complex128], xc: float, yc: float, axis: int = 0
) -> NDArray[np.float64]:
    return np.unwrap(np.angle(signals - (xc + 1j * yc)), axis=axis)


def phase_func(
    freqs: NDArray[np.float64],
    resonant_f: float,
    Ql: float,
    theta0: float,
) -> NDArray[np.float64]:
    return theta0 + 2 * np.arctan(2 * Ql * (1 - freqs / resonant_f))


def fit_resonant_params(
    freqs: NDArray[np.float64],
    signals: NDArray[np.complex128],
    circle_params: tuple[float, float, float],
    fit_theta0: bool = True,
) -> tuple[float, float, float]:
    """Return ``[resonant_freq, Ql, theta0]`` from circle phase."""
    phases = calc_phase(signals, circle_params[0], circle_params[1])

    phase_slopes = np.gradient(phases, freqs)
    resonance_index = int(np.argmax(np.abs(phase_slopes)))
    init_freq = freqs[resonance_index]
    init_Ql = abs(0.25 * init_freq * phase_slopes[resonance_index])
    init_theta0 = 0.5 * float(np.max(phases) + np.min(phases))

    fixedparams: list[float | None] = [None] * 3
    if not fit_theta0:
        init_theta0 = np.angle(circle_params[0] + 1j * circle_params[1]).item()
        while init_theta0 < np.min(phases):
            init_theta0 += 2 * np.pi
        while init_theta0 > np.max(phases):
            init_theta0 -= 2 * np.pi
        fixedparams[2] = init_theta0

    pOpt, _ = fit_func(
        freqs,
        phases,
        phase_func,
        init_p=[init_freq, init_Ql, init_theta0],
        bounds=(
            [np.min(freqs), 0, init_theta0 - np.pi],
            [np.max(freqs), 5 * init_Ql, init_theta0 + np.pi],
        ),
        fixedparams=fixedparams,
    )

    return (pOpt[0], pOpt[1], pOpt[2])


def validate_complex_fit_inputs(
    freqs: NDArray[np.float64], signals: NDArray[np.complex128]
) -> float:
    """Validate a resonance fit and return its positive frequency span."""
    if freqs.ndim != 1 or signals.ndim != 1:
        raise ValueError(
            "resonance fit expects one-dimensional frequency and signal arrays"
        )
    if freqs.shape != signals.shape:
        raise ValueError(
            "frequency and signal arrays must have the same shape, got "
            f"{freqs.shape} and {signals.shape}"
        )
    if len(freqs) < 4:
        raise ValueError("resonance fit requires at least four samples")
    if not np.all(np.isfinite(freqs)) or not np.all(np.isfinite(signals)):
        raise ValueError("resonance fit inputs must be finite")
    span = float(np.ptp(freqs))
    if span <= 0.0:
        raise ValueError("resonance fit requires a positive frequency span")
    if len(np.unique(freqs)) != len(freqs):
        raise ValueError("resonance fit frequency points must be distinct")
    return span


def remove_background(
    freqs: NDArray[np.float64],
    signals: NDArray[np.complex128],
    *,
    freq: float,
    edelay: float,
    bg_amp_slope: float,
) -> NDArray[np.complex128]:
    """Remove global delay and multiplicative log-amplitude background."""
    return (
        signals
        * np.exp(1j * 2.0 * np.pi * freqs * edelay)
        * np.exp(-bg_amp_slope * (freqs - freq))
    )


def align_phase_to_data(
    phase: float,
    freqs: NDArray[np.float64],
    data_phases: NDArray[np.float64],
    resonant_f: float,
) -> float:
    """Move an analytic phase onto the unwrapped data branch at resonance."""
    order = np.argsort(freqs)
    reference = float(np.interp(resonant_f, freqs[order], data_phases[order]))
    return phase + 2.0 * np.pi * round((reference - phase) / (2.0 * np.pi))


def normalize_signal(
    signals: NDArray[np.complex128],
    circle_params: tuple[float, float, float],
    a0: complex,
) -> tuple[NDArray[np.complex128], tuple[float, float, float]]:
    xc, yc, r0 = circle_params
    center = xc + 1j * yc

    norm_signals = signals / a0
    norm_center = center / a0
    norm_xc, norm_yc = norm_center.real, norm_center.imag
    norm_r0 = r0 / np.abs(a0)

    return norm_signals, (norm_xc, norm_yc, norm_r0)
