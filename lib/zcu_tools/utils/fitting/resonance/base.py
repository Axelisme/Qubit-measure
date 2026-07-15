from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import numpy as np
import scipy as sp
from numpy.typing import NDArray

from ..base import fit_func


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


def fit_edelay(freqs: NDArray[np.float64], signals: NDArray[np.complex128]) -> float:
    rough_edelay = get_rough_edelay(freqs, signals)
    signals = remove_edelay(freqs, signals, rough_edelay)

    def loss_func(edelay: float) -> float:
        rot_signals = remove_edelay(freqs, signals, edelay)
        xc, yc, r0 = fit_circle_params(rot_signals.real, rot_signals.imag)
        norm_signals = rot_signals - (xc + 1j * yc)

        return np.sum((r0 - np.abs(norm_signals)) ** 2).item()

    fit_range = 5.0 / np.ptp(freqs)
    edelays = np.linspace(-fit_range, fit_range, 1000)
    loss_values = [loss_func(edelay) for edelay in edelays]
    edelay = edelays[np.argmin(loss_values)] + rough_edelay

    return edelay


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
