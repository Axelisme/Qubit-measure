from __future__ import annotations

from functools import wraps

import numpy as np
from scipy.special import erf
from scipy.signal import savgol_filter
from numpy.typing import NDArray
from typing_extensions import Any, Callable, Optional, Sequence, TypeVar

from zcu_tools.experiment.v2.tracker import KMeansTracker
from zcu_tools.experiment.v2.runner import default_raw2signal_fn
from zcu_tools.program.v2 import ModularProgramV2
from zcu_tools.utils.func_tools import min_interval


def estimate_snr(real_signals: NDArray[np.float64]) -> float:
    # Savitzky-Golay requires an odd window length and window_length > polyorder.
    n = real_signals.shape[-1]
    window_length = min(7, n if n % 2 else n - 1)
    if window_length >= 3:
        smooth_signals = savgol_filter(
            real_signals, window_length=window_length, polyorder=2, axis=-1
        )
    else:
        smooth_signals = real_signals
    noise = np.mean(np.abs(real_signals - smooth_signals))
    return (np.ptp(smooth_signals) / noise).item()


def calc_snr(
    mean_d: NDArray[np.float64],
    center_d: NDArray[np.float64],
    cov_d: NDArray[np.float64],
    ge_axis: int = 0,
) -> NDArray[np.float64]:
    """SNR = erf(separability) × symmetry-about-perpendicular-bisector.

    The bisector is defined by the representative centers of g and e. g's mean
    and covariance are reflected across it; the result is compared to e's
    via the Bhattacharyya coefficient. Perfect mirror symmetry → 1.
    """
    # per-state centers define the bisector; (..., 2)
    med_g = np.take(center_d, 0, axis=ge_axis)
    med_e = np.take(center_d, 1, axis=ge_axis)
    diff = med_e - med_g
    peak_contrast: NDArray[np.float64] = np.linalg.norm(diff, axis=-1)

    # pooled covariance (..., 2, 2); σ² = trace/2
    pooled_cov = np.mean(cov_d, axis=ge_axis)
    trace = pooled_cov[..., 0, 0] + pooled_cov[..., 1, 1]
    sigma = np.sqrt(np.clip(trace / 2.0, 1e-24, None))
    erf_factor = erf(peak_contrast / (np.sqrt(32) * sigma))

    # reflection R = I - 2 n nᵀ across the perpendicular bisector
    n = diff / np.clip(peak_contrast, 1e-24, None)[..., None]  # (..., 2)
    eye = np.broadcast_to(np.eye(2), n.shape[:-1] + (2, 2))
    R = eye - 2.0 * n[..., :, None] * n[..., None, :]  # (..., 2, 2)
    midpoint = 0.5 * (med_g + med_e)  # (..., 2)

    # reflect g's gaussian: μ_g' = R(μ_g - m) + m, Σ_g' = R Σ_g Rᵀ
    mean_g = np.take(mean_d, 0, axis=ge_axis)
    mean_e = np.take(mean_d, 1, axis=ge_axis)
    cov_g = np.take(cov_d, 0, axis=ge_axis)
    cov_e = np.take(cov_d, 1, axis=ge_axis)

    mean_g_refl = np.einsum("...ij,...j->...i", R, mean_g - midpoint) + midpoint
    cov_g_refl = np.einsum("...ij,...jk,...lk->...il", R, cov_g, R)

    # Bhattacharyya distance between N(μ_g', Σ_g') and N(μ_e, Σ_e)
    sigma_bar = 0.5 * (cov_g_refl + cov_e)
    delta = (mean_e - mean_g_refl)[..., None]  # (..., 2, 1)
    sol = np.linalg.solve(sigma_bar, delta)
    mahal = np.einsum("...ij,...ij->...", delta, sol)  # (μ_e-μ_g')ᵀ Σ̃⁻¹ (...)

    det_refl = np.clip(np.linalg.det(cov_g_refl), 1e-24, None)
    det_e = np.clip(np.linalg.det(cov_e), 1e-24, None)
    det_bar = np.clip(np.linalg.det(sigma_bar), 1e-24, None)
    shape_term = 0.5 * np.log(det_bar / np.sqrt(det_refl * det_e))

    d_b = mahal / 8.0 + shape_term
    symmetry = np.exp(-np.clip(d_b, 0.0, None))

    return erf_factor * symmetry


def snr_as_signal(
    raw: Sequence[KMeansTracker],
    ge_axis: int = 0,
) -> NDArray[np.float64]:
    """Compute SNR from a sequence of PCATrackers (one per readout channel).

    Currently uses the first tracker only (single readout); extend if
    multi-channel SNR is ever needed.
    """
    tracker = raw[0]
    mean_d = tracker.mean  # (..., 2)
    center_d = tracker.leader_center  # (..., 2)
    cov_d = tracker.covariance  # (..., 2, 2)
    assert mean_d is not None
    assert center_d.shape[ge_axis] == 2
    assert mean_d.shape == center_d.shape
    assert cov_d.shape[:-1] == center_d.shape

    return calc_snr(mean_d, center_d, cov_d, ge_axis=ge_axis)


T_RawResult = TypeVar("T_RawResult")


def wrap_earlystop_check(
    prog: ModularProgramV2,
    callback_fn: Callable[[int, T_RawResult], None],
    snr_threshold: Optional[float],
    signal2real_fn: Callable[[np.ndarray], np.ndarray],
    raw2signal_fn: Callable[[T_RawResult], np.ndarray] = default_raw2signal_fn,
    after_check: Optional[Callable[[float], Any]] = None,
    check_interval: Optional[float] = 0.1,
) -> Callable[[int, T_RawResult], None]:
    if snr_threshold is None:
        return callback_fn

    def check_snr(raw: T_RawResult) -> None:
        signals = raw2signal_fn(raw)
        snr = estimate_snr(signal2real_fn(signals))
        if snr >= snr_threshold:
            prog.set_early_stop(silent=True)

        if after_check is not None:
            after_check(snr)

    check_snr = min_interval(check_snr, check_interval)

    @wraps(callback_fn)
    def wrapped_callback_fn(i: int, raw: T_RawResult) -> None:
        callback_fn(i, raw)
        check_snr(raw)

    return wrapped_callback_fn
