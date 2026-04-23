from __future__ import annotations

from functools import wraps

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter
from scipy.special import erf
from typing_extensions import Any, Callable, Optional, Sequence, TypeVar

from zcu_tools.experiment.v2.runner import default_raw2signal_fn
from zcu_tools.experiment.v2.utils.tracker import MomentTracker
from zcu_tools.program.v2 import ModularProgramV2
from zcu_tools.utils.func_tools import min_interval

DISC_WEIGHT = 1.0
SYM_WEIGHT = 8.0


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
    cov_d: NDArray[np.float64],
    m3_d: NDArray[np.float64],
    ge_axis: int = 0,
    *,
    disc_weight: float = DISC_WEIGHT,
    sym_weight: float = SYM_WEIGHT,
) -> NDArray[np.float64]:
    """SNR score combining a discrimination term and a symmetry term.

    Inputs:
        mean_d: shape (..., 2, 2)        — per-state mean on ge_axis
        cov_d:  shape (..., 2, 2, 2)     — per-state 2x2 covariance
        m3_d:   shape (..., 2, 2, 2, 2)  — per-state 3rd central moment tensor

    Returns weighted score in [0, 1]:
        disc = ½·(erf(Δμ / (2√2 σ_g)) + erf(Δμ / (2√2 σ_e)))
        sym  = exp(−½·(skew_g + skew_e)² − ½·ln²(σ_g / σ_e))
        snr  = disc**disc_weight * sym**sym_weight
    with all σ/skew computed on the 1D projection onto the g→e axis.
    """
    if disc_weight < 0.0 or sym_weight < 0.0:
        raise ValueError("disc_weight and sym_weight must be non-negative")

    mean_g = np.take(mean_d, 0, axis=ge_axis)
    mean_e = np.take(mean_d, 1, axis=ge_axis)
    cov_g = np.take(cov_d, 0, axis=ge_axis)
    cov_e = np.take(cov_d, 1, axis=ge_axis)
    m3_g = np.take(m3_d, 0, axis=ge_axis)
    m3_e = np.take(m3_d, 1, axis=ge_axis)

    axis_vec = mean_e - mean_g  # (..., 2)
    delta_mu = np.linalg.norm(axis_vec, axis=-1)  # (...)
    axis = axis_vec / np.clip(delta_mu, 1e-24, None)[..., None]

    var_g = np.clip(np.einsum("...i,...ij,...j->...", axis, cov_g, axis), 1e-24, None)
    var_e = np.clip(np.einsum("...i,...ij,...j->...", axis, cov_e, axis), 1e-24, None)
    sigma_g = np.sqrt(var_g)
    sigma_e = np.sqrt(var_e)

    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    disc = 0.5 * (
        erf((delta_mu / (4.0 * sigma_g)) * inv_sqrt2)
        + erf((delta_mu / (4.0 * sigma_e)) * inv_sqrt2)
    )
    disc = np.clip(disc, 0.0, 1.0)

    proj_m3_g = np.einsum("...i,...j,...k,...ijk->...", axis, axis, axis, m3_g)
    proj_m3_e = np.einsum("...i,...j,...k,...ijk->...", axis, axis, axis, m3_e)
    skew_g = proj_m3_g / (sigma_g**3)
    skew_e = proj_m3_e / (sigma_e**3)

    d_sym = 0.5 * (skew_g + skew_e) ** 2 + 0.5 * np.log(sigma_g / sigma_e) ** 2
    sym = np.exp(-d_sym)

    return (disc**disc_weight) * (sym**sym_weight)


def snr_as_signal(
    raw: Sequence[MomentTracker],
    ge_axis: int = 0,
) -> NDArray[np.float64]:
    """Compute SNR from the first readout channel tracker."""
    tracker = raw[0]
    mean_d = tracker.mean
    cov_d = tracker.covariance
    m3_d = tracker.third_moment
    assert mean_d is not None
    assert mean_d.shape[ge_axis] == 2
    return calc_snr(mean_d, cov_d, m3_d, ge_axis=ge_axis)


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    sigma = 1.0
    n = 12000

    def _isotropic(center: list[float], std: float, size: int) -> NDArray[np.float64]:
        return rng.normal(loc=np.asarray(center), scale=std, size=(size, 2))

    def _third_moment(samples: NDArray[np.float64]) -> NDArray[np.float64]:
        centered = samples - samples.mean(axis=0, keepdims=True)
        return (
            np.einsum("mi,mj,mk->ijk", centered, centered, centered) / centered.shape[0]
        )

    def _calc_from_samples(g: NDArray[np.float64], e: NDArray[np.float64]) -> float:
        mean = np.stack([g.mean(axis=0), e.mean(axis=0)], axis=0)
        cov = np.stack([np.cov(g, rowvar=False), np.cov(e, rowvar=False)], axis=0)
        m3 = np.stack([_third_moment(g), _third_moment(e)], axis=0)
        return float(calc_snr(mean, cov, m3, ge_axis=0))

    def _score_third_contamination(
        separation: float, third_strength: float, mode: str
    ) -> float:
        if mode == "e_only":
            n_g_third = 0
            n_e_third = int(round(n * third_strength))
        elif mode == "ge_both":
            n_third_each = int(round(n * third_strength / 2.0))
            n_g_third = n_third_each
            n_e_third = n_third_each
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        g_main = _isotropic([0.0, 0.0], sigma, n - n_g_third)
        e_main = _isotropic([separation, 0.0], sigma, n - n_e_third)
        g = g_main
        e = e_main
        if n_g_third > 0:
            g = np.concatenate(
                [
                    g_main,
                    _isotropic([separation / 2.0, 0.0], sigma, n_g_third),
                ],
                axis=0,
            )
        if n_e_third > 0:
            e = np.concatenate(
                [
                    e_main,
                    _isotropic([separation / 2.0, 0.0], sigma, n_e_third),
                ],
                axis=0,
            )
        return _calc_from_samples(g, e)

    separations = np.linspace(0.2, 8.0, 24)
    snr_clean = np.array(
        [
            _score_third_contamination(sep, third_strength=0.0, mode="e_only")
            for sep in separations
        ]
    )
    snr_e_only = np.array(
        [
            _score_third_contamination(sep, third_strength=0.2, mode="e_only")
            for sep in separations
        ]
    )
    snr_ge_both = np.array(
        [
            _score_third_contamination(sep, third_strength=0.2, mode="ge_both")
            for sep in separations
        ]
    )

    strengths = np.linspace(0.0, 0.4, 17)
    fixed_sep = 2.0
    snr_strength_e_only = np.array(
        [_score_third_contamination(fixed_sep, s, mode="e_only") for s in strengths]
    )
    snr_strength_ge_both = np.array(
        [_score_third_contamination(fixed_sep, s, mode="ge_both") for s in strengths]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].plot(separations, snr_clean, label="clean")
    axes[0].plot(separations, snr_e_only, label="e_only third (20%)")
    axes[0].plot(separations, snr_ge_both, label="ge symmetric third (20%)")
    axes[0].set_xlabel("Center separation (sigma)")
    axes[0].set_ylabel("SNR")
    axes[0].set_title("SNR vs center separation")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(strengths, snr_strength_e_only, label="e_only third")
    axes[1].plot(strengths, snr_strength_ge_both, label="ge symmetric third")
    axes[1].set_xlabel("Third-distribution strength")
    axes[1].set_ylabel("SNR")
    axes[1].set_title(f"SNR vs contamination strength (separation={fixed_sep:.1f}σ)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.show()
