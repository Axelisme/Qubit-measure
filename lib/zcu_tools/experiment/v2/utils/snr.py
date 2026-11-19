from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

from zcu_tools.experiment.acquisition_facts import StopDecision

DEFAULT_SKEW_PENALTY = 0.0


def _project_ge_stats(
    mean_d: NDArray[np.float64],
    cov_d: NDArray[np.float64],
    ge_axis: int,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    mean_g = np.take(mean_d, 0, axis=ge_axis)
    mean_e = np.take(mean_d, 1, axis=ge_axis)
    cov_g = np.take(cov_d, 0, axis=ge_axis)
    cov_e = np.take(cov_d, 1, axis=ge_axis)

    axis_vec = mean_e - mean_g
    delta_mu = np.linalg.norm(axis_vec, axis=-1)
    axis = axis_vec / np.clip(delta_mu, 1e-24, None)[..., None]

    var_g = np.clip(np.einsum("...i,...ij,...j->...", axis, cov_g, axis), 1e-24, None)
    var_e = np.clip(np.einsum("...i,...ij,...j->...", axis, cov_e, axis), 1e-24, None)
    sigma_g = np.sqrt(var_g)
    sigma_e = np.sqrt(var_e)

    return axis, delta_mu, sigma_g, sigma_e


def estimate_snr(real_signals: NDArray[np.float64]) -> float:
    # Savitzky-Golay requires an odd window length and window_length > polyorder.
    n = real_signals.shape[-1]
    window_length = min(7, n if n % 2 else n - 1)
    if window_length >= 3:
        # scipy 1.17 stubs type savgol_filter as a union of array types;
        # collapse it back to the concrete float64 ndarray it returns here.
        smooth_signals = np.asarray(
            savgol_filter(
                real_signals, window_length=window_length, polyorder=2, axis=-1
            ),
            dtype=np.float64,
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
    skew_penalty: float = DEFAULT_SKEW_PENALTY,
) -> NDArray[np.float64]:
    """Compute shape-aware g/e separation SNR.

    Inputs:
        mean_d: shape (..., 2, 2)        — per-state mean on ge_axis
        cov_d:  shape (..., 2, 2, 2)     — per-state 2x2 covariance
        m3_d:   shape (..., 2, 2, 2, 2)  — per-state 3rd central moment tensor

    Returns:
        snr = Δμ / sqrt(0.5 * (σ_g² + σ_e²))
        shape_cost = ln²(σ_g / σ_e) + (skew_g + skew_e)²
        score = snr / (1 + skew_penalty * shape_cost)

    ``skew_penalty=0`` gives pure, unbounded separation SNR. Larger values
    gradually down-weight one-sided skew and g/e shape mismatch without the
    cliff-like exponential penalty of the previous bounded score.
    """
    if skew_penalty < 0.0:
        raise ValueError("skew_penalty must be non-negative")

    m3_g = np.take(m3_d, 0, axis=ge_axis)
    m3_e = np.take(m3_d, 1, axis=ge_axis)

    axis, delta_mu, sigma_g, sigma_e = _project_ge_stats(mean_d, cov_d, ge_axis)
    pooled_sigma = np.sqrt(0.5 * (sigma_g**2 + sigma_e**2))
    snr = delta_mu / np.clip(pooled_sigma, 1e-24, None)
    if skew_penalty == 0.0:
        return snr

    proj_m3_g = np.einsum("...i,...j,...k,...ijk->...", axis, axis, axis, m3_g)
    proj_m3_e = np.einsum("...i,...j,...k,...ijk->...", axis, axis, axis, m3_e)
    skew_g = proj_m3_g / (sigma_g**3)
    skew_e = proj_m3_e / (sigma_e**3)

    shape_cost = np.log(sigma_g / sigma_e) ** 2 + (skew_g + skew_e) ** 2
    return snr / (1.0 + skew_penalty * shape_cost)


def _central_moments(
    sums: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    count = sums[..., 0]
    raw = sums / np.where(count > 0, count, np.nan)[..., None]
    mean = raw[..., 1:3]
    second = raw[..., [[3, 4], [4, 5]]]
    third = raw[..., [[[6, 7], [7, 8]], [[7, 8], [8, 9]]]]

    # Use sample covariance (zero for a single shot) and population third moment.
    cov = second - mean[..., :, None] * mean[..., None, :]
    cov *= (count / np.maximum(count - 1, 1))[..., None, None]
    mi = mean[..., :, None, None]
    mj = mean[..., None, :, None]
    mk = mean[..., None, None, :]
    centered_third = (
        third
        - mi * second[..., None, :, :]
        - mj * second[..., :, None, :]
        - mk * second[..., :, :, None]
        + 2 * mi * mj * mk
    )
    return mean, cov, centered_third


def snr_as_signal(
    sums: NDArray[np.float64],
    ge_axis: int = 0,
    *,
    skew_penalty: float = DEFAULT_SKEW_PENALTY,
) -> NDArray[np.float64]:
    """Convert a single ``IQ(read, order=3)`` shot-sum axis to g/e SNR.

    The last axis of sums contains ten raw sums, starting with the count.
    ge_axis selects the length-2 g/e axis among the preceding axes; negative
    indices are relative to sums. Remaining sweep axes keep their order in
    the float64 result. With no sweep axes, return a zero-dimensional array.
    Use sample covariance and population third central moments for the
    pooled-sigma SNR and shape penalty described in calc_snr. A zero or
    missing count in either state yields NaN for that sweep point.

    Raise ValueError for an invalid sums shape, state axis or negative penalty.
    """
    if sums.ndim < 2 or sums.shape[-1] != 10:
        raise ValueError(
            f"Expected cubic IQ sums with final axis length 10, got {sums.shape}"
        )
    if ge_axis < 0:
        ge_axis += sums.ndim
    if not 0 <= ge_axis < sums.ndim - 1:
        raise ValueError("ge_axis must select a state axis, not the final IQ sums axis")
    if sums.shape[ge_axis] != 2:
        raise ValueError(
            f"Expected state axis length 2, got shape {sums.shape} with ge_axis={ge_axis}"
        )
    mean, cov, third = _central_moments(sums)
    return np.asarray(
        calc_snr(mean, cov, third, ge_axis=ge_axis, skew_penalty=skew_penalty)
    )


def snr_checker(
    ctx: Any,
    snr_threshold: float | None,
    signal2real_fn: Callable[[np.ndarray], np.ndarray],
    after_check: Callable[[float], Any] | None = None,
    check_interval: float | None = 0.1,
) -> Callable[[], StopDecision]:
    """Check a slot's projected completed-round curve against an SNR threshold.

    ctx.value supplies cumulative data; signal2real_fn projects it to a real curve.
    None disables stopping. Checks are throttled by check_interval seconds (None
    checks every call). All-NaN raw or projected curves skip estimation and do
    not call after_check. Otherwise after_check receives each estimate. NaN
    estimates do not stop acquisition. Positive infinity meets the threshold, as for a varying
    curve with zero estimated noise. A met decision keeps the first crossing SNR
    for subsequent calls; nonfinite crossing values are recorded as None.
    """
    if snr_threshold is None:
        return lambda: StopDecision(False)

    import time

    _decision = StopDecision(False)
    _last_check = 0.0

    def check() -> StopDecision:
        nonlocal _decision, _last_check
        if _decision:
            return _decision
        now = time.time()
        if check_interval is not None and now - _last_check < check_interval:
            return StopDecision(False)
        _last_check = now
        signals = np.asarray(ctx.value)
        if np.all(np.isnan(signals)):
            return StopDecision(False)
        real_signals = signal2real_fn(signals)
        if np.all(np.isnan(real_signals)):
            return StopDecision(False)
        snr = estimate_snr(real_signals)
        if after_check is not None:
            after_check(snr)
        if snr >= snr_threshold:
            _decision = StopDecision(True, snr if np.isfinite(snr) else None)
        return _decision

    return check


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
