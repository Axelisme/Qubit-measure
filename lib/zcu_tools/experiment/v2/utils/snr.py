from __future__ import annotations

from functools import wraps

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from typing_extensions import Any, Callable, Optional, Sequence, TypeVar

from zcu_tools.experiment.v2.runner import default_raw2signal_fn
from zcu_tools.program.v2 import ModularProgramV2
from zcu_tools.utils.func_tools import min_interval


def estimate_snr(real_signals: NDArray[np.float64]) -> float:
    smooth_signals = gaussian_filter(real_signals, sigma=1)
    noise = np.mean(np.abs(real_signals - smooth_signals))
    return (np.ptp(smooth_signals) / noise).item()


def snr_as_signal(
    raw: tuple[
        Sequence[NDArray[np.float64]],
        Sequence[NDArray[np.float64]],
        Sequence[NDArray[np.float64]],
    ],
    ge_axis: int = 0,
) -> NDArray[np.float64]:
    avg_d, cov_d, med_d = raw

    avg_d = avg_d[0][0]
    cov_d = cov_d[0][0]
    med_d = med_d[0][0]
    assert avg_d.shape[ge_axis] == 2
    assert cov_d.shape[ge_axis] == 2
    assert med_d.shape[ge_axis] == 2

    from scipy.special import erf

    # (ge, *sweep)
    peak_contrast = np.abs(
        (np.take(med_d, 1, axis=ge_axis) - np.take(med_d, 0, axis=ge_axis)).dot([1, 1j])
    ).real

    noise = np.mean(np.sqrt(np.diagonal(cov_d, axis1=-2, axis2=-1)), axis=ge_axis)
    noise_max = np.clip(np.max(noise, axis=-1), 1e-12, None)
    noise_min = np.clip(np.min(noise, axis=-1), 1e-12, None)

    return erf(peak_contrast / (np.sqrt(32) * noise_min)) * (noise_min / noise_max)


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
