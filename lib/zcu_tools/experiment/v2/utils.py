from functools import wraps
from typing import Any, Callable, Optional, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from zcu_tools.program.v2 import ModularProgramV2, PulseCfg
from zcu_tools.utils.func_tools import min_interval

from .runner import T_ResultType, default_raw2signal_fn


def calc_snr(real_signals: NDArray[np.float64]) -> float:
    smooth_signals = gaussian_filter(real_signals, sigma=1)
    noise = np.mean(np.abs(real_signals - smooth_signals))
    return float((np.max(smooth_signals) - np.min(smooth_signals)) / noise)


def set_pulse_freq(pulse_cfg: PulseCfg, freq: float) -> PulseCfg:
    pulse_cfg["freq"] = freq
    if "mixer_freq" in pulse_cfg:
        pulse_cfg["mixer_freq"] = freq
    return pulse_cfg


T_RawResult = TypeVar("T_RawResult")


def wrap_earlystop_check(
    prog: ModularProgramV2,
    update_hook: Callable[[int, T_RawResult], Any],
    snr_threshold: Optional[float],
    signal2real_fn: Callable[[np.ndarray], np.ndarray],
    raw2signal_fn: Callable[[T_RawResult], np.ndarray] = default_raw2signal_fn,
    snr_hook: Optional[Callable[[float], Any]] = None,
    update_interval: Optional[float] = 0.1,
) -> Callable[[int, T_RawResult], Any]:
    if snr_threshold is None:
        return update_hook

    def check_snr(raw: T_RawResult) -> None:
        signals = raw2signal_fn(raw)
        snr = calc_snr(signal2real_fn(signals))
        if snr >= snr_threshold:
            prog.set_early_stop(silent=True)

        if snr_hook is not None:
            snr_hook(snr)

    check_snr = min_interval(check_snr, update_interval)

    @wraps(update_hook)
    def wrapped_update_hook(i: int, raw: T_RawResult) -> None:
        update_hook(i, raw)
        check_snr(raw)

    return wrapped_update_hook


def merge_result_list(results: Sequence[T_ResultType]) -> T_ResultType:
    assert isinstance(results, list) and len(results) > 0
    if isinstance(results[0], dict):
        return {
            name: merge_result_list([r[name] for r in results])  # type: ignore
            for name in results[0]
        }
    return np.asarray(results)  # type: ignore
