"""Shared Lorentzian center tracking for AC Stark and CKP spectra."""

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import fitlor


def get_resonance_freq(
    xs: NDArray[np.float64], freqs: NDArray[np.float64], amps: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit traces without NaN and retain centers without large jumps.

    xs labels the rows of amps; freqs is their common MHz axis. Skip rows
    containing NaN and centers more than 10% of the signed frequency span
    from the previous accepted center. Return accepted xs and centers in MHz.
    Lorentzian fitting errors propagate; inputs are not changed.
    """
    s_xs = []
    s_freqs = []

    prev_freq = np.nan
    for x, amp in zip(xs, amps, strict=False):
        if np.any(np.isnan(amp)):
            continue

        param, _ = fitlor(freqs, amp)
        curr_freq = param[3]

        if abs(curr_freq - prev_freq) > 0.1 * (freqs[-1] - freqs[0]):
            continue

        prev_freq = curr_freq

        s_xs.append(x)
        s_freqs.append(curr_freq)

    return np.array(s_xs), np.array(s_freqs)
