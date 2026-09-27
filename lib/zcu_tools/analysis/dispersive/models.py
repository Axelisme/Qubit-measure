"""Data carriers for dispersive numerical preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PreprocessResult:
    """Output of the signal-preprocessing pipeline (the notebook's cells 5-6).

    ``norm_phases`` is the normalized phase-difference image the tuning / fit work
    against. ``sp_fluxs`` / ``sp_freqs`` are the (flux, GHz-freq) axes. ``edelays``
    / ``edelay`` are the per-row and median electronic-delay diagnostics (for the
    3-panel preview). ``median_rf`` is the median over flux of each row's peak
    frequency (GHz) — the data-derived seed for the r_f tuning slider. ``signature``
    fingerprints the smoothing parameters so a stale fit can be invalidated when
    preprocessing is re-run differently.
    """

    sp_fluxs: NDArray[np.float64]
    sp_freqs: NDArray[np.float64]  # GHz
    norm_phases: NDArray[np.float64]  # (n_flux, n_freq)
    edelays: NDArray[np.float64]
    edelay: float
    median_rf: float = 0.0  # GHz — median of per-flux peak frequencies
    signature: tuple = ()
