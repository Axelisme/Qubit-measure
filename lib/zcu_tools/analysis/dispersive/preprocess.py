"""Numerical dispersive one-tone preprocessing."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.dispersive._fast_edelay import fast_edelays
from zcu_tools.analysis.dispersive.models import PreprocessResult
from zcu_tools.analysis.fitting.resonance import (
    calc_phase,
    fit_circle_params,
    remove_edelay,
)
from zcu_tools.utils.process import SmoothMethod, smooth_signal1d

# Smoothing divisors (the notebook's hard-coded factors): the per-row smooth
# strength is ``n_freq // EDELAY_SMOOTH_DIV`` before the circle fit, and
# ``n_freq // PHASE_SMOOTH_DIV`` before the phase difference. They are part of
# the preprocessing signature so a re-run with different smoothing invalidates a fit.
PREPROCESS_SMOOTH_METHOD: SmoothMethod = "wavelet"
EDELAY_SMOOTH_DIV = 30
PHASE_SMOOTH_DIV = 10


def _smooth_sigma(n_freq: int, divisor: int) -> int:
    """Smooth strength = ``n_freq // divisor``, floored at 1.

    A coarse grid with fewer than ``divisor`` frequency points would otherwise
    disable smoothing. Flooring at 1 keeps the GUI pipeline deterministic on
    small spectra.
    """
    return max(1, n_freq // divisor)


def _smooth_freq_axis[T: (np.float64, np.complex128)](
    signals: NDArray[T], divisor: int
) -> NDArray[T]:
    return smooth_signal1d(
        signals,
        method=PREPROCESS_SMOOTH_METHOD,
        sigma=float(_smooth_sigma(int(signals.shape[1]), divisor)),
        axis=1,
    )


def compute_preprocess(
    sp_fluxs: NDArray[np.float64],
    sp_freqs: NDArray[np.float64],
    signals: NDArray[np.complex128],
) -> PreprocessResult:
    """Run the preprocessing pipeline on a raw one-tone spectrum (pure, off-main-safe).

    ``signals`` is the (n_flux, n_freq) complex grid; ``sp_freqs`` is in GHz. Returns
    the ``PreprocessResult`` (norm_phases + axes + edelay diagnostics).
    """
    # The heavy per-flux edelay fit, parallelised in the numba kernel (see
    # _fast_edelay); the median over flux is the spectrum's electronic delay.
    edelays = fast_edelays(sp_freqs, signals)
    edelay = float(np.median(edelays))

    n_freq = int(signals.shape[1])
    rot_signals = remove_edelay(sp_freqs, signals, edelay)
    rot_signals = _smooth_freq_axis(rot_signals, EDELAY_SMOOTH_DIV)
    rot_signals = np.asarray(rot_signals, dtype=np.complex128)

    circle_param = np.median(
        [fit_circle_params(s.real, s.imag) for s in rot_signals], axis=0
    )
    phases = calc_phase(rot_signals, circle_param[0], circle_param[1], axis=1)

    norm_phases = _smooth_freq_axis(phases, PHASE_SMOOTH_DIV)
    norm_phases = np.diff(norm_phases, axis=1, prepend=norm_phases[:, :1])
    norm_phases = np.abs(norm_phases)
    norm_phases /= np.max(norm_phases, axis=1, keepdims=True)

    # Data-derived r_f seed: each flux row's peak (the resonance), median over flux
    # (robust to outlier rows). The slider defaults here.
    peak_freqs = sp_freqs[np.argmax(norm_phases, axis=1)]
    median_rf = float(np.median(peak_freqs))

    return PreprocessResult(
        sp_fluxs=sp_fluxs.astype(np.float64),
        sp_freqs=sp_freqs.astype(np.float64),
        norm_phases=norm_phases.astype(np.float64),
        edelays=edelays,
        edelay=edelay,
        median_rf=median_rf,
        signature=(
            PREPROCESS_SMOOTH_METHOD,
            EDELAY_SMOOTH_DIV,
            PHASE_SMOOTH_DIV,
            int(signals.shape[0]),
            n_freq,
        ),
    )
