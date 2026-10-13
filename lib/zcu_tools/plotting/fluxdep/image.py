"""Spectrum background normalization shared by fluxdep previews and pickers.

Qt-free numpy helpers. The GUI result preview, the OneTone sample picker and
the native OneTone pick figure call them so a spectrum's background looks the
same in each view under ``gray_r`` (high values dark).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.fluxdep.processing import cast2real_and_norm


def onetone_dip_depth(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Return how far each OneTone pixel lies below the spectrum's peak magnitude.

    signals is the complex spectrum shaped (device, frequency). The magnitude
    is smoothed along frequency (``cast2real_and_norm`` without phase), then
    subtracted from its largest finite value, so a resonator dip becomes the
    largest value and draws dark on a light background under ``gray_r``. NaN
    pixels stay NaN. The result has the shape of signals.
    """
    magnitude = cast2real_and_norm(signals, use_phase=False)
    return np.nanmax(magnitude) - magnitude


def contrast_limits(values: NDArray[np.float64]) -> tuple[float, float]:
    """Robust display limits (2nd–98th percentile) to boost feature contrast.

    Clipping the colour range to percentiles (instead of min/max) keeps a few
    outliers from compressing the dynamic range, so the spectral feature stands
    out against the background. NaNs are ignored; no finite value gives
    ``(0, 1)`` and a degenerate range falls back to ``(min, max)``.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [2.0, 98.0])
    if hi <= lo:
        return float(finite.min()), float(finite.max()) or 1.0
    return float(lo), float(hi)
