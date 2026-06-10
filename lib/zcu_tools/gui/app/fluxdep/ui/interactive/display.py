"""Shared display helpers for the interactive spectrum widgets.

Qt-free numpy utilities reused by the interactive picker/preview widgets so the
spectrum imshow looks consistent across them (one definition of "robust contrast").
"""

from __future__ import annotations

import numpy as np


def contrast_limits(amp: np.ndarray) -> tuple[float, float]:
    """Robust display limits (2nd–98th percentile) to boost feature contrast.

    Clipping the colour range to percentiles (instead of min/max) keeps a few
    outliers from compressing the dynamic range, so the spectral feature stands
    out against the background. NaNs are ignored; a degenerate range falls back
    to ``(min, max)``.
    """
    finite = amp[np.isfinite(amp)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [2.0, 98.0])
    if hi <= lo:
        return float(finite.min()), float(finite.max()) or 1.0
    return float(lo), float(hi)
