"""mist_fluxdep analysis helpers for lab workflows."""

import numpy as np
from numpy.typing import NDArray


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Project centred MIST IQ magnitude/spread to display gain response without a fit.

    signals is one complex 1-D gain trace. Return same-shaped absolute centred
    magnitude divided by its population standard deviation plus 1e-12.
    All-NaN traces stay NaN; constant traces return zero. Input is unchanged;
    no I/O or threshold fit occurs.
    """
    if signals.ndim != 1:
        raise ValueError("MIST projection requires a 1-D gain trace")
    if np.all(np.isnan(signals)):
        return np.abs(signals)
    magnitude = np.abs(signals - np.mean(signals))
    return magnitude / (np.std(magnitude) + 1e-12)


def mist_fluxdep_signal2real(
    signals: NDArray[np.complex128],
) -> NDArray[np.float64]:
    """Project flux-by-gain IQ, subtract median background and clip at three sigma.

    Each row uses mist_signal2real. All-NaN maps remain NaN; missing rows are
    excluded from median/spread statistics. Raw IQ is never changed.
    """
    if signals.ndim != 2:
        raise ValueError("MIST map projection requires a 2-D flux/gain array")
    projected = np.array([mist_signal2real(row) for row in signals])
    if not np.all(np.isnan(projected)):
        projected = np.abs(projected - np.nanmedian(projected))
        projected = np.clip(projected, 0, 3 * np.nanstd(projected))
    return projected
