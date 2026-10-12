"""mist_fluxdep analysis helpers for lab workflows."""

import numpy as np
from numpy.typing import NDArray


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Project MIST IQ with the old centred-magnitude/spread policy, no fit.

    signals is one complex 1-D gain trace. Return same-shaped absolute centred
    magnitude divided by its population standard deviation. All-NaN traces
    stay NaN; zero spread retains native division diagnostics rather than a
    threshold, epsilon or invented success. Input is unchanged; no I/O occurs.
    """
    if signals.ndim != 1:
        raise ValueError("MIST projection requires a 1-D gain trace")
    if np.all(np.isnan(signals)):
        return np.abs(signals)
    magnitude = np.abs(signals - np.mean(signals))
    return magnitude / np.std(magnitude)
