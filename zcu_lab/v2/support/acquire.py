"""Completed-round SNR stopping for simultaneous coherence measurements."""

from collections.abc import Callable

import numpy as np
from zcu_tools.experiment.v2.runtime.schedule import SignalBuffer
from zcu_tools.experiment.v2.utils.snr import estimate_snr
from zcu_tools.utils.process import rotate2real


def mux_snr_checker(
    signals: SignalBuffer, threshold: float | None, *, tone_axis: int
) -> Callable[[], bool]:
    """Stop only when every active tone's PCA-projected trace reaches threshold.

    signals holds a two-dimensional completed-round IQ average. tone_axis selects
    its active-tone coordinate (0 for Echo, 1 for T1); the other axis is delay.
    None disables stopping. NaN or incomplete traces never trigger. This condition
    ends only the current acquisition, without cancelling the caller's scope.
    """

    def check() -> bool:
        if threshold is None:
            return False
        traces = np.moveaxis(signals.array, tone_axis, 0)
        if not np.all(np.isfinite(traces)):
            return False
        return all(
            estimate_snr(rotate2real(trace).real) >= threshold for trace in traces
        )

    return check
