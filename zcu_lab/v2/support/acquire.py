"""Completed-round SNR stopping for simultaneous coherence measurements."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from zcu_tools.experiment.acquisition_facts import StopDecision
from zcu_tools.experiment.v2.runtime.schedule import SignalBuffer
from zcu_tools.experiment.v2.utils.snr import estimate_snr
from zcu_tools.utils.process import rotate2real


def mux_snr_checker(
    signals: SignalBuffer, threshold: float | None, *, tone_axis: int
) -> Callable[[], StopDecision]:
    """Stop only when every active tone's PCA-projected trace reaches threshold.

    signals holds a two-dimensional completed-round IQ average. tone_axis selects
    its active-tone coordinate (0 for Echo, 1 for T1); the other axis is delay.
    None disables stopping. NaN or incomplete traces never trigger. This condition
    ends only the current acquisition, without cancelling the caller's scope.
    The decision records the minimum active-tone SNR; nonfinite values use None.
    """

    return mux_mean_snr_checker(lambda: signals.array, threshold, tone_axis=tone_axis)


def mux_mean_snr_checker(
    mean_iq: Callable[[], NDArray[np.complex128]],
    threshold: float | None,
    *,
    tone_axis: int,
) -> Callable[[], StopDecision]:
    """Check current committed round means without needing a separate IQ buffer.

    mean_iq returns a 2D complex IQ array with tones along tone_axis. Call the
    condition after committing the round. A missing threshold or incomplete IQ
    never stops. A successful decision reports the minimum active-tone SNR
    when finite. Checking does not request Stop on any scope.
    """

    def check() -> StopDecision:
        if threshold is None:
            return StopDecision(False)
        traces = np.moveaxis(mean_iq(), tone_axis, 0)
        if not np.all(np.isfinite(traces)):
            return StopDecision(False)
        snrs = [estimate_snr(rotate2real(trace).real) for trace in traces]
        met = all(snr >= threshold for snr in snrs)
        minimum = min(snrs)
        return StopDecision(met, minimum if met and np.isfinite(minimum) else None)

    return check
