"""Mean-IQ display mechanics shared by reset calibration scans."""

import numpy as np
from numpy.typing import NDArray
from zcu_tools.experiment.v2.runtime.schedule import SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support.live import live_line


def reset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Project complex mean IQ onto its response axis, without population calibration."""
    return rotate2real(signals).real


def reset_scan_buffer(
    plots: Plots, coordinates: NDArray[np.float64], *, xlabel: str
) -> SignalBuffer:
    """Retain a 1D complex IQ scan and publish its projection as measurement.

    coordinates contains the hardware-rounded scan values. xlabel names their
    quantity and unit. Unmeasured points remain NaN; the display projection has
    no calibrated ground/excited direction and must not select a reset optimum.
    """
    viewer = live_line(plots, "measurement", xlabel, "Amplitude")
    return SignalBuffer(
        (len(coordinates),),
        on_update=lambda data: viewer.update(coordinates, reset_signal2real(data)),
    )
