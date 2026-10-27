"""Optional device-value axes for author-composed normalized-flux plots."""

from collections.abc import Mapping

import numpy as np
from matplotlib.axes import Axes
from matplotlib.axes._secondary_axes import SecondaryAxis

from zcu_tools.analysis.flux import FluxFrame, device_value_at_flux


def add_device_axis(
    ax: Axes,
    frame: FluxFrame,
    *,
    device_name: str,
    values: Mapping[str, float] | None = None,
) -> SecondaryAxis:
    """Add a top native-device-value axis to a normalized-Phi0 x axis.

    frame declares the offset and signed periods. device_name selects and labels
    a frame device as "name (unit)"; it must be nonblank without ASCII dots.
    values fixes other devices at finite native values; absent devices contribute
    zero. The selected device's current value is unused. No units are converted.
    Return Matplotlib's SecondaryAxis, tracking parent limits in both directions
    without changing parent data or limits. Caller owns drawing and saving.
    Invalid/unknown device names or invalid used values raise ValueError.
    Invalid frame fields fail at construction. Matplotlib errors propagate.
    """
    integer = device_value_at_flux(
        frame, {} if values is None else values, device_name=device_name, flux=0.0
    )
    device = frame.devices[device_name]
    secondary = ax.secondary_xaxis(
        "top",
        functions=(
            lambda flux: np.asarray(flux, dtype=np.float64) * device.period + integer,
            lambda value: (
                (np.asarray(value, dtype=np.float64) - integer) / device.period
            ),
        ),
    )
    secondary.set_xlabel(f"{device_name} ({device.unit})")
    return secondary
