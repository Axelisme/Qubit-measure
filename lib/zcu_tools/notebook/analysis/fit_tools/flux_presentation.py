"""Optional device-value axes for author-composed normalized-flux plots."""

from collections.abc import Mapping

import numpy as np
from matplotlib.axes import Axes
from matplotlib.axes._secondary_axes import SecondaryAxis

from zcu_tools.analysis.flux import (
    DeviceFluxFrame,
    DeviceFluxPoint,
    FluxFrame,
    actual_flux_points,
)
from zcu_tools.cfg_model import validate_device_name
from zcu_tools.resources.record_table import RecordFluxFrame


def add_device_axis(
    ax: Axes,
    frame: RecordFluxFrame | FluxFrame,
    *,
    device_name: str,
    values: Mapping[str, float] | None = None,
) -> SecondaryAxis:
    """Add a top device-value axis to a normalized-Phi0 x axis.

    frame is a pure-analysis FluxFrame, or an explicit single-device
    RecordFluxFrame. device_name must be nonblank without ASCII dots; valid names
    are preserved and select/label the device as "name (unit)".
    values supplies every OTHER frame device's current finite real value in
    its declared unit; the selected value is unused. Omit values for a
    single-device frame. No device selection or unit conversion is inferred.

    Return Matplotlib's SecondaryAxis. It tracks parent limits in both
    directions using the selected signed period and integer shifted for the
    other devices' current flux contribution. Parent data/limits are unchanged;
    showing, drawing and saving remain the caller's responsibility.

    ValueError names an invalid/unknown device or missing/invalid used value.
    Invalid frame fields fail at construction. Matplotlib errors propagate.
    """
    validate_device_name(device_name)
    if isinstance(frame, RecordFluxFrame):
        frame = FluxFrame(
            {
                device_name: DeviceFluxFrame(
                    frame.flux_int, frame.flux_period, frame.dev_unit
                )
            }
        )
    if device_name not in frame.devices:
        raise ValueError(f"unknown frame device {device_name!r}")
    device = frame.devices[device_name]
    points = actual_flux_points(
        DeviceFluxPoint(device.integer, device.integer + device.period / 2),
        device_name=device_name,
        frame=frame,
        values={} if values is None else values,
    )
    secondary = ax.secondary_xaxis(
        "top",
        functions=(
            lambda flux: (
                np.asarray(flux, dtype=np.float64) * device.period + points.integer
            ),
            lambda value: (
                (np.asarray(value, dtype=np.float64) - points.integer) / device.period
            ),
        ),
    )
    secondary.set_xlabel(f"{device_name} ({device.unit})")
    return secondary
