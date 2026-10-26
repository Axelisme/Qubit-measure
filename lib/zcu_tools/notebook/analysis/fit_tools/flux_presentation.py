"""Optional device-value axes for author-composed normalized-flux plots."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.axes._secondary_axes import SecondaryAxis

from zcu_tools.resources.record_table import RecordFluxFrame


def add_device_axis(
    ax: Axes,
    frame: RecordFluxFrame,
    *,
    device_name: str,
) -> SecondaryAxis:
    """Add a top device-value axis to an existing normalized-flux x axis.

    ax must use normalized Phi0 on x. frame is the author's selected single
    device frame: flux_int is its zero-flux value, flux_period is the positive
    value span per Phi0, and dev_unit is A, V or native. Values stay in that
    declared unit; no milli-unit scaling or device selection is inferred.
    device_name labels the secondary axis as "name (unit)".

    Return Matplotlib's SecondaryAxis for further author styling. It tracks
    parent limits using frame.dev_value_from_flux and its inverse. The parent
    data/limits are unchanged; showing, drawing and saving remain the caller's
    responsibility. Invalid frame values fail at RecordFluxFrame construction;
    Matplotlib axis/transform errors propagate.
    """
    secondary = ax.secondary_xaxis(
        "top",
        functions=(
            lambda flux: frame.dev_value_from_flux(np.asarray(flux, dtype=np.float64)),
            lambda value: frame.flux_from_dev_value(
                np.asarray(value, dtype=np.float64)
            ),
        ),
    )
    secondary.set_xlabel(f"{device_name} ({frame.dev_unit})")
    return secondary
