"""Author-selected device axes on normalized-flux plots."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import zcu_tools.notebook.analysis.fit_tools as fit_tools
from zcu_tools.resources.record_table import DeviceValueUnit, RecordFluxFrame


@pytest.mark.parametrize("unit", ["A", "V", "native"])
def test_device_axis_tracks_flux_limits_in_the_declared_unit(
    unit: DeviceValueUnit,
) -> None:
    fig, ax = plt.subplots()
    ax.set_xlim(0.2, 0.6)
    frame = RecordFluxFrame(dev_unit=unit, flux_int=0.001, flux_period=0.002)
    secondary = fit_tools.add_device_axis(ax, frame, device_name="bias")
    try:
        fig.canvas.draw()
        np.testing.assert_allclose(secondary.get_xlim(), [0.0014, 0.0022])
        assert secondary.get_xlabel() == f"bias ({unit})"
        ax.set_xlim(-0.5, 0.5)
        fig.canvas.draw()
        np.testing.assert_allclose(secondary.get_xlim(), [0.0, 0.002])
    finally:
        plt.close(fig)
