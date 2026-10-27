"""Author-selected device axes on normalized-flux plots."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from zcu_tools.analysis.flux import DevicePeriod, FluxFrame
from zcu_tools.notebook.analysis.fit_tools import add_device_axis


@pytest.mark.parametrize("period", [4.0, -4.0])
def test_device_axis_projects_signed_period_with_other_devices_fixed(
    period: float,
) -> None:
    fig, ax = plt.subplots()
    frame = FluxFrame(
        0.5, {"bias": DevicePeriod(period, "V"), "coil": DevicePeriod(-4, "A")}
    )
    try:
        secondary = add_device_axis(ax, frame, device_name="bias", values={"coil": 3})
        ax.set_xlim(0.2, 0.6)
        fig.canvas.draw()
        expected = [0.45 * period, 0.85 * period]
        np.testing.assert_allclose(secondary.get_xlim(), sorted(expected))
        transform = secondary.xaxis.get_transform()
        np.testing.assert_allclose(transform.transform(expected), [0.2, 0.6])
        np.testing.assert_allclose(transform.inverted().transform([0.2, 0.6]), expected)
        assert secondary.get_xlabel() == "bias (V)"
        ax.set_xlim(0, 1)
        fig.canvas.draw()
        np.testing.assert_allclose(
            secondary.get_xlim(), sorted([0.25 * period, 1.25 * period])
        )
        np.testing.assert_allclose(ax.get_xlim(), [0, 1])
    finally:
        plt.close(fig)
