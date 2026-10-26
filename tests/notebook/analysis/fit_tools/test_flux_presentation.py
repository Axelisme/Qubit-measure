"""Author-selected device axes on normalized-flux plots."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import zcu_tools.notebook.analysis.fit_tools as fit_tools
from zcu_tools.analysis.flux import DeviceFluxFrame, FluxFrame
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


@pytest.mark.parametrize(
    ("period", "expected_values"),
    [(4.0, [2.8, 4.4]), (-4.0, [-0.8, -2.4])],
)
def test_multiple_device_axis_maps_both_directions_at_current_other_values(
    period: float, expected_values: list[float]
) -> None:
    fig, ax = plt.subplots()
    ax.set_xlim(0.2, 0.6)
    frame = FluxFrame(
        {"bias": DeviceFluxFrame(1, period, "V"), "coil": DeviceFluxFrame(2, -4, "A")}
    )
    try:
        secondary = fit_tools.add_device_axis(
            ax, frame, device_name="bias", values={"coil": 3}
        )
        fig.canvas.draw()
        np.testing.assert_allclose(secondary.get_xlim(), sorted(expected_values))
        assert secondary.get_xlabel() == "bias (V)"
        transform = secondary.xaxis.get_transform()
        np.testing.assert_allclose(transform.transform(expected_values), [0.2, 0.6])
        np.testing.assert_allclose(
            transform.inverted().transform([0.2, 0.6]), expected_values
        )
        ax.set_xlim(0, 1)
        fig.canvas.draw()
        expected_limits = [2, 6] if period > 0 else [-4, 0]
        np.testing.assert_allclose(secondary.get_xlim(), expected_limits)
        np.testing.assert_allclose(ax.get_xlim(), [0, 1])
    finally:
        plt.close(fig)


def test_multiple_device_axis_requires_other_current_device_values() -> None:
    fig, ax = plt.subplots()
    frame = FluxFrame({"bias": DeviceFluxFrame(1, 4), "coil": DeviceFluxFrame(2, -4)})
    try:
        with pytest.raises(ValueError, match="coil"):
            fit_tools.add_device_axis(ax, frame, device_name="bias")
    finally:
        plt.close(fig)
