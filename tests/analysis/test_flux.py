"""Public linear flux coordinates in native device units."""

import pytest
from zcu_tools.analysis.flux import (
    DeviceFluxFrame,
    DeviceFluxPoint,
    FluxFrame,
    actual_flux_points,
    flux_from_device_values,
    reference_flux_points,
)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"bias": 1.0, "coil": 2.0}, 0.0),
        ({"bias": 3.0, "coil": 3.0}, 0.25),
        ({"bias": 1.0, "coil": 6.0}, -1.0),
    ],
)
def test_flux_sums_all_device_contributions_with_signed_periods(
    values: dict[str, float], expected: float
) -> None:
    frame = FluxFrame(
        {"bias": DeviceFluxFrame(1, 4, "V"), "coil": DeviceFluxFrame(2, -4, "A")}
    )
    assert flux_from_device_values(frame, values) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), 1j, "1"])
def test_frame_rejects_invalid_integer(bad: float) -> None:
    with pytest.raises(ValueError, match="integer"):
        DeviceFluxFrame(bad, 4)


@pytest.mark.parametrize("bad", [True, 0, float("nan"), float("inf"), 1j, "1"])
def test_frame_rejects_invalid_signed_period(bad: float) -> None:
    with pytest.raises(ValueError, match="period"):
        DeviceFluxFrame(1, bad)


@pytest.mark.parametrize("values", [{}, {"bias": True}, {"bias": float("inf")}])
def test_flux_rejects_missing_or_invalid_contributing_device(
    values: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="bias"):
        flux_from_device_values(FluxFrame({"bias": DeviceFluxFrame(1, 4)}), values)


def test_frame_snapshots_nonempty_named_devices() -> None:
    devices = {"bias": DeviceFluxFrame(1, 4)}
    frame = FluxFrame(devices)
    devices.clear()
    assert flux_from_device_values(frame, {"bias": 3}) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="nonempty"):
        FluxFrame({})
    with pytest.raises(ValueError, match="name"):
        FluxFrame({"": DeviceFluxFrame(1, 4)})
    with pytest.raises(ValueError, match="unit"):
        DeviceFluxFrame(1, 4, "")


@pytest.mark.parametrize(
    ("period", "coil_value", "expected"),
    [(4.0, 2.0, (2.0, 0.0)), (4.0, 3.0, (1.0, -1.0)), (-4.0, 3.0, (3.0, 1.0))],
)
def test_fitted_points_return_to_integer_reference(
    period: float, coil_value: float, expected: tuple[float, float]
) -> None:
    frame = FluxFrame(
        {"bias": DeviceFluxFrame(1, period), "coil": DeviceFluxFrame(2, -4)}
    )
    result = reference_flux_points(
        DeviceFluxPoint(integer=2, half=0),
        device_name="bias",
        frame=frame,
        values={"coil": coil_value},
    )
    assert (result.integer, result.half) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("period", "coil_value", "expected"),
    [(4.0, 2.0, (1.0, -1.0)), (4.0, 3.0, (2.0, 0.0)), (-4.0, 3.0, (0.0, -2.0))],
)
def test_actual_points_account_for_current_other_device_values(
    period: float, coil_value: float, expected: tuple[float, float]
) -> None:
    frame = FluxFrame(
        {"bias": DeviceFluxFrame(1, period), "coil": DeviceFluxFrame(2, -4)}
    )
    result = actual_flux_points(
        DeviceFluxPoint(integer=1, half=-1),
        device_name="bias",
        frame=frame,
        values={"coil": coil_value},
    )
    assert (result.integer, result.half) == pytest.approx(expected)


@pytest.mark.parametrize("convert", [actual_flux_points, reference_flux_points])
def test_point_conversion_rejects_inconsistent_half_integer(convert) -> None:
    frame = FluxFrame({"bias": DeviceFluxFrame(1, -4)})
    with pytest.raises(ValueError, match="bias.*half|half.*bias"):
        convert(DeviceFluxPoint(1, 0), device_name="bias", frame=frame, values={})


@pytest.mark.parametrize("convert", [actual_flux_points, reference_flux_points])
def test_point_conversion_single_device_and_missing_other_values(convert) -> None:
    reference = DeviceFluxPoint(1, -1)
    frame = FluxFrame({"bias": DeviceFluxFrame(1, 4)})
    assert convert(reference, device_name="bias", frame=frame, values={}) == reference
    with pytest.raises(ValueError, match="unknown"):
        convert(reference, device_name="unknown", frame=frame, values={})
    frame = FluxFrame({"bias": DeviceFluxFrame(1, 4), "coil": DeviceFluxFrame(2, -4)})
    with pytest.raises(ValueError, match="coil"):
        convert(reference, device_name="bias", frame=frame, values={})


@pytest.mark.parametrize("bad", [True, float("inf"), float("nan"), 1j, "1"])
def test_points_reject_invalid_native_values(bad: float) -> None:
    with pytest.raises(ValueError, match="half"):
        DeviceFluxPoint(integer=1, half=bad)
    with pytest.raises(ValueError, match="integer"):
        DeviceFluxPoint(integer=bad, half=1)
