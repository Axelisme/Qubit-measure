"""Public linear flux coordinates in native device units."""

import pytest
from zcu_tools.analysis import flux


def test_flux_adds_offset_and_signed_device_contributions() -> None:
    frame = flux.FluxFrame(
        offset=0.5,
        devices={
            "bias": flux.DevicePeriod(0.002, "A"),
            "coil": flux.DevicePeriod(-0.01, "A"),
        },
    )
    assert flux.flux_from_device_values(
        frame, {"bias": 0.001, "coil": 0.002}
    ) == pytest.approx(0.8)


@pytest.mark.parametrize("period", [0.002, -0.002])
def test_projection_inverts_forward_flux_with_other_devices_fixed(
    period: float,
) -> None:
    frame = flux.FluxFrame(
        0.5, {"bias": flux.DevicePeriod(period), "coil": flux.DevicePeriod(0.01)}
    )
    value = flux.device_value_at_flux(
        frame, {"coil": 0.002, "bias": float("nan")}, device_name="bias", flux=0.5
    )
    assert value == pytest.approx(-0.2 * period)
    assert flux.flux_from_device_values(
        frame, {"coil": 0.002, "bias": value}
    ) == pytest.approx(0.5)


def test_inverse_recovers_unwrapped_offset_and_ignores_missing_devices() -> None:
    devices = {
        "bias": flux.DevicePeriod(4),
        "coil": flux.DevicePeriod(-2),
        "absent": flux.DevicePeriod(1),
    }
    values = {"bias": -8.0, "coil": 0.5, "unused": float("nan")}
    offset = flux.offset_for_flux(devices, values, flux=0.5)
    assert offset == pytest.approx(2.75)
    assert flux.flux_from_device_values(
        flux.FluxFrame(offset, devices), values
    ) == pytest.approx(0.5)


@pytest.mark.parametrize("values", [{}, {"unused": float("nan")}])
def test_absent_devices_contribute_zero(values: dict[str, float]) -> None:
    frame = flux.FluxFrame(1.25, {"bias": flux.DevicePeriod(4)})
    assert flux.flux_from_device_values(frame, values) == 1.25
    assert flux.device_value_at_flux(frame, values, device_name="bias", flux=0.5) == -3
    assert flux.offset_for_flux(frame.devices, values, flux=0.5) == 0.5


@pytest.mark.parametrize(
    "bad", [True, float("nan"), float("inf"), -float("inf"), 1j, "1"]
)
def test_used_invalid_value_names_device_at_all_math_seams(bad: float) -> None:
    frame = flux.FluxFrame(
        0.5, {"bias": flux.DevicePeriod(4), "coil": flux.DevicePeriod(-2)}
    )
    with pytest.raises(ValueError, match="coil"):
        flux.flux_from_device_values(frame, {"coil": bad})
    with pytest.raises(ValueError, match="coil"):
        flux.device_value_at_flux(frame, {"coil": bad}, device_name="bias", flux=0.5)
    with pytest.raises(ValueError, match="coil"):
        flux.offset_for_flux(frame.devices, {"coil": bad}, flux=0.5)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), 1j, "1"])
def test_offsets_and_targets_reject_invalid_numbers(bad: float) -> None:
    with pytest.raises(ValueError, match="offset"):
        flux.FluxFrame(bad, {})
    frame = flux.FluxFrame(0, {"bias": flux.DevicePeriod(1)})
    with pytest.raises(ValueError, match="flux"):
        flux.device_value_at_flux(frame, {}, device_name="bias", flux=bad)
    with pytest.raises(ValueError, match="flux"):
        flux.offset_for_flux(frame.devices, {}, flux=bad)


@pytest.mark.parametrize("bad", [0, True, float("nan"), float("inf"), 1j, "1"])
def test_period_rejects_zero_and_invalid_numbers(bad: float) -> None:
    with pytest.raises(ValueError, match="period"):
        flux.DevicePeriod(bad)


@pytest.mark.parametrize("name", ["bias.local", "", " \t "])
def test_math_rejects_invalid_device_names(name: str) -> None:
    with pytest.raises(ValueError, match="(?i)device name"):
        flux.FluxFrame(0, {name: flux.DevicePeriod(4)})
    frame = flux.FluxFrame(0, {"bias": flux.DevicePeriod(4)})
    with pytest.raises(ValueError, match="(?i)device name"):
        flux.device_value_at_flux(frame, {}, device_name=name, flux=0.5)


@pytest.mark.parametrize("name", [" Bias ", "bias", "BIAS", "偏壓", "bias．local"])
def test_math_preserves_valid_names_and_snapshots_periods(name: str) -> None:
    devices = {name: flux.DevicePeriod(4, "V")}
    frame = flux.FluxFrame(-0.25, devices)
    devices.clear()
    assert flux.flux_from_device_values(frame, {name: 3}) == 0.5
    assert flux.device_value_at_flux(frame, {}, device_name=name, flux=0.5) == 3


@pytest.mark.parametrize(
    ("actual", "expected", "matches"),
    [
        ("A", "A", True),
        (None, None, True),
        (None, "native", True),
        ("native", None, True),
        ("mA", "A", False),
        (None, "A", False),
        ("A", None, False),
        ("native", "V", False),
        ("a", "A", False),
        (" A ", "A", False),
    ],
)
def test_device_units_match_exactly_without_conversion(
    actual: str | None, expected: str | None, matches: bool
) -> None:
    if matches:
        assert flux.require_matching_device_unit("coil", actual, expected) is None
    else:
        with pytest.raises(ValueError, match="coil") as error:
            flux.require_matching_device_unit("coil", actual, expected)
        message = str(error.value)
        assert repr(actual if actual is not None else "native") in message
        assert repr(expected if expected is not None else "native") in message


def test_constant_frame_and_unknown_projection() -> None:
    frame = flux.FluxFrame(1.5, {})
    assert flux.flux_from_device_values(frame, {}) == 1.5
    assert flux.offset_for_flux({}, {}, flux=1.5) == 1.5
    with pytest.raises(ValueError, match="unknown.*bias"):
        flux.device_value_at_flux(frame, {}, device_name="bias", flux=0.5)
    with pytest.raises(ValueError, match="unit"):
        flux.DevicePeriod(4, " ")
