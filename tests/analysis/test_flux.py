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


def test_joint_solve_reaches_unwrapped_targets_with_crosstalk() -> None:
    frames = {
        "F1": flux.FluxFrame(
            0.25,
            {
                "bias1": flux.DevicePeriod(2, "A"),
                "bias2": flux.DevicePeriod(8, "V"),
                "coil": flux.DevicePeriod(-4, "A"),
            },
        ),
        "F2": flux.FluxFrame(
            -0.5,
            {"bias2": flux.DevicePeriod(-4, "V"), "coil": flux.DevicePeriod(8, "A")},
        ),
        "coupler": flux.FluxFrame(
            0.125,
            {
                "bias1": flux.DevicePeriod(16, "A"),
                "bias2": flux.DevicePeriod(8, "V"),
                "coil": flux.DevicePeriod(2, "A"),
            },
        ),
    }
    targets = {"coupler": 1.625, "F1": 1.25, "F2": 0.375}

    values = flux.solve_device_values(
        frames, targets, device_names=("coil", "bias1", "bias2"), fixed_values={}
    )

    assert values.keys() == {"bias1", "bias2", "coil"}
    for name, frame in frames.items():
        assert flux.flux_from_device_values(frame, values) == pytest.approx(
            targets[name], abs=1e-12, rel=0
        )


@pytest.fixture
def coupled_bias_frame() -> flux.FluxFrame:
    return flux.FluxFrame(
        0.5, {"bias": flux.DevicePeriod(2, "A"), "coil": flux.DevicePeriod(-4, "A")}
    )


def test_joint_solve_accounts_for_fixed_device(
    coupled_bias_frame: flux.FluxFrame,
) -> None:
    target = -1.25
    fixed = {"coil": 2.0}
    values = flux.solve_device_values(
        {"F1": coupled_bias_frame},
        {"F1": target},
        device_names=("bias",),
        fixed_values=fixed,
    )

    assert values.keys() == {"bias"}
    assert flux.flux_from_device_values(
        coupled_bias_frame, {**fixed, **values}
    ) == pytest.approx(target, abs=1e-12, rel=0)


@pytest.mark.parametrize("device_names", [(), ("bias", "coil")])
def test_joint_solve_requires_as_many_devices_as_targets(
    coupled_bias_frame: flux.FluxFrame, device_names: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="one solve device per target component"):
        flux.solve_device_values(
            {"F1": coupled_bias_frame},
            {"F1": 0.5},
            device_names=device_names,
            fixed_values={},
        )


@pytest.mark.parametrize("second_period", [1.0, 1.0 + 1e-13])
def test_joint_solve_rejects_singular_and_nearly_singular_frames(
    second_period: float,
) -> None:
    frames = {
        "F1": flux.FluxFrame(
            0, {"bias": flux.DevicePeriod(1), "coil": flux.DevicePeriod(1)}
        ),
        "F2": flux.FluxFrame(
            0, {"bias": flux.DevicePeriod(1), "coil": flux.DevicePeriod(second_period)}
        ),
    }
    with pytest.raises(ValueError, match="singular or nearly singular"):
        flux.solve_device_values(
            frames,
            {"F1": 0.5, "F2": 0.5},
            device_names=("bias", "coil"),
            fixed_values={},
        )


def test_joint_solve_names_a_device_missing_its_fixed_value(
    coupled_bias_frame: flux.FluxFrame,
) -> None:
    with pytest.raises(ValueError, match="F1.*coil.*fixed value"):
        flux.solve_device_values(
            {"F1": coupled_bias_frame},
            {"F1": 0.5},
            device_names=("bias",),
            fixed_values={},
        )


@pytest.mark.parametrize(
    ("device_names", "fixed_values"),
    [(("bias", "coil"), {"trim": 0.0}), (("bias", "trim"), {"coil": 0.0})],
)
def test_joint_solve_requires_matching_units_for_each_shared_device(
    coupled_bias_frame: flux.FluxFrame,
    device_names: tuple[str, ...],
    fixed_values: dict[str, float],
) -> None:
    frames = {
        "F1": coupled_bias_frame,
        "F2": flux.FluxFrame(
            0, {"coil": flux.DevicePeriod(8, "mA"), "trim": flux.DevicePeriod(-2, "V")}
        ),
    }
    with pytest.raises(ValueError, match="coil.*'mA'.*'A'"):
        flux.solve_device_values(
            frames,
            {"F1": 0.5, "F2": 0.5},
            device_names=device_names,
            fixed_values=fixed_values,
        )


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), 1j, "0.5"])
@pytest.mark.parametrize("invalid_field", ["target", "fixed"])
def test_joint_solve_rejects_nonfinite_or_nonreal_values(
    coupled_bias_frame: flux.FluxFrame, invalid_field: str, bad: float
) -> None:
    target, fixed = (bad, 0.0) if invalid_field == "target" else (0.5, bad)
    field = "target flux for 'F1'" if invalid_field == "target" else "device 'coil'"
    with pytest.raises(ValueError, match=field + ".*finite"):
        flux.solve_device_values(
            {"F1": coupled_bias_frame},
            {"F1": target},
            device_names=("bias",),
            fixed_values={"coil": fixed},
        )


def test_joint_solve_requires_matching_component_names(
    coupled_bias_frame: flux.FluxFrame,
) -> None:
    with pytest.raises(ValueError, match="missing targets:.*F1.*missing frames:.*F2"):
        flux.solve_device_values(
            {"F1": coupled_bias_frame},
            {"F2": 0.5},
            device_names=("bias",),
            fixed_values={"coil": 0.0},
        )


def test_joint_solve_rejects_duplicate_device_names(
    coupled_bias_frame: flux.FluxFrame,
) -> None:
    frames = {"F1": coupled_bias_frame, "F2": coupled_bias_frame}
    with pytest.raises(ValueError, match="duplicate.*bias"):
        flux.solve_device_values(
            frames,
            {"F1": 0.5, "F2": 0.5},
            device_names=("bias", "bias"),
            fixed_values={"coil": 0.0},
        )


def test_joint_solve_with_no_targets_returns_no_device_values() -> None:
    assert flux.solve_device_values({}, {}, device_names=(), fixed_values={}) == {}


@pytest.mark.parametrize(
    ("offset", "period", "target", "field"),
    [
        (0.0, 1e-320, 0.5, "coefficients"),
        (-1e308, 1.0, 1e308, "remaining flux"),
        (0.0, 1e308, 2.0, "device 'bias' solution"),
    ],
)
def test_joint_solve_rejects_nonfinite_computed_values(
    offset: float, period: float, target: float, field: str
) -> None:
    frames = {"F1": flux.FluxFrame(offset, {"bias": flux.DevicePeriod(period)})}
    with pytest.raises(ValueError, match=field + ".*finite"):
        flux.solve_device_values(
            frames, {"F1": target}, device_names=("bias",), fixed_values={}
        )


def test_constant_frame_and_unknown_projection() -> None:
    frame = flux.FluxFrame(1.5, {})
    assert flux.flux_from_device_values(frame, {}) == 1.5
    assert flux.offset_for_flux({}, {}, flux=1.5) == 1.5
    with pytest.raises(ValueError, match="unknown.*bias"):
        flux.device_value_at_flux(frame, {}, device_name="bias", flux=0.5)
    with pytest.raises(ValueError, match="unit"):
        flux.DevicePeriod(4, " ")
