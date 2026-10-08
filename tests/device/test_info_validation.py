from __future__ import annotations

from typing import Any, Literal, cast

import pytest
from pydantic import ValidationError
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.device.yoko import YOKOGS200Info


def test_fake_device_info_rejects_bool_value() -> None:
    with pytest.raises(ValidationError, match="real numeric scalar"):
        FakeDeviceInfo(address="none", value=cast(Any, True))


def test_fake_device_info_rejects_string_value() -> None:
    with pytest.raises(ValidationError, match="real numeric scalar"):
        FakeDeviceInfo(address="none", value=cast(Any, "1.23"))


def test_yoko_info_rejects_bool_value() -> None:
    with pytest.raises(ValidationError, match="real numeric scalar"):
        YOKOGS200Info(address="GPIB::1", value=cast(Any, True))


def test_yoko_info_rejects_string_value() -> None:
    with pytest.raises(ValidationError, match="real numeric scalar"):
        YOKOGS200Info(address="GPIB::1", value=cast(Any, "1.23"))


def test_fake_device_info_accepts_int_and_float_values() -> None:
    assert FakeDeviceInfo(address="none", value=1).value == pytest.approx(1.0)
    assert FakeDeviceInfo(address="none", value=1.23).value == pytest.approx(1.23)


def test_yoko_info_accepts_int_and_float_values() -> None:
    assert YOKOGS200Info(address="GPIB::1", value=1).value == pytest.approx(1.0)
    assert YOKOGS200Info(address="GPIB::1", value=1.23).value == pytest.approx(1.23)


@pytest.mark.parametrize("mode, expected", [("current", 1e-6), ("voltage", 1e-3)])
def test_yoko_missing_rampstep_uses_mode_default(
    mode: Literal["current", "voltage"], expected: float
) -> None:
    info = YOKOGS200Info(address="GPIB::1", mode=mode)
    saved = YOKOGS200Info.model_validate_json(
        f'{{"address": "GPIB::1", "mode": "{mode}"}}'
    )
    assert info.rampstep == pytest.approx(expected)
    assert saved.rampstep == pytest.approx(expected)
    assert YOKOGS200Info.model_validate_json(info.to_json()) == info


@pytest.mark.parametrize("mode", ["current", "voltage"])
def test_yoko_explicit_rampstep_survives_updates(
    mode: Literal["current", "voltage"],
) -> None:
    info = YOKOGS200Info(address="GPIB::1", mode=mode, rampstep=2e-7)
    updated = info.with_updates(value=1e-4)
    assert updated.rampstep == pytest.approx(2e-7)
    assert info.with_updates(rampstep=3e-7).rampstep == pytest.approx(3e-7)


@pytest.mark.parametrize("step", [0.0, -1e-6, float("inf"), float("nan")])
def test_yoko_rejects_invalid_rampstep_at_model_boundary(step: float) -> None:
    with pytest.raises(ValidationError, match="rampstep"):
        YOKOGS200Info(address="GPIB::1", rampstep=step)
    info = YOKOGS200Info(address="GPIB::1", mode="current")
    with pytest.raises(ValidationError, match="rampstep"):
        info.with_updates(rampstep=step)


@pytest.mark.parametrize("mode, limit", [("voltage", 0.5), ("current", 2e-3)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_yoko_enforces_configured_output_limits(
    mode: Literal["current", "voltage"], limit: float, sign: int
) -> None:
    info = YOKOGS200Info(
        address="GPIB::1",
        mode=mode,
        max_voltage=0.5,
        max_current=2e-3,
        value=sign * limit,
    )
    with pytest.raises(ValidationError, match="output limit"):
        info.with_updates(value=sign * limit * 1.01)
    assert YOKOGS200Info.model_validate_json(info.to_json()) == info


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("inf"), float("nan")])
def test_yoko_rejects_invalid_output_limits(invalid: float) -> None:
    with pytest.raises(ValidationError, match="max_voltage"):
        YOKOGS200Info(address="GPIB::1", max_voltage=invalid)
    with pytest.raises(ValidationError, match="max_current"):
        YOKOGS200Info(address="GPIB::1", max_current=invalid)


@pytest.mark.parametrize("mode", ["voltage", "current"])
def test_yoko_rejects_values_outside_default_output_limit(
    mode: Literal["current", "voltage"],
) -> None:
    with pytest.raises(ValidationError, match="output limit"):
        YOKOGS200Info(address="GPIB::1", mode=mode, value=21.0)


@pytest.mark.parametrize("mode, limit", [("voltage", 1e-2), ("current", 1e-5)])
def test_yoko_default_rampstep_limit_boundary(
    mode: Literal["voltage", "current"],
    limit: float,
) -> None:
    info = YOKOGS200Info(address="GPIB::1", mode=mode, rampstep=limit)
    assert info.rampstep == pytest.approx(limit)
    assert YOKOGS200Info.model_validate_json(info.to_json()) == info
    with pytest.raises(ValidationError, match="rampstep limit"):
        YOKOGS200Info(address="GPIB::1", mode=mode, rampstep=limit * 1.01)
    with pytest.raises(ValidationError, match="rampstep limit"):
        info.with_updates(rampstep=limit * 1.01)


@pytest.mark.parametrize("mode, limit", [("voltage", 2e-2), ("current", 2e-5)])
def test_yoko_custom_rampstep_limit_boundary(
    mode: Literal["voltage", "current"],
    limit: float,
) -> None:
    info = YOKOGS200Info(
        address="GPIB::1",
        mode=mode,
        rampstep=limit,
        max_voltage_rampstep=2e-2,
        max_current_rampstep=2e-5,
    )
    assert YOKOGS200Info.model_validate_json(info.to_json()) == info
    with pytest.raises(ValidationError, match="rampstep limit"):
        info.with_updates(rampstep=limit * 1.01)
    field = "max_voltage_rampstep" if mode == "voltage" else "max_current_rampstep"
    with pytest.raises(ValidationError, match="rampstep limit"):
        info.with_updates(**{field: limit / 2})


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("inf"), float("nan")])
def test_yoko_rejects_invalid_rampstep_limits(invalid: float) -> None:
    with pytest.raises(ValidationError, match="max_voltage_rampstep"):
        YOKOGS200Info(address="GPIB::1", max_voltage_rampstep=invalid)
    with pytest.raises(ValidationError, match="max_current_rampstep"):
        YOKOGS200Info(address="GPIB::1", max_current_rampstep=invalid)


def test_yoko_set_flux_rejects_out_of_range_without_changing_level() -> None:
    info = YOKOGS200Info(address="GPIB::1", mode="current", max_current=1e-4)
    info.set_flux(5e-5)
    with pytest.raises(ValidationError, match="output limit"):
        info.set_flux(2e-4)
    assert info.value == pytest.approx(5e-5)


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_yoko_rejects_nonfinite_output_values(value: float) -> None:
    with pytest.raises(ValidationError, match="value"):
        YOKOGS200Info(address="GPIB::1", value=value)
