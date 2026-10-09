from __future__ import annotations

import threading
from typing import Any, Literal, cast

import pytest
from pydantic import ValidationError
from zcu_tools.device import FakeDevice, FakeDeviceInfo
from zcu_tools.device.yoko import YOKOGS200, YOKOGS200Info


class DummyYokoSession:
    def __init__(
        self,
        *,
        output: Literal["0", "1"],
        mode: Literal["VOLT", "CURR"],
        level: float,
    ) -> None:
        self.resource_name = "YOKO::INSTR"
        self.read_termination = ""
        self.write_termination = ""
        self.output = output
        self.mode = mode
        self.level = level
        self.level_writes: list[float] = []
        self.queries: list[str] = []
        self.writes: list[str] = []

    def query(self, cmd: str) -> str:
        self.queries.append(cmd)
        if cmd == "*IDN?":
            return "yoko-dummy"
        if cmd == ":OUTPut?":
            return self.output
        if cmd == ":SOURce:FUNCtion?":
            return self.mode
        if cmd == ":SOURce:LEVel?":
            return f"{self.level:.12f}"
        raise ValueError(f"unsupported query: {cmd}")

    def write(self, cmd: str) -> object:
        self.writes.append(cmd)
        if cmd.startswith(":OUTPut "):
            self.output = cast(Literal["0", "1"], cmd.rsplit(" ", 1)[1])
            return None
        if cmd.startswith(":SOURce:FUNCtion "):
            self.mode = cast(Literal["VOLT", "CURR"], cmd.rsplit(" ", 1)[1])
            return None
        if cmd.startswith(":SOURce:LEVel:AUTO "):
            self.level = float(cmd.rsplit(" ", 1)[1])
            self.level_writes.append(self.level)
            return None
        raise ValueError(f"unsupported write: {cmd}")

    def close(self) -> None:
        return None


class DummyYokoResourceManager:
    def __init__(self, session: DummyYokoSession) -> None:
        self.session = session

    def open_resource(self, address: str) -> DummyYokoSession:
        return self.session


def _make_yoko(
    *,
    output: Literal["on", "off"] = "on",
    mode: Literal["voltage", "current"] = "voltage",
    level: float = 0.0,
) -> tuple[YOKOGS200, DummyYokoSession]:
    session = DummyYokoSession(
        output="1" if output == "on" else "0",
        mode="VOLT" if mode == "voltage" else "CURR",
        level=level,
    )
    rm = DummyYokoResourceManager(session)
    dev = YOKOGS200("YOKO::INSTR", cast(Any, rm))
    return dev, session


def test_fake_device_stop_event_prevents_ramp_value_change() -> None:
    dev = FakeDevice(fast_mode=True)
    stop_event = threading.Event()
    stop_event.set()
    cfg = FakeDeviceInfo(address="none", output="on", value=1.0, rampstep=0.25)

    dev.setup(cfg, progress=False, stop_event=stop_event)

    assert dev.get_output() == "on"
    assert dev.get_value() == 0.0


def test_fake_device_rejects_non_positive_rampstep() -> None:
    dev = FakeDevice(fast_mode=True)
    cfg = FakeDeviceInfo(address="none", output="on", value=1.0, rampstep=0.0)

    with pytest.raises(ValueError, match="ramp step must be positive"):
        dev.setup(cfg, progress=False)


def test_yoko_voltage_ramp_preserves_include_start_behavior() -> None:
    dev, session = _make_yoko(mode="voltage", output="on", level=0.0)
    dev.set_mode("voltage", rampstep=1e-3)

    result = dev.set_voltage(4e-3, progress=False)

    assert result == pytest.approx(4e-3)
    assert session.level_writes == pytest.approx([0.0, 1e-3, 2e-3, 3e-3, 4e-3])


def test_yoko_current_ramp_preserves_include_start_behavior() -> None:
    dev, session = _make_yoko(mode="current", output="on", level=0.0)
    dev.set_mode("current", rampstep=1e-6)

    result = dev.set_current(4e-6, progress=False)

    assert result == pytest.approx(4e-6)
    assert session.level_writes == pytest.approx([0.0, 1e-6, 2e-6, 3e-6, 4e-6])


def test_yoko_output_off_nonzero_target_raises_without_level_write() -> None:
    dev, session = _make_yoko(mode="voltage", output="off", level=0.0)

    with pytest.raises(RuntimeError, match="Output is off"):
        dev.set_voltage(1.0, progress=False)

    assert session.level_writes == []


@pytest.mark.parametrize("mode", ["voltage", "current"])
@pytest.mark.parametrize("status", ["on", "off"])
@pytest.mark.parametrize("level", [-1e-6, 1e-6])
def test_yoko_output_transition_rejects_nonzero_level_without_write(
    mode: Literal["voltage", "current"],
    status: Literal["on", "off"],
    level: float,
) -> None:
    initial_output = "off" if status == "on" else "on"
    dev, session = _make_yoko(mode=mode, output=initial_output, level=level)

    with pytest.raises(RuntimeError, match="ramp to zero first"):
        dev.set_output(status)

    assert session.writes == []
    assert session.output == ("0" if initial_output == "off" else "1")
    assert session.level == level


@pytest.mark.parametrize("mode", ["voltage", "current"])
@pytest.mark.parametrize("status", ["on", "off"])
def test_yoko_output_transition_allows_zero_level(
    mode: Literal["voltage", "current"],
    status: Literal["on", "off"],
) -> None:
    dev, session = _make_yoko(
        mode=mode, output="off" if status == "on" else "on", level=0.0
    )

    dev.set_output(status)

    expected_output = "1" if status == "on" else "0"
    assert session.writes == [f":OUTPut {expected_output}"]
    assert session.output == expected_output
    assert session.level == 0.0


@pytest.mark.parametrize("mode", ["voltage", "current"])
@pytest.mark.parametrize("status", ["on", "off"])
@pytest.mark.parametrize("level", [0.0, 1e-6])
def test_yoko_unchanged_output_skips_level_check_and_write(
    mode: Literal["voltage", "current"],
    status: Literal["on", "off"],
    level: float,
) -> None:
    dev, session = _make_yoko(mode=mode, output=status, level=level)
    session.queries.clear()

    dev.set_output(status)

    assert session.queries == [":OUTPut?"]
    assert session.writes == []
    assert session.output == ("1" if status == "on" else "0")
    assert session.level == level


def test_yoko_voltage_safety_raises_without_level_write() -> None:
    dev, session = _make_yoko(mode="voltage", output="on", level=0.0)

    with pytest.raises(RuntimeError, match="over 20V"):
        dev.set_voltage(20.1, progress=False)

    assert session.level_writes == []


@pytest.mark.parametrize("mode, limit", [("voltage", 20.0), ("current", 20e-3)])
def test_yoko_setters_reject_values_over_fixed_output_limit(
    mode: Literal["voltage", "current"],
    limit: float,
) -> None:
    dev, session = _make_yoko(mode=mode)
    cfg = YOKOGS200Info(address=dev.address, output="on", mode=mode)
    dev.setup(cfg, progress=False)
    session.level_writes.clear()
    setter = dev.set_voltage if mode == "voltage" else dev.set_current
    for value in (-1.1 * limit, 1.1 * limit):
        with pytest.raises(RuntimeError, match="in magnitude"):
            setter(value, progress=False)
    assert session.level_writes == []


@pytest.mark.parametrize("mode, target", [("voltage", 2e-3), ("current", 2e-6)])
def test_yoko_setup_uses_mode_default_rampstep(
    mode: Literal["voltage", "current"],
    target: float,
) -> None:
    dev, session = _make_yoko(mode=mode)
    dev.setup(
        YOKOGS200Info(address=dev.address, output="on", mode=mode, value=target),
        progress=False,
    )
    assert session.level_writes == pytest.approx([0.0, target / 2, target])


@pytest.mark.parametrize("mode, limit", [("voltage", 1e-2), ("current", 1e-5)])
def test_yoko_rampstep_limits_hold_across_mode_changes(
    mode: Literal["voltage", "current"],
    limit: float,
) -> None:
    dev, session = _make_yoko(mode=mode)
    cfg = YOKOGS200Info(address=dev.address, output="on", mode=mode, rampstep=limit)
    dev.setup(cfg, progress=False)
    assert dev.get_info() == cfg
    dev.set_mode("voltage" if mode == "current" else "current")
    dev.set_mode(mode, rampstep=limit)
    session.level_writes.clear()
    with pytest.raises(ValidationError, match="rampstep limit"):
        dev.set_mode(mode, rampstep=limit * 1.01)
    assert dev.get_info() == cfg
    assert session.level_writes == []


@pytest.mark.parametrize("step", [0.0, -1.0, float("inf"), float("nan")])
def test_yoko_set_mode_rejects_invalid_step_before_mode_change(step: float) -> None:
    dev, session = _make_yoko(mode="voltage")
    with pytest.raises(ValidationError, match="rampstep"):
        dev.set_mode("current", rampstep=step)
    assert session.mode == "VOLT"
    assert session.level_writes == []


@pytest.mark.parametrize("mode", ["voltage", "current"])
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_yoko_setter_rejects_nonfinite_level_without_write(
    mode: Literal["voltage", "current"],
    value: float,
) -> None:
    dev, session = _make_yoko(mode=mode)
    setter = dev.set_voltage if mode == "voltage" else dev.set_current
    with pytest.raises(RuntimeError, match="finite"):
        setter(value, progress=False)
    assert session.level_writes == []
