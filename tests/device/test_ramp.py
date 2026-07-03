from __future__ import annotations

import threading
from typing import Any, Literal, cast

import pytest
from zcu_tools.device import FakeDevice, FakeDeviceInfo
from zcu_tools.device.yoko import YOKOGS200


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

    def query(self, cmd: str) -> str:
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
    dev._rampinterval = 0.0
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
    dev._rampstep = 0.25

    result = dev.set_voltage(1.0, progress=False)

    assert result == pytest.approx(1.0)
    assert session.level_writes == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_yoko_current_ramp_preserves_include_start_behavior() -> None:
    dev, session = _make_yoko(mode="current", output="on", level=0.0)
    dev._rampstep = 1e-6

    result = dev.set_current(4e-6, progress=False)

    assert result == pytest.approx(4e-6)
    assert session.level_writes == pytest.approx([0.0, 1e-6, 2e-6, 3e-6, 4e-6])


def test_yoko_output_off_nonzero_target_raises_without_level_write() -> None:
    dev, session = _make_yoko(mode="voltage", output="off", level=0.0)

    with pytest.raises(RuntimeError, match="Output is off"):
        dev.set_voltage(1.0, progress=False)

    assert session.level_writes == []


def test_yoko_voltage_safety_raises_without_level_write() -> None:
    dev, session = _make_yoko(mode="voltage", output="on", level=0.0)

    with pytest.raises(RuntimeError, match="over 20V"):
        dev.set_voltage(20.1, progress=False)

    assert session.level_writes == []
