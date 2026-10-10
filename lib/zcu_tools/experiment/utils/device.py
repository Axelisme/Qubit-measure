from __future__ import annotations

from typing import Any, Literal

from zcu_tools.device import (
    YOKOGS200,
    AnritsuMG3692,
    BaseDevice,
    FakeDevice,
    RohdeSchwarzSGS100A,
)
from zcu_tools.experiment.stop_signal import StopSignal


def set_flux(
    device: BaseDevice[Any],
    value: float,
    *,
    unit: Literal["A", "V", "native"],
    stop: StopSignal,
    progress: bool = False,
) -> None:
    """Set a selected flux source without changing mode, rampstep or output.

    YOKO accepts A in current mode or V in voltage mode, ramps with the Run's
    stop event and retains driver safety limits. Fake accepts only native and
    writes its single value. Unit/mode mismatch raises ValueError before writes;
    unsupported drivers raise TypeError. Driver failures propagate unchanged.
    """
    if isinstance(device, YOKOGS200):
        mode = {"A": "current", "V": "voltage"}.get(unit)
        if mode is None or device.get_mode() != mode:
            raise ValueError(f"Flux unit {unit!r} does not match YOKO mode")
        if unit == "A":
            device.set_current(value, progress=progress, stop_event=stop.event)
        else:
            device.set_voltage(value, progress=progress, stop_event=stop.event)
    elif isinstance(device, FakeDevice):
        if unit != "native":
            raise ValueError("Fake flux requires unit 'native'")
        if not stop.is_set():
            device.set_value(value)
    else:
        raise TypeError(f"Unsupported flux device: {type(device).__name__}")


def set_rf_frequency(device: BaseDevice[Any], freq_hz: float) -> None:
    """Set RF frequency in Hz on SGS100A, or the single Fake value.

    Unsupported drivers raise TypeError; driver range and I/O errors propagate.
    This operation does not enable output or change any other setting.
    """
    if isinstance(device, RohdeSchwarzSGS100A):
        device.set_frequency(freq_hz)
    elif isinstance(device, FakeDevice):
        device.set_value(freq_hz)
    else:
        raise TypeError(f"Unsupported RF frequency device: {type(device).__name__}")


def set_rf_power(device: BaseDevice[Any], power_dbm: float) -> None:
    """Set RF power in dBm on SGS100A, or the single Fake value.

    Unsupported drivers raise TypeError; driver range and I/O errors propagate.
    This operation does not enable output or change any other setting.
    """
    if isinstance(device, RohdeSchwarzSGS100A):
        device.set_power(power_dbm)
    elif isinstance(device, FakeDevice):
        device.set_value(power_dbm)
    else:
        raise TypeError(f"Unsupported RF power device: {type(device).__name__}")


def set_output(device: BaseDevice[Any], output: Literal["on", "off"]) -> None:
    """Set on/off on an existing supported driver; never ramp or switch mode.

    Invalid output raises ValueError before writes, unsupported driver raises
    TypeError and driver errors (including YOKO's zero-level guard) propagate.
    """
    if output not in ("on", "off"):
        raise ValueError("Output must be 'on' or 'off'")
    if isinstance(device, (YOKOGS200, RohdeSchwarzSGS100A, FakeDevice, AnritsuMG3692)):
        device.set_output(output)
    else:
        raise TypeError(f"Unsupported output device: {type(device).__name__}")
