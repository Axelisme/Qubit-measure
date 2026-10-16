from __future__ import annotations

from typing import Any, Literal

from zcu_tools.device import (
    YOKOGS200,
    AnritsuMG3692,
    BaseDevice,
    BaseDeviceInfo,
    FakeDevice,
    FakeDeviceInfo,
    RohdeSchwarzSGS100A,
    YOKOGS200Info,
)
from zcu_tools.experiment.stop_signal import StopSignal


def check_flux_unit(cfg: BaseDeviceInfo, unit: str) -> None:
    """Check a flux coordinate against stored Info.value_unit; return None.

    YOKOGS200Info validates current/A and voltage/V; FakeDeviceInfo stores its fixed unit.
    unit is required and must equal that stored unit. Mismatch or any other unit
    raises ValueError. Other Info models raise TypeError naming the unsupported
    device class. No conversion, I/O, cache, cancellation or cfg mutation occurs.
    """
    if not isinstance(cfg, (YOKOGS200Info, FakeDeviceInfo)):
        raise TypeError(f"{type(cfg).__name__} does not support flux settings")
    if unit != cfg.value_unit:
        raise ValueError(
            f"Flux unit {unit!r} does not match {type(cfg).__name__} "
            f"value unit {cfg.value_unit!r}"
        )


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
    stop event and retains driver safety limits. Fake accepts its fixed unit and
    writes its single value without conversion. Unit/mode mismatch raises ValueError before writes;
    unsupported drivers raise TypeError. Driver failures propagate unchanged.
    """
    if isinstance(device, YOKOGS200):
        if unit not in ("A", "V"):
            # Synthetic carrier rejects impossible units before any mode query;
            # it is not hardware readback.
            check_flux_unit(YOKOGS200Info(address=device.address, mode="current"), unit)
        mode = device.get_mode()
        check_flux_unit(YOKOGS200Info(address=device.address, mode=mode), unit)
        if unit == "A":
            device.set_current(value, progress=progress, stop_event=stop.event)
        else:
            device.set_voltage(value, progress=progress, stop_event=stop.event)
    elif isinstance(device, FakeDevice):
        check_flux_unit(device.get_info(), unit)
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
