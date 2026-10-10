from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from zcu_tools.device import (
    YOKOGS200,
    AnritsuMG3692,
    BaseDevice,
    DeviceInfo,
    FakeDevice,
    RohdeSchwarzSGS100A,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
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


# ==================== Helpers for device config ==================== #


def get_labeled_device_cfg(
    devs_cfg: Mapping[str, DeviceInfo], label: str
) -> DeviceInfo:
    """Get the device configuration with the given label."""
    match_list = []
    for name, dev_cfg in devs_cfg.items():
        if dev_cfg.label == label:
            match_list.append(name)
    if len(match_list) == 0:
        raise ValueError(f"Device with label '{label}' not found in dev configuration.")
    if len(match_list) > 1:
        raise ValueError(
            f"Multiple devices with label '{label}' found in dev configuration: {match_list}"
        )

    return devs_cfg[match_list[0]]


def set_flux_in_dev_cfg(
    devs_cfg: Mapping[str, DeviceInfo], value: float, label: str = "flux_dev"
) -> None:
    """Set the flux value in the device configuration with the given label."""
    get_labeled_device_cfg(devs_cfg, label).set_flux(value)


def set_freq_in_dev_cfg(
    devs_cfg: Mapping[str, DeviceInfo], freq_Hz: float, label: str = "rf_dev"
) -> None:
    """Set the rf frequency value in the device configuration with the given label."""
    get_labeled_device_cfg(devs_cfg, label).set_freq(freq_Hz)


def set_power_in_dev_cfg(
    devs_cfg: Mapping[str, DeviceInfo], power_dBm: float, label: str = "rf_dev"
) -> None:
    """Set the rf power value in the device configuration with the given label."""
    get_labeled_device_cfg(devs_cfg, label).set_power(power_dBm)


def set_output_in_dev_cfg(
    devs_cfg: Mapping[str, DeviceInfo],
    output: Literal["on", "off"],
    label: str = "rf_dev",
) -> None:
    """Set the output state in the device configuration with the given label."""
    get_labeled_device_cfg(devs_cfg, label).set_output(output)


def setup_devices(
    cfg: ExpCfgModel,
    devices: Mapping[str, BaseDevice[Any]],
    *,
    progress: bool = False,
    cancel_signal: StopSignal | None = None,
) -> None:
    """Apply device setup when the experiment config contains a dev section."""

    if cfg.dev is None:
        return

    # Resolve the entire batch before touching any driver.
    missing = cfg.dev.keys() - devices.keys()
    if missing:
        raise ValueError(f"Devices not found: {', '.join(sorted(missing))}")
    snapshot = [(devices[name], info) for name, info in cfg.dev.items()]
    for device, info in snapshot:
        if cancel_signal is not None and cancel_signal.is_set():
            return
        device.setup(
            info,
            progress=progress,
            stop_event=cancel_signal.event if cancel_signal is not None else None,
        )
