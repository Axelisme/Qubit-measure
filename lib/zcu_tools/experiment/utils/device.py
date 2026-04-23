from __future__ import annotations

from typing_extensions import TYPE_CHECKING, Any, Literal, Mapping, Optional, cast

from zcu_tools.device import DeviceInfo, GlobalDeviceManager
from zcu_tools.experiment.cfg_model import ExpCfgModel

if TYPE_CHECKING:
    from zcu_tools.device.sgs100a import RohdeSchwarzSGS100AInfo
    from zcu_tools.device.yoko import YOKOGS200Info

# ==================== Helpers for device config ==================== #


def get_labeled_device_cfg(
    devs_cfg: Mapping[str, DeviceInfo], label: str
) -> DeviceInfo:
    """Get the device configuration with the given label."""
    match_list = []
    for name, dev_cfg in devs_cfg.items():
        if dev_cfg.get("label") == label:
            match_list.append(name)
    if len(match_list) == 0:
        raise ValueError(f"Device with label '{label}' not found in dev configuration.")
    elif len(match_list) > 1:
        raise ValueError(
            f"Multiple devices with label '{label}' found in dev configuration: {match_list}"
        )

    return devs_cfg[match_list[0]]


def set_flux_in_dev_cfg(
    devs_cfg: Mapping[str, DeviceInfo], value: float, label: str = "flux_dev"
) -> None:
    """Set the flux value in the device configuration with the given label."""

    flux_cfg = get_labeled_device_cfg(devs_cfg, label)

    if flux_cfg["type"] == "YOKOGS200":
        flux_cfg = cast("YOKOGS200Info", flux_cfg)
        flux_cfg["value"] = value
    else:
        raise NotImplementedError(
            f"Flux device type {flux_cfg['type']} not supported yet"
        )


def set_freq_in_dev_cfg(
    devs_cfg: Mapping[str, DeviceInfo], freq_Hz: float, label: str = "rf_dev"
) -> None:
    """Set the rf frequency value in the device configuration with the given label."""

    rf_cfg = get_labeled_device_cfg(devs_cfg, label)

    if rf_cfg["type"] == "RohdeSchwarzSGS100A":
        rf_cfg = cast("RohdeSchwarzSGS100AInfo", rf_cfg)
        rf_cfg["freq_Hz"] = freq_Hz
    else:
        raise NotImplementedError(f"RF device type {rf_cfg['type']} not supported yet")


def set_power_in_dev_cfg(
    devs_cfg: Mapping[str, DeviceInfo], power_dBm: float, label: str = "rf_dev"
) -> None:
    """Set the rf power value in the device configuration with the given label."""

    rf_cfg = get_labeled_device_cfg(devs_cfg, label)

    if rf_cfg["type"] == "RohdeSchwarzSGS100A":
        rf_cfg = cast("RohdeSchwarzSGS100AInfo", rf_cfg)
        rf_cfg["power_dBm"] = power_dBm
    else:
        raise NotImplementedError(f"RF device type {rf_cfg['type']} not supported yet")


def set_output_in_dev_cfg(
    devs_cfg: Mapping[str, DeviceInfo],
    output: Literal["on", "off"],
    label: str = "rf_dev",
) -> None:
    """Set the rf power value in the device configuration with the given label."""

    rf_cfg = get_labeled_device_cfg(devs_cfg, label)

    if rf_cfg["type"] == "RohdeSchwarzSGS100A":
        rf_cfg = cast("RohdeSchwarzSGS100AInfo", rf_cfg)
        rf_cfg["output"] = output
    else:
        raise NotImplementedError(f"RF device type {rf_cfg['type']} not supported yet")


def setup_devices(cfg: ExpCfgModel, *, progress: bool = False) -> None:
    """Apply device setup when the experiment config contains a dev section."""

    if cfg.dev is None:
        return

    GlobalDeviceManager.setup_devices(cfg.dev, progress=progress)
