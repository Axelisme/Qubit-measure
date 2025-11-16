from typing import Dict, Any, Literal


from zcu_tools.device import DeviceInfo

# ==================== Helpers for device config ==================== #


def get_labeled_device_cfg(
    devs_cfg: Dict[str, DeviceInfo], label: str
) -> Dict[str, Any]:
    """Get the device configuration with the given label."""
    for dev_cfg in devs_cfg.values():
        if dev_cfg.get("label") == label:
            return dev_cfg

    raise ValueError(f"Device with label '{label}' not found in dev configuration.")


def set_flux_in_dev_cfg(
    devs_cfg: Dict[str, DeviceInfo], value: float, label: str = "flux_dev"
) -> None:
    """Set the flux value in the device configuration with the given label."""

    flux_cfg = get_labeled_device_cfg(devs_cfg, label)

    if flux_cfg["type"] == "YOKOGS200":
        flux_cfg["value"] = value
    else:
        raise NotImplementedError(
            f"Flux device type {flux_cfg['type']} not supported yet"
        )


def set_freq_in_dev_cfg(
    devs_cfg: Dict[str, DeviceInfo], freq_Hz: float, label: str = "rf_dev"
) -> None:
    """Set the rf frequency value in the device configuration with the given label."""

    rf_cfg = get_labeled_device_cfg(devs_cfg, label)

    if rf_cfg["type"] == "RohdeSchwarzSGS100A":
        rf_cfg["freq_Hz"] = freq_Hz
    else:
        raise NotImplementedError(f"RF device type {rf_cfg['type']} not supported yet")


def set_power_in_dev_cfg(
    devs_cfg: Dict[str, DeviceInfo], power_dBm: float, label: str = "rf_dev"
) -> None:
    """Set the rf power value in the device configuration with the given label."""

    rf_cfg = get_labeled_device_cfg(devs_cfg, label)

    if rf_cfg["type"] == "RohdeSchwarzSGS100A":
        rf_cfg["power_dBm"] = power_dBm
    else:
        raise NotImplementedError(f"RF device type {rf_cfg['type']} not supported yet")


def set_output_in_dev_cfg(
    devs_cfg: Dict[str, DeviceInfo], output: Literal["on", "off"], label: str = "rf_dev"
) -> None:
    """Set the rf power value in the device configuration with the given label."""

    rf_cfg = get_labeled_device_cfg(devs_cfg, label)

    if rf_cfg["type"] == "RohdeSchwarzSGS100A":
        rf_cfg["output"] = output
    else:
        raise NotImplementedError(f"RF device type {rf_cfg['type']} not supported yet")
