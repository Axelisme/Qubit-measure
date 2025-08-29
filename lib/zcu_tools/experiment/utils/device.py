from typing import Dict, Any


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
    devs_cfg: Dict[str, DeviceInfo], value: float, label: str = "rf_dev"
) -> None:
    """Set the rf frequency value in the device configuration with the given label."""

    rf_cfg = get_labeled_device_cfg(devs_cfg, label)

    if rf_cfg["type"] == "RFSource":
        rf_cfg["freq"] = value
    else:
        raise NotImplementedError(f"RF device type {rf_cfg['type']} not supported yet")
