from zcu_tools.device import GlobalDeviceManager


def set_flux(flux_dev: str, current, progress: bool = False) -> None:
    """Set flux/current to the specified flux device.

    This function controls the flux bias for superconducting qubit experiments
    by directing commands to the appropriate flux control device.

    Parameters
    ----------
    flux_dev : str
        The type of flux device to use. Supported options are:
        - 'yoko': Yokogawa current source
        - 'none': No flux device (no-op)
    flux : float
        The flux/current value to set in appropriate units for the device.
        For 'yoko', this is typically in amperes.

    Raises
    ------
    ValueError
        If the specified flux device is not supported.
    """
    if flux_dev == "yoko":
        from zcu_tools.device.yoko import YOKOGS200

        yoko_device = GlobalDeviceManager.get_device("flux_yoko")
        assert isinstance(yoko_device, YOKOGS200)

        mode = yoko_device.get_mode()
        if mode == "current":
            yoko_device.set_current(current, progress=progress)
        elif mode == "voltage":
            yoko_device.set_voltage(current, progress=progress)
        else:
            raise ValueError(f"Unsupported mode {mode} for flux device {flux_dev}.")
    elif flux_dev == "none":
        pass
    else:
        raise ValueError(f"Unknown flux device: {flux_dev}")
