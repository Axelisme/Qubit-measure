from zcu_tools.device import YokoDevControl


def set_flux(flux_dev: str, flux):
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
        YokoDevControl.set_current(flux)
    elif flux_dev == "none":
        pass
    else:
        raise ValueError(f"Unknown flux device: {flux_dev}")
