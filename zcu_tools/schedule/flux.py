from zcu_tools.device import YokoDevControl


def set_flux(flux_dev: str, flux):
    if flux_dev == "yoko":
        YokoDevControl.set_current(flux)
    elif flux_dev == "none":
        pass
    else:
        raise ValueError(f"Unknown flux device: {flux_dev}")
