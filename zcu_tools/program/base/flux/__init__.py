from .base import NoneFluxControl, FluxControl


def make_fluxControl(prog, flux_cfg: dict) -> FluxControl:
    dev_name = flux_cfg["name"]
    if dev_name == "yoko":
        from .yoko import YokoFluxControl

        return YokoFluxControl(prog, flux_cfg)
    elif dev_name == "zcu216":
        from .zcu216 import ZCUFluxControl

        return ZCUFluxControl(prog, flux_cfg)
    elif dev_name == "none":
        return NoneFluxControl(prog, flux_cfg)
    else:
        raise ValueError(f"Unknown flux control method: {dev_name}")
