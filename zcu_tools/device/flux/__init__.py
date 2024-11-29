from .base import NoneFluxControl, FluxControl
from .zcu216 import ZCUFluxControl
from .yoko import Labber_YokoFluxControl, Qcodes_YokoFluxControl


def make_fluxControl(prog, flux_cfg) -> FluxControl:
    dev_name = flux_cfg["name"]
    if dev_name == "qcodes_yoko":
        return Qcodes_YokoFluxControl(prog, flux_cfg)
    elif dev_name == "labber_yoko":
        return Labber_YokoFluxControl(prog, flux_cfg)
    elif dev_name == "zcu216":
        return ZCUFluxControl(prog, flux_cfg)
    elif dev_name == "none":
        return NoneFluxControl()
    else:
        raise ValueError(f"Unknown flux control method: {dev_name}")

