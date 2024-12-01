from .base import NoneFluxControl
from .zcu216 import ZCUFluxControl
from .yoko import Labber_YokoFluxControl


def get_fluxControl(flux_dev: dict) -> type:
    dev_name = flux_dev["name"]
    if dev_name == "labber_yoko":
        dev_cls = Labber_YokoFluxControl
    elif dev_name == "zcu216":
        dev_cls = ZCUFluxControl
    elif dev_name == "none":
        dev_cls = NoneFluxControl
    else:
        raise ValueError(f"Unknown flux control method: {dev_name}")

    dev_cls.register(flux_dev)

    return dev_cls
