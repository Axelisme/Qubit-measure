from numbers import Number
from typing import Optional

from .base import FluxControl
from zcu_tools.device.labber import YokoDevControl


class YokoFluxControl(FluxControl):
    def __init__(self, prog, flux_cfg: dict):
        YokoDevControl.connect_server(flux_cfg)

    def set_flux(self, value: Optional[Number]) -> None:
        if value is None:
            return  # default do nothing

        YokoDevControl.set_current(value)

    def trigger(self):
        pass
