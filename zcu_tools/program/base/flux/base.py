from numbers import Number
from typing import Optional

from qick.asm_v1 import AcquireProgram


class FluxControl:
    def __init__(self, program: AcquireProgram, flux_cfg: dict):
        self.prog = program
        self.cfg = flux_cfg

    def set_flux(self, flux: Optional[Number]) -> None:
        raise NotImplementedError

    def trigger(self) -> None:
        raise NotImplementedError


class NoneFluxControl(FluxControl):
    def __init__(self):
        pass

    def set_flux(self, flux):
        pass

    def trigger(self):
        pass








