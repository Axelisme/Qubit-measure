from numbers import Number
from typing import Optional

from qick.asm_v1 import AcquireProgram


class FluxControl:
    def __init__(self, prog: AcquireProgram, flux_cfg: dict):
        raise NotImplementedError

    def set_flux(self, value: Optional[Number]) -> None:
        raise NotImplementedError

    def trigger(self) -> None:
        raise NotImplementedError


class NoneFluxControl(FluxControl):
    def __init__(self, prog: AcquireProgram, flux_cfg: dict):
        pass

    def set_flux(self, value: Optional[Number]) -> None:
        pass

    def trigger(self):
        pass
