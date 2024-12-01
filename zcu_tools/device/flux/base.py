from numbers import Number
from typing import Optional

from qick.asm_v1 import AcquireProgram


class FluxControl:
    @classmethod
    def register(cls, flux_dev: dict, force=False):
        raise NotImplementedError

    def __init__(self, prog: AcquireProgram):
        raise NotImplementedError

    def set_flux(self, value: Optional[Number]) -> None:
        raise NotImplementedError

    def trigger(self) -> None:
        raise NotImplementedError


class NoneFluxControl(FluxControl):
    @classmethod
    def register(cls, flux_dev: dict, force=False):
        pass

    def __init__(self, prog: AcquireProgram):
        pass

    def set_flux(self, value: Optional[Number]) -> None:
        pass

    def trigger(self):
        pass
