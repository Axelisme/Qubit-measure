from .base import FluxControl
from typing import Optional
from numbers import Number


class ZCUFluxControl(FluxControl):
    @classmethod
    def register(cls, flux_dev: dict, force=False):
        cls.cfg = flux_dev
        cls.ch = flux_dev["ch"]
        cls.saturate = flux_dev["saturate"]

    def __init__(self, prog):
        self.prog = prog

    def set_flux(self, value: Optional[Number]) -> None:
        if value is None:
            value = 0  # default to zero

        # cast numpy int to python int
        if hasattr(value, "item"):
            value = value.item()

        if not isinstance(value, int):
            raise ValueError(f"Flux must be an int in ZCUFluxControl, but got {value}")
        assert (
            -30000 <= value <= 30000
        ), f"Flux must be in the range [-30000, 30000], but got {value}"

        cls = type(self)
        self.prog.declare_gen(cls.ch, nqz=1)
        self.prog.default_pulse_registers(
            cls.ch, style="const", freq=0, phase=0, stdysel="last", length=3, gain=value
        )

    def trigger(self):
        cls = type(self)
        self.prog.pulse(cls.ch)
        self.prog.synci(self.prog.us2cycles(cls.saturate))

