from .base import FluxControl
from typing import Optional
from numbers import Number


class ZCUFluxControl(FluxControl):
    def __init__(self, prog, flux_cfg: dict):
        self.prog = prog
        self.ch = flux_cfg["ch"]
        self.saturate = prog.us2cycles(flux_cfg["saturate"])

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

        self.prog.declare_gen(self.ch, nqz=1)
        self.prog.default_pulse_registers(
            self.ch,
            style="const",
            freq=0,
            phase=0,
            stdysel="last",
            length=3,
            gain=value,
        )

    def trigger(self):
        self.prog.pulse(self.ch)
        self.prog.synci(self.saturate)
