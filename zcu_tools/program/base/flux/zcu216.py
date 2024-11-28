from .base import FluxControl
from typing import Optional
from numbers import Number

class ZCUFluxControl(FluxControl):
    def __init__(self, program, flux_cfg):
        super().__init__(program, flux_cfg)

        self.ch = self.cfg["ch"]
        self.saturate = self.cfg["saturate"]
        self.first_set = True

    def set_flux(self, flux: Optional[Number]) -> None:
        if flux is None:
            flux = 0  # default to zero

        # cast numpy int to python int
        if hasattr(flux, "item"):
            flux = flux.item()

        if not isinstance(flux, int):
            raise ValueError(f"Flux must be an int in ZCUFluxControl, but got {flux}")
        assert (
            -30000 <= flux <= 30000
        ), f"Flux must be in the range [-30000, 30000], but got {flux}"

        self.prog.declare_gen(ch=self.ch, nqz=1)
        if self.first_set:
            self.first_set = False
            self.prog.default_pulse_registers(
                ch=self.ch, style="const", freq=0, phase=0, stdysel="last", length=3
            )
        self.prog.set_pulse_registers(ch=self.ch, gain=flux)

    def trigger(self):
        self.prog.pulse(ch=self.ch)
        self.prog.synci(self.prog.us2cycles(self.saturate))