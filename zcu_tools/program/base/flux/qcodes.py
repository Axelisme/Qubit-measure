from .base import FluxControl
from typing import Optional
from numbers import Number

class Qcodes_YokoFluxControl(FluxControl):
    def __init__(self, program, cfg):
        super().__init__(program, cfg)

        self.name = cfg["name"]
        self.address = cfg["address"]
        self.limit = cfg["limit"]
        self.rate = cfg["rate"]

        self.yoko = None

    def _init_dev(self):
        try:
            from qcodes.instrument_drivers.yokogawa import YokogawaGS200  # type: ignore
        except ImportError:
            raise ImportError(
                "Please install qcodes to use YokoFluxControl in the program"
            )

        self.yoko = YokogawaGS200(self.name, address=self.address, terminator="n")
        self.source_mode("CURR")
        self.yoko.current_limit(self.limit)

    def set_flux(self, flux: Optional[Number]) -> None:
        if flux is None:
            flux = 0.0  # default to zero

        # cast numpy float to python float
        if hasattr(flux, "item"):
            flux = flux.item()

        # if not np.issubdtype(flux, np.floating):
        if not isinstance(flux, float):
            raise ValueError(f"Flux must be a float in YokoFluxControl, but got {flux}")
        assert (
            self.limit[0] <= flux < self.limit[1]
        ), f"Flux must be in the range {self.limit}, but got {flux}"

        if self.yoko is None:
            self._init_dev()

        self.yoko.ramp_current(flux, self.rate, 0.01)

    def trigger(self):
        pass