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


class YokoFluxControl(FluxControl):
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


def make_fluxControl(prog, flux_cfg) -> FluxControl:
    dev_name = flux_cfg["name"]
    if dev_name == "yokogawa":
        return YokoFluxControl(prog, flux_cfg)
    elif dev_name == "zcu216":
        return ZCUFluxControl(prog, flux_cfg)
    elif dev_name == "none":
        return NoneFluxControl()
    else:
        raise ValueError(f"Unknown flux control method: {dev_name}")
