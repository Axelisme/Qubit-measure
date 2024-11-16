import numpy as np

import qick as qk


class FluxControl:
    def __init__(self, program: qk.AveragerProgram, flux_cfg: dict):
        self.prog = program
        self.cfg = flux_cfg

    def set_flux(self, flux: int | float) -> None:
        raise NotImplementedError

    def trigger(self) -> None:
        raise NotImplementedError


class YokoFluxControl(FluxControl):
    def __init__(self, program, flux_cfg):
        super().__init__(program, flux_cfg)

        self.name = self.cfg["name"]
        self.address = self.cfg["address"]
        self.limit = 10e-3
        self.rate = 5e-6

        self.yoko = None

    def _init_dev(self):
        try:
            from qcodes.instrument_drivers.yokogawa import YokogawaGS200
        except ImportError:
            raise ImportError(
                "Please install qcodes to use YokoFluxControl in the program"
            )

        self.yoko = YokogawaGS200(self.name, address=self.address, terminator="n")
        self.source_mode("CURR")
        self.yoko.current_limit(self.limit)

    def set_flux(self, flux):
        assert np.issubdtype(
            type(flux), np.floating
        ), f"Flux must be a float in YokoFluxControl, but got {flux}"
        assert (
            abs(flux) <= self.limit
        ), f"Flux must be in the range [-0.01, 0.01], but got {flux}"

        if self.yoko is None:
            self._init_dev()

        self.yoko.ramp_current(flux, self.rate, 0.01)

    def trigger(self):
        pass


class ZCUFluxControl(FluxControl):
    def __init__(self, program, flux_cfg):
        super().__init__(program, flux_cfg)

        self.ch = self.cfg["ch"]
        self.saturate = 0.1
        self.first_set = True

    def set_flux(self, flux):
        assert np.issubdtype(
            type(flux), np.integer
        ), f"Flux must be an int in ZCUFluxControl, but got {flux}"
        assert (
            abs(flux) <= 40000
        ), f"Flux must be in the range [-40000, 40000], but got {flux}"

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


def make_fluxControl(prog, method, flux_cfg) -> FluxControl:
    match method:
        case "yokogawa":
            return YokoFluxControl(prog, flux_cfg)
        case "zcu216":
            return ZCUFluxControl(prog, flux_cfg)
        case _:
            raise ValueError(f"Unknown flux control method: {method}")
