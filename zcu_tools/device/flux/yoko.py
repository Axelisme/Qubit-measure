from numbers import Number
from typing import Optional

from .base import FluxControl


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
            # flux = 0.0  # default to zero
            return  # default do nothing

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


class Labber_YokoFluxControl(FluxControl):
    def __init__(self, program, cfg):
        super().__init__(program, cfg)

        self.sHardware = cfg["sHardware"]
        self.dev_cfg = cfg["dev_cfg"]
        self.flux_cfg = cfg["flux_cfg"]
        self.sweep_rate = self.flux_cfg["Current - Sweep rate"]
        self.server_ip = cfg["server_ip"]

        self.dev_cfg["name"] = "globalFlux"
        self.flux_cfg.update(
            {
                "Output": True,
                "Function": "Current",
                "Range (I)": "10 mA",
            }
        )

        self.yoko = None

    def _init_dev(self):
        from .labber import InstrManager

        self.yoko = InstrManager(server_ip=self.server_ip)
        self.yoko.add_instrument(
            sHardware=self.sHardware, dComCfg=self.dev_cfg, silent=True
        )
        self.yoko.ctrl.globalFlux.setInstrConfig(self.flux_cfg)

    def set_flux(self, flux: Optional[Number]) -> None:
        if flux is None:
            return  # default do nothing

        # cast numpy float to python float
        if hasattr(flux, "item"):
            flux = flux.item()

        # if not np.issubdtype(flux, np.floating):
        if not isinstance(flux, float):
            raise ValueError(f"Flux must be a float in YokoFluxControl, but got {flux}")
        assert (
            -0.01 <= flux < 0.01
        ), f"Flux must be in the range [-0.01, 0.01], but got {flux}"

        if self.yoko is None:
            self._init_dev()

        for _ in range(5):
            try:
                self.yoko.ctrl.globalFlux.setValue(
                    "Current", flux, rate=self.sweep_rate
                )
                self.yoko.ctrl.globalFlux.setValue(
                    "Current", flux, rate=self.sweep_rate
                )  # again to make sure it is set
                break
            except Exception as e:
                print(f"Error setting flux: {e}, retrying...")
                self._init_dev()
        else:
            print("Failed to set flux")

    def trigger(self):
        pass
