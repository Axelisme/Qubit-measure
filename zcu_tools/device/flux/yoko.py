import time
from numbers import Number
from typing import Optional

from .base import FluxControl


class Labber_YokoFluxControl(FluxControl):
    yoko = None

    @classmethod
    def register(cls, flux_dev: dict, force=False):
        if not force and cls.yoko is not None:
            return  # only register once if not forced

        cls.cfg = flux_dev
        cls.sweep_rate = cls.cfg["flux_cfg"]["sweep_rate"]
        cls.server_ip = flux_dev["server_ip"]

        # overwrite the cfg
        cls.cfg["dev_cfg"]["name"] = "globalFlux"
        cls.cfg["flux_cfg"].update(
            {
                "Output": True,
                "Function": "Current",
                "Range (I)": "10 mA",
            }
        )

        cls._init_dev()

    @classmethod
    def _init_dev(cls):
        from .labber import InstrManager

        sHardware = cls.cfg["sHardware"]
        dev_cfg = cls.cfg["dev_cfg"]
        flux_cfg = cls.cfg["flux_cfg"]

        cls.yoko = InstrManager(server_ip=cls.server_ip)
        cls.yoko.add_instrument(sHardware, dev_cfg, silent=True)
        cls.yoko.ctrl.globalFlux.setInstrConfig(flux_cfg)

    def __init__(self, prog):
        pass

    def set_flux(self, value: Optional[Number]) -> None:
        if value is None:
            return  # default do nothing

        # cast numpy float to python float
        if hasattr(value, "item"):
            value = value.item()

        # if not np.issubdtype(flux, np.floating):
        if not isinstance(value, float):
            raise ValueError(
                f"Flux must be a float in YokoFluxControl, but got {value}"
            )
        assert (
            -0.01 <= value < 0.01
        ), f"Flux must be in the range [-0.01, 0.01], but got {value}"

        yoko = type(self).yoko

        for _ in range(5):
            try:
                # run twice to make sure it is set
                yoko.ctrl.globalFlux.setValue("Current", value, rate=self.sweep_rate)
                yoko.ctrl.globalFlux.setValue("Current", value, rate=self.sweep_rate)
                break
            except Exception as e:
                print(f"Error setting flux: {e}, retrying...")
                time.sleep(5)  # wait for 5 seconds
                type(self)._init_dev()
        else:
            print("Failed to set flux")

    def trigger(self):
        pass
