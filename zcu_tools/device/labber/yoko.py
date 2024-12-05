import time
from numbers import Number

from .manager import InstrManager


class YokoDevControl:
    yoko = None

    @classmethod
    def connect_server(cls, flux_dev: dict, reinit=False):
        if not reinit and cls.yoko is not None:
            print("YokoDevControl already registered, do nothing")
            return  # only register once if not reinit

        cls.flux_dev = flux_dev
        cls.sweep_rate = cls.flux_dev["flux_cfg"]["Current - Sweep rate"]
        cls.server_ip = flux_dev["server_ip"]

        # overwrite the cfg
        cls.flux_dev["dev_cfg"]["name"] = "globalFlux"
        cls.flux_dev["flux_cfg"].update(
            {
                "Output": True,
                "Function": "Current",
                "Range (I)": "10 mA",
            }
        )

        cls._init_dev()

    @classmethod
    def _init_dev(cls):
        sHardware = cls.flux_dev["sHardware"]
        dev_cfg = cls.flux_dev["dev_cfg"]
        flux_cfg = cls.flux_dev["flux_cfg"]

        cls.yoko = InstrManager(server_ip=cls.server_ip, timeout=25 * 60 * 1000)
        cls.yoko.add_instrument(sHardware, dev_cfg, silent=True)
        cls.yoko.ctrl.globalFlux.setInstrConfig(flux_cfg)

    @classmethod
    def set_current(cls, value: Number) -> None:
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

        for _ in range(5):
            try:
                # run twice to make sure it is set
                cls.yoko.ctrl.globalFlux.setValue("Current", value, rate=cls.sweep_rate)
                cls.yoko.ctrl.globalFlux.setValue("Current", value, rate=cls.sweep_rate)
                break
            except Exception as e:
                print(f"Error setting flux: {e}, retrying...")
                for _ in range(10):
                    try:
                        time.sleep(60)  # wait for 1 min
                        cls._init_dev()
                        break
                    except Exception as e:
                        print("Error init yoko device: ", e)
                else:
                    raise RuntimeError("Failed to set flux")
        else:
            raise RuntimeError("Failed to set flux")
