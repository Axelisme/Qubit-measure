from numbers import Number

from .manager import InstrManager


class YokoDevControl:
    yoko = None
    TIMEOUT = 1 * 60 * 60 * 1000  # 1 hour

    @classmethod
    def connect_server(cls, dev_cfg: dict, reinit=False):
        if cls.yoko is not None:
            if not reinit:
                return  # only register once if not reinit
            print("Reinit YokoDevControl")

        cls.host_ip = dev_cfg["host_ip"]
        cls.sweep_rate = dev_cfg["outputCfg"]["Current - Sweep rate"]

        # overwrite the cfg
        dev_cfg["dComCfg"]["name"] = "globalFlux"
        dev_cfg["outputCfg"].update(
            {
                "Output": True,
                "Function": "Current",
                "Range (I)": "10 mA",
            }
        )

        cls.dev_cfg = dev_cfg
        cls._init_dev()

    @classmethod
    def _init_dev(cls):
        sHardware = "Yokogawa GS200 DC Source"
        dComCfg = cls.dev_cfg["dComCfg"]
        output_cfg = cls.dev_cfg["outputCfg"]

        cls.yoko = InstrManager(server_ip=cls.host_ip, timeout=cls.TIMEOUT)
        cls.yoko.add_instrument(sHardware, dComCfg)
        cls.yoko.ctrl.globalFlux.setInstrConfig(output_cfg)

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

        if cls.yoko is None:
            raise RuntimeError("YokoDevControl not initialized")

        # run twice to make sure it is set
        cls.yoko.ctrl.globalFlux.setValue("Current", value, rate=cls.sweep_rate)
        cls.yoko.ctrl.globalFlux.setValue("Current", value, rate=cls.sweep_rate)
