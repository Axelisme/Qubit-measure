from typing import Union

from tqdm.auto import tqdm

from zcu_tools.config import config

from .manager import InstrManager


class YokoDevControl:
    yoko = None
    TIMEOUT = 5 * 60  # 5 minutes
    SWEEP_RATE = None  # A/s
    dev_cfg = {}

    @classmethod
    def connect_server(cls, dev_cfg: dict, reinit=False) -> None:
        if cls.yoko is not None:
            if not reinit:
                return  # only register once if not reinit
            cls.disconnect_server()
            print("Reinit YokoDevControl")
        cls.SWEEP_RATE = dev_cfg["outputCfg"]["Current - Sweep rate"]
        cls.dev_cfg = dev_cfg

        # overwrite the cfg
        dev_cfg["dComCfg"]["name"] = "globalFlux"
        dev_cfg["outputCfg"].update(
            {
                "Output": True,
                "Function": "Current",
                "Range (I)": "10 mA",
            }
        )
        if dev_cfg["host_ip"] is None:
            dev_cfg["host_ip"] = "127.0.0.1"

        sHardware = "Yokogawa GS200 DC Source"
        dComCfg = cls.dev_cfg["dComCfg"]
        output_cfg = cls.dev_cfg["outputCfg"]

        if config.YOKO_DRY_RUN:
            print("Dry run mode, skip connecting to Yoko")
            cls.yoko = tuple()  # dummy object that are not None
            cls.cur_val = 0.0
            return

        cls.yoko = InstrManager(server_ip=dev_cfg["host_ip"], timeout=cls.TIMEOUT)
        cls.yoko.add_instrument(sHardware, dComCfg)
        cls.yoko.ctrl.globalFlux.setInstrConfig(output_cfg)

    @classmethod
    def disconnect_server(cls) -> None:
        if cls.yoko is not None:
            cls.yoko = None
            cls.SWEEP_RATE = None
            cls.dev_cfg = {}

    @classmethod
    def get_current(cls) -> float:
        if config.YOKO_DRY_RUN:
            return cls.cur_val

        if cls.yoko is None:
            raise RuntimeError("YokoDevControl not initialized")

        return cls.yoko.ctrl.globalFlux.getValue("Current")

    @classmethod
    def _set_current_direct(cls, value) -> None:
        if config.YOKO_DRY_RUN:
            if cls.cur_val != value:
                cls.cur_val = value
            return
        cls.yoko.ctrl.globalFlux.setValue("Current", value, rate=cls.SWEEP_RATE)

    @classmethod
    def _set_current_smart(cls, value, progress=True) -> None:
        # sweep to the target value step by step
        cur = cls.get_current()
        if cur == value:
            return

        if progress:
            dist = 1e3 * abs(cur - value)
            pbar = tqdm(total=round(dist, 2), unit="mA", leave=False)

        step = 0.5 * cls.SWEEP_RATE  # ~0.5 second a step
        while cur != value:
            if value > cur:
                cur += step
                if cur > value:
                    cur = value
            else:
                cur -= step
                if cur < value:
                    cur = value

            cls._set_current_direct(cur)

            if progress:
                cur_dist = 1e3 * abs(cur - value)
                pbar.update(round(dist - cur_dist, 2) - pbar.n)

        if progress:
            pbar.close()

    @classmethod
    def set_current(cls, value: Union[int, float], progress=True) -> None:
        # cast numpy float to python float
        if hasattr(value, "item"):
            value = value.item()

        # if not np.issubdtype(flux, np.floating):
        if not isinstance(value, float):
            raise ValueError(
                f"Flux must be a float in YokoFluxControl, but got {value}"
            )
        assert -0.01 <= value <= 0.01, (
            f"Flux must be in the range [-0.01, 0.01], but got {value}"
        )

        if cls.yoko is None:
            raise RuntimeError("YokoDevControl not initialized")

        if config.YOKO_DRY_RUN and value != cls.cur_val:
            print(f"DRY RUN: Set current to {value}A\r")

        try:
            cls._set_current_smart(value, progress=progress)
        except KeyboardInterrupt:
            # don't catch KeyboardInterrupt
            raise KeyboardInterrupt
        except Exception as e:
            # reconnect and try again
            print(f"Error in setting current, reconnect and try again: {e}")
            cls.connect_server(cls.dev_cfg, reinit=True)

            cls._set_current_smart(value, progress=progress)
