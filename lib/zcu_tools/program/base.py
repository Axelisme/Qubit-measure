import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Optional

from qick.qick_asm import AcquireMixin
from zcu_tools.config import config
from zcu_tools.tools import AsyncFunc


class AbsProxy(ABC):
    @abstractmethod
    def test_remote_callback(self) -> bool:
        pass

    @abstractmethod
    def get_acc_buf(self, prog) -> list:
        pass

    @abstractmethod
    def set_early_stop(self) -> None:
        pass

    @abstractmethod
    def acquire(self, prog, decimated, **kwargs) -> list:
        pass


class MyProgram(AcquireMixin):
    proxy: Optional[AbsProxy] = None

    @classmethod
    def init_proxy(cls, proxy: AbsProxy, test=False):
        if test:
            success = proxy.test_remote_callback()
            if not success:
                warnings.warn(
                    "Callback test failed, remote callback may not work, you should check your LOCAL_IP or LOCAL_PORT, it may be blocked by firewall"
                )

        cls.proxy = proxy

    @classmethod
    def clear_proxy(cls):
        cls.proxy = None

    @classmethod
    def is_use_proxy(cls):
        return cls.proxy is not None

    def __init__(self, soccfg, cfg: Dict[str, Any], **kwargs):
        self._parse_cfg(cfg)  # parse config first
        super().__init__(soccfg, cfg=cfg, **kwargs)

        self.buf_expired = False

    def _parse_cfg(self, cfg: dict):
        # dac and adc config
        self.cfg = cfg
        self.dac: Dict[str, Any] = cfg.get("dac", {})
        self.adc: Dict[str, Any] = cfg.get("adc", {})
        if "sweep" in cfg:
            self.sweep_cfg = cfg["sweep"]

        # dac pulse
        for name, pulse in self.dac.items():
            if not isinstance(pulse, dict) or not name.endswith("_pulse"):
                continue
            if hasattr(self, name):
                raise ValueError(f"Pulse name {name} already exists")
            setattr(self, name, pulse)

        # dac pulse channel count
        self.ch_count = defaultdict(int)
        nqzs = dict()
        for pulse in self.dac.values():
            if not isinstance(pulse, dict) or "ch" not in pulse:
                continue
            ch, nqz = pulse["ch"], pulse["nqz"]
            self.ch_count[ch] += 1
            cur_nqz = nqzs.setdefault(ch, nqz)
            assert cur_nqz == nqz, "Found different nqz on the same channel"

    def set_early_stop(self) -> None:
        # tell program to return as soon as possible
        if self.is_use_proxy():
            self.proxy.set_early_stop()
        else:
            print("Program received early stop signal")
            self.early_stop = True

    def _local_acquire(self, soc, decimated=False, **kwargs) -> list:
        # non-overridable method, for ProgramServer to call
        if self.is_use_proxy():
            raise RuntimeError("_local_acquire should not be called when using proxy")

        if decimated:
            return super().acquire_decimated(soc, **kwargs)
        return super().acquire(soc, **kwargs)

    def __getattribute__(self, name):
        if name == "acc_buf":
            # intercept acc_buf to fetch from proxy
            if self.is_use_proxy():
                # fetch acc_buf from proxy
                if self.buf_expired:
                    self.acc_buf = self.proxy.get_acc_buf(self)
                    self.buf_expired = False
        return object.__getattribute__(self, name)

    def _acquire(self, soc, decimated=False, **kwargs) -> list:
        with AsyncFunc(kwargs.get("callback")) as cb:
            kwargs["callback"] = cb

            if self.is_use_proxy():
                self.acc_buf = None  # clear local acc_buf
                self.buf_expired = True
                return self.proxy.acquire(self, decimated=decimated, **kwargs)

            if config.ZCU_DRY_RUN:
                raise NotImplementedError(
                    "ZCU_DRY_RUN is enabled, but not supported in local mode"
                )

            return self._local_acquire(soc, decimated=decimated, **kwargs)

    def acquire(self, soc, **kwargs):
        return self._acquire(soc, decimated=False, **kwargs)

    def acquire_decimated(self, soc, **kwargs):
        return self._acquire(soc, decimated=True, **kwargs)
