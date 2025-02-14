import warnings
from collections import defaultdict
from typing import Any, Dict, Optional

from zcu_tools.remote.client import ProgramClient


class MyProgram:
    proxy: Optional[ProgramClient] = None

    @classmethod
    def init_proxy(cls, proxy: ProgramClient, test=False):
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
        super().__init__(soccfg, cfg=cfg, **kwargs)  # type: ignore
        if not self.is_use_proxy():
            # flag for interrupt
            self._interrupt = False
            self._interrupt_err = None

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

    def set_interrupt(self, err="Unknown error"):
        # acquire method will check this flag
        self._interrupt = True
        self._interrupt_err = err

    def _handle_early_stop(self):
        # call by loop in acquire method
        # handle client-side interrupt
        if self._interrupt:
            print("Interrupted by client-side")
            raise RuntimeError(self._interrupt_err)

    def _local_acquire(self, soc, **kwargs):
        # non-overridable method, for ProgramServer to call
        return super().acquire(soc, **kwargs)  # type: ignore

    def _local_acquire_decimated(self, soc, **kwargs):
        # non-overridable method, for ProgramServer to call
        return super().acquire_decimated(soc, **kwargs)  # type: ignore

    def acquire(self, soc, **kwargs):
        if self.is_use_proxy():
            return self.proxy.acquire(self, **kwargs)  # type: ignore

        return super().acquire(soc, **kwargs)  # type: ignore

    def acquire_decimated(self, soc, **kwargs):
        if self.is_use_proxy():
            return self.proxy.acquire_decimated(self, **kwargs)  # type: ignore

        return super().acquire_decimated(soc, **kwargs)  # type: ignore
