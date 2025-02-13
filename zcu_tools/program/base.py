from collections import defaultdict
from typing import Any, Dict

from zcu_tools.remote.client import ProgramClient

DEFAULT_CALLBACK_TIMES = 50


class MyProgram:
    proxy: ProgramClient = None

    @classmethod
    def init_proxy(cls, proxy: ProgramClient, test=False):
        if test:
            proxy.test_remote_callback()
        cls.proxy = proxy

    @classmethod
    def clear_proxy(cls):
        cls.proxy = None

    @classmethod
    def is_use_proxy(cls):
        return cls.proxy is not None

    def __init__(self, soccfg, cfg, **kwargs):
        self._parse_cfg(cfg)  # parse config first
        super().__init__(soccfg, cfg=cfg, **kwargs)
        if not self.is_use_proxy():
            # flag for interrupt
            self._interrupt = False
            self._interrupt_err = None

    def _parse_cfg(self, cfg: dict):
        # dac and adc config
        self.dac: Dict[str, Any] = cfg.get("dac", {})
        self.adc: Dict[str, Any] = cfg.get("adc", {})

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

        # other modules
        self.parse_modules(cfg)

    def parse_modules(self, cfg: dict):
        pass

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
        return super().acquire(soc, **kwargs)

    def _local_acquire_decimated(self, soc, **kwargs):
        # non-overridable method, for ProgramServer to call
        return super().acquire_decimated(soc, **kwargs)

    def derive_default_kwargs(self, kwargs: dict):
        # derive default callback_period from soft_avgs
        kwargs.setdefault(
            "callback_period", max(self.cfg["soft_avgs"] // DEFAULT_CALLBACK_TIMES, 1)
        )

        return kwargs

    def acquire(self, soc, **kwargs):
        kwargs = self.derive_default_kwargs(kwargs)
        if self.is_use_proxy():
            return self.proxy.acquire(self, **kwargs)

        return super().acquire(soc, **kwargs)

    def acquire_decimated(self, soc, **kwargs):
        kwargs = self.derive_default_kwargs(kwargs)
        if self.is_use_proxy():
            return self.proxy.acquire_decimated(self, **kwargs)

        return super().acquire_decimated(soc, **kwargs)
