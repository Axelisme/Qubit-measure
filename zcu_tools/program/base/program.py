from collections import defaultdict

import Pyro4
from qick import AveragerProgram, RAveragerProgram, NDAveragerProgram

from .readout import make_readout
from .reset import make_reset


SYNC_TIME = 200  # cycles


class MyProgram:
    proxy = None

    @classmethod
    def init_proxy(cls, proxy):
        cls.proxy = proxy

    def __init__(self, soccfg, cfg):
        if self.proxy is not None:
            # use remote proxy, so we don't need to do anything
            self.cfg = cfg
        else:
            self._parse_cfg(cfg)
            super().__init__(soccfg, cfg)

    def _parse_cfg(self, cfg: dict):
        self.dac = cfg.get("dac", {})
        self.adc = cfg.get("adc", {})
        if "sweep" in cfg:
            self.sweep_cfg = cfg["sweep"]
            if isinstance(self.sweep_cfg, dict) and "start" in self.sweep_cfg:
                cfg["start"] = self.sweep_cfg["start"]
                cfg["step"] = self.sweep_cfg["step"]
                cfg["expts"] = self.sweep_cfg["expts"]

        self.resetM = make_reset(cfg["reset"])
        self.readoutM = make_readout(cfg["readout"])

        for name, pulse in self.dac.items():
            if hasattr(self, name):
                raise ValueError(f"Pulse name {name} already exists")
            setattr(self, name, pulse)

        self.ch_count = defaultdict(int)
        nqzs = dict()
        for pulse in self.dac.values():
            ch, nqz = pulse["ch"], pulse["nqz"]
            self.ch_count[ch] += 1
            cur_nqz = nqzs.setdefault(ch, nqz)
            assert cur_nqz == nqz, "Found different nqz on the same channel"

    def _override_cfg(self, kwargs: dict):
        kwargs["progress"] = False  # progress bar is not supported
        kwargs["round_callback"] = None  # callback is not supported

        return kwargs

    def acquire(self, soc, *args, **kwargs):
        if self.proxy is not None:
            self._override_cfg(kwargs)
            try:
                return self.proxy.run_program(
                    self.__class__.__name__, self.cfg, *args, **kwargs
                )
            except Pyro4.errors.CommunicationError as e:
                print("Error: ", e)
                return None

        return super().acquire(soc, *args, **kwargs)

    def acquire_decimated(self, soc, *args, **kwargs):
        if self.proxy is not None:
            self._override_cfg(kwargs)
            try:
                return self.proxy.run_program_decimated(
                    self.__class__.__name__, self.cfg, *args, **kwargs
                )
            except Pyro4.errors.CommunicationError as e:
                print("Error: ", e)
                return None

        return super().acquire_decimated(soc, *args, **kwargs)


class MyAveragerProgram(MyProgram, AveragerProgram):
    pass


class MyRAveragerProgram(MyProgram, RAveragerProgram):
    pass


class MyNDAveragerProgram(MyProgram, NDAveragerProgram):
    pass
