from collections import defaultdict

import tqdm.auto as tqdm

from qick import AveragerProgram, NDAveragerProgram, RAveragerProgram
from zcu_tools.remote.client import pyro_callback

from .dry_run import DryRunProgram  # noqa
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

        self.resetM = make_reset(self.dac["reset"])
        self.readoutM = make_readout(self.dac["readout"])

        for name, pulse in self.dac.items():
            if not isinstance(pulse, dict):
                continue
            if hasattr(self, name):
                raise ValueError(f"Pulse name {name} already exists")
            setattr(self, name, pulse)

        self.ch_count = defaultdict(int)
        nqzs = dict()
        for pulse in self.dac.values():
            if not isinstance(pulse, dict):
                continue
            ch, nqz = pulse["ch"], pulse["nqz"]
            self.ch_count[ch] += 1
            cur_nqz = nqzs.setdefault(ch, nqz)
            assert cur_nqz == nqz, "Found different nqz on the same channel"

    def _override_remote(self, kwargs: dict):
        if kwargs.get("progress", False):
            # replace internal progress with callback
            # to make remote progress bar work

            kwargs.setdefault("callback_period", self.cfg["rounds"] // 10)
            total = int(self.cfg["rounds"] / kwargs["callback_period"] + 0.99)

            bar = tqdm.tqdm(
                total=total,
                desc="soft_avgs",
                leave=False,
            )

            kwargs["progress"] = False

            if kwargs.get("round_callback") is not None:
                callback = kwargs["round_callback"]

                def _update(*args, **kwargs):
                    bar.update()
                    callback(*args, **kwargs)
            else:

                def _update(*args, **kwargs):
                    bar.update()

            kwargs["round_callback"] = _update

        kwargs["round_callback"] = (
            pyro_callback(kwargs["round_callback"])
            if kwargs.get("round_callback") is not None
            else None
        )

        return kwargs

    def acquire(self, soc, **kwargs):
        if self.proxy is not None:
            self._override_remote(kwargs)
            return self.proxy.run_program(self.__class__.__name__, self.cfg, **kwargs)

        return super().acquire(soc, **kwargs)

    def acquire_decimated(self, soc, **kwargs):
        if self.proxy is not None:
            self._override_remote(kwargs)
            return self.proxy.run_program_decimated(
                self.__class__.__name__, self.cfg, **kwargs
            )

        return super().acquire_decimated(soc, **kwargs)


class MyAveragerProgram(MyProgram, AveragerProgram):
    pass


class MyRAveragerProgram(MyProgram, RAveragerProgram):
    pass


class MyNDAveragerProgram(MyProgram, NDAveragerProgram):
    pass
