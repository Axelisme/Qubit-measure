from collections import defaultdict

import tqdm.auto as tqdm

from qick import AveragerProgram, NDAveragerProgram, RAveragerProgram

from .readout import make_readout
from .reset import make_reset

SYNC_TIME = 200  # cycles


class MyProgram:
    proxy = None

    @classmethod
    def init_proxy(cls, proxy):
        cls.proxy = proxy

    def run_in_remote(self):
        return self.proxy is not None

    def __init__(self, soccfg, cfg):
        self._parse_cfg(cfg)
        if self.run_in_remote():
            # use remote proxy, so we don't need to do anything
            self.soccfg = soccfg
            self.cfg = cfg
        else:
            super().__init__(soccfg, cfg)
            self._interrupt = False
            self._interrupt_err = None

    def _parse_cfg(self, cfg: dict):
        self.dac = cfg.get("dac", {})
        self.adc = cfg.get("adc", {})
        if "sweep" in cfg:
            self.sweep_cfg = cfg["sweep"]
            if isinstance(self.sweep_cfg, dict) and "start" in self.sweep_cfg:
                cfg["start"] = self.sweep_cfg["start"]
                cfg["step"] = self.sweep_cfg["step"]
                cfg["expts"] = self.sweep_cfg["expts"]

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

        self.resetM = make_reset(self.dac["reset"])
        self.readoutM = make_readout(self.dac["readout"])

    def _override_remote(self, kwargs: dict):
        from zcu_tools.remote.client import pyro_callback  # lazy import

        if kwargs.get("progress", False):
            # replace internal progress with callback
            # to make remote progress bar work

            kwargs.setdefault("callback_period", self.cfg["rounds"] // 50)
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

    def set_interrupt(self, err="Unknown error"):
        self._interrupt = True
        self._interrupt_err = err

    def _handle_early_stop(self):
        # call by loop in acquire method
        # handle client-side interrupt
        if self._interrupt:
            print("Interrupted by client")
            raise RuntimeError(self._interrupt_err)

    def _remote_acquire(self, remote_func, **kwargs):
        self._override_remote(kwargs)
        prog_name = self.__class__.__name__
        try:
            # call remote function
            return remote_func(prog_name, self.cfg, **kwargs)
        except BaseException as e:
            import sys

            # if find '_pyroTraceback' in error value, it's a remote error
            # if not, need to raise it on remote side
            if not hasattr(sys.exc_info()[1], "_pyroTraceback"):
                print("Client-side error, raise it on remote side...")
                self.proxy._pyroTimeout, old = 1, self.proxy._pyroTimeout
                self.proxy.set_interrupt(str(e))
                self.proxy._pyroTimeout = old

            raise e

    def _local_acquire(self, soc, **kwargs):
        # non-overridable method
        return super().acquire(soc, **kwargs)

    def _local_acquire_decimated(self, soc, **kwargs):
        # non-overridable method
        return super().acquire_decimated(soc, **kwargs)

    def acquire(self, soc, **kwargs):
        if self.run_in_remote():
            return self._remote_acquire(self.proxy.run_program, **kwargs)

        return self._local_acquire(soc, **kwargs)

    def acquire_decimated(self, soc, **kwargs):
        if self.run_in_remote():
            return self._remote_acquire(self.proxy.run_program_decimated, **kwargs)

        return self._local_acquire_decimated(soc, **kwargs)


class MyAveragerProgram(MyProgram, AveragerProgram):
    pass


class MyRAveragerProgram(MyProgram, RAveragerProgram):
    pass


class MyNDAveragerProgram(MyProgram, NDAveragerProgram):
    pass
