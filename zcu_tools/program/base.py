from collections import defaultdict
from typing import Any, Dict

from tqdm.auto import tqdm

DEFAULT_CALLBACK_TIMES = 50


class MyProgram:
    proxy = None

    @classmethod
    def init_proxy(cls, proxy):
        cls.proxy = proxy

    @classmethod
    def run_in_remote(cls):
        return cls.proxy is not None

    def __init__(self, soccfg, cfg, **kwargs):
        self._parse_cfg(cfg)  # parse config first
        super().__init__(soccfg, cfg=cfg, **kwargs)
        if not self.run_in_remote():
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

    def _override_remote(self, kwargs: dict):
        from zcu_tools.remote.client import pyro_callback  # lazy import

        # remote progress bar
        if kwargs.get("progress", False):
            # replace tqdm progress with callback
            # to make remote progress bar work
            kwargs["progress"] = False

            total = int(self.cfg["rounds"] / kwargs["callback_period"] + 0.99)

            bar = tqdm(total=total, desc="soft_avgs", leave=False)
            if kwargs.get("round_callback") is not None:
                # wrap existing callback
                orig_callback = kwargs["round_callback"]

                def callback_with_bar(*args, **kwargs):
                    bar.update()
                    orig_callback(*args, **kwargs)

                kwargs["round_callback"] = callback_with_bar
            else:

                def update_bar(*args, **kwargs):
                    bar.update()

                kwargs["round_callback"] = update_bar

        # remote callback
        if kwargs.get("round_callback") is not None:
            callback = kwargs["round_callback"]

            def remote_callback(*args, **kwargs):
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    # don't raise error during remote callback execution
                    print(f"Error during remote callback execution: {e}")

            kwargs["round_callback"] = pyro_callback(remote_callback)

        return kwargs

    def set_interrupt(self, err="Unknown error"):
        self._interrupt = True
        self._interrupt_err = err

    def _handle_early_stop(self):
        # call by loop in acquire method
        # handle client-side interrupt
        if self._interrupt:
            print("Interrupted by client-side")
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
        kwargs.setdefault(
            "callback_period", max(self.cfg["rounds"] // DEFAULT_CALLBACK_TIMES, 1)
        )
        if self.run_in_remote():
            return self._remote_acquire(self.proxy.run_program, **kwargs)

        return self._local_acquire(soc, **kwargs)

    def acquire_decimated(self, soc, **kwargs):
        kwargs.setdefault(
            "callback_period", max(self.cfg["rounds"] // DEFAULT_CALLBACK_TIMES, 1)
        )
        if self.run_in_remote():
            return self._remote_acquire(self.proxy.run_program_decimated, **kwargs)

        return self._local_acquire_decimated(soc, **kwargs)

    @classmethod
    def test_remote_callback(cls):
        assert cls.run_in_remote()

        def callback():
            print("Client-side callback executed")

        print("Sending callback to server...")
        cls.proxy.test_callback(callback)
        print("Test finished")
