import threading
import warnings
from collections import defaultdict
from queue import Queue
from typing import Any, Callable, Dict, Optional

from zcu_tools.remote.client import ProgramClient


class CallbackWrapper:
    def __init__(self, func: Optional[Callable]):
        self.func = func

    def __enter__(self):
        if self.func is None:
            return None  # do nothing

        self.lock = threading.Lock()

        # these variables are protected by lock
        self.acquiring = True
        self.last_ir = -1  # initial to -1 to accept the first job

        # start worker thread
        self.worker_t = threading.Thread(target=self.exec, daemon=True)
        self.last_job = Queue(maxsize=1)

        self.worker_t.start()  # start worker thread

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.func is None:
            return  # do nothing

        self.acquiring = False
        with self.lock:
            self.clear_last_job()
            self.last_job.put_nowait(None)  # tell worker no more job
        self.worker_t.join(2)

    def clear_last_job(self):
        assert self.lock.locked(), "This method should be called within lock"
        while not self.last_job.empty():
            self.last_job.get_nowait()

    def exec(self):
        assert self.func is not None, "This method should not be called if func is None"
        while True:
            job = self.last_job.get()
            if job is None:
                break  # end of job

            # do not raise exception in this thread
            try:
                ir, args, kwargs = job
                self.func(ir, *args, **kwargs)
            except BaseException as e:
                print(f"Error in callback: {e}")

    def __call__(self, ir: int, *args, **kwargs):
        # this method may be called concurrently, so we need to protect it
        with self.lock:
            # only keep the latest job
            if ir > self.last_ir and self.acquiring:
                self.last_ir = ir
                self.clear_last_job()
                self.last_job.put_nowait((ir, args, kwargs))


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

    def _local_acquire(self, soc, decimated=False, **kwargs):
        # non-overridable method, for ProgramServer to call
        try:
            if decimated:
                return super().acquire_decimated(soc, **kwargs)  # type: ignore
            return super().acquire(soc, **kwargs)  # type: ignore
        finally:
            soc.reset_gens()  # reset the tProc

    def acquire(self, soc, **kwargs):
        with CallbackWrapper(kwargs.get("round_callback")) as cb:
            kwargs["round_callback"] = cb

            if self.is_use_proxy():
                return self.proxy.acquire(self, **kwargs)  # type: ignore

            return self._local_acquire(soc, decimated=False, **kwargs)

    def acquire_decimated(self, soc, **kwargs):
        with CallbackWrapper(kwargs.get("round_callback")) as cb:
            kwargs["round_callback"] = cb

            if self.is_use_proxy():
                return self.proxy.acquire_decimated(self, **kwargs)  # type: ignore

            return self._local_acquire(soc, decimated=True, **kwargs)
