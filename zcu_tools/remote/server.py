import time

import Pyro4
import Pyro4.errors

from qick.qick_asm import AcquireMixin

from . import pyro  # noqa , 初始化Pyro4.config
from .client import CallbackWrapper

MIN_CALLBACK_INTERVAL = 0.5  # seconds


class ProgramServer:
    def __init__(self, soc, zp):
        self.soc = soc
        self.zp = zp  # zcu_tools.program.v1 or v2

        self.cur_prog = None  # current running program
        self.cur_cfg = None  # current running program config
        self.prev_t = None  # previous successful callback time

    def _make_prog(self, name: str, cfg: dict) -> AcquireMixin:
        return getattr(self.zp, name)(self.soc, cfg)

    def _before_run_program(self, prog, cfg, kwargs):
        if self.cur_prog is not None:
            raise RuntimeError("Only one program can be run at a time")
        self.cur_prog = prog
        self.cur_cfg = cfg
        self.prev_t = None

        kwargs["progress"] = False  # disable progress bar
        if kwargs.get("round_callback") is not None:
            kwargs["round_callback"] = self._wrap_callback(kwargs["round_callback"])

    def _after_run_program(self):
        self.cur_prog = None
        self.cur_cfg = None
        self.prev_t = None

    def _wrap_callback(self, cb: CallbackWrapper):
        def wrapped_cb(ir, *args, **kwargs):
            cur_t = time.time()
            if self.prev_t is not None and cur_t - self.prev_t < MIN_CALLBACK_INTERVAL:
                return  # do nothing if callback is called too frequently

            # don't raise exception in callback
            try:
                cb._pyroTimeout = 1.0  # 1s timeout for callback
                cb.oneway_callback(ir, *args, **kwargs)
                self.prev_t = time.time()
            except Pyro4.errors.CommunicationError as e:
                print(f"Error during callback execution: {e}")

        return wrapped_cb

    @Pyro4.expose
    @Pyro4.oneway
    def set_interrupt(self, err="Unknown error"):
        if self.cur_prog is not None:
            self.cur_prog, prog = None, self.cur_prog
            prog.set_interrupt(err)  # set interrupt flag in program
        else:
            print("Warning: no program is running but received KeyboardInterrupt")

    @Pyro4.expose
    def run_program(self, name: str, cfg: dict, **kwargs):
        prog = self._make_prog(name, cfg)
        self._before_run_program(prog, cfg, kwargs)
        try:
            # call original method from MyProgram instead of subclass method
            # in case of multiple execution of overridden method
            return prog._local_acquire(self.soc, **kwargs)
        finally:
            self._after_run_program()

    @Pyro4.expose
    def run_program_decimated(self, name: str, cfg: dict, **kwargs):
        prog = self._make_prog(name, cfg)
        self._before_run_program(prog, cfg, kwargs)
        try:
            # call original method from MyProgram instead of subclass method
            # in case of multiple execution of overridden method
            return prog._local_acquire_decimated(self.soc, **kwargs)
        finally:
            self._after_run_program()

    @Pyro4.expose
    def test_callback(self, cb: CallbackWrapper):
        print("Server received callback")
        print("executing callback...")
        self._wrap_callback(cb)()
        print("Finished processing callback")
