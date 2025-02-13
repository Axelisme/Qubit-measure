import time

import Pyro4
import Pyro4.errors

from . import pyro  # noqa , 初始化Pyro4.config
from .wrapper import CallbackWrapper

MIN_CALLBACK_INTERVAL = 0.5  # seconds


class ProgramServer:
    def __init__(self, soc, zp):
        self.soc = soc
        self.zp = zp  # zcu_tools.program.v1 or v2

        self.cur_prog = None  # current running program
        self.cur_cb = None  # delayed callback
        self.delay_args = None  # arguments for delayed program execution
        self.prev_t = None  # previous successful callback time

    def _make_prog(self, name: str, cfg: dict):
        return getattr(self.zp, name)(self.soc, cfg)

    def _before_run_program(self, prog, kwargs):
        if self.cur_prog is not None:
            raise RuntimeError("Only one program can be run at a time")
        self.cur_prog = prog

        kwargs["progress"] = False  # disable progress bar
        if kwargs.get("round_callback") is not None:
            kwargs["round_callback"] = self._wrap_callback(kwargs["round_callback"])

        self.cur_cb = kwargs.get("round_callback")
        self.delay_args = None
        self.prev_t = None

    def _after_run_program(self):
        self.prev_t = None  # reset previous time
        if self.delay_args is not None:
            # because prev_t is None, this must be executed
            self.cur_cb(*self.delay_args[0], **self.delay_args[1])
        self.cur_prog = None
        self.cur_cb = None
        self.delay_args = None

    def _wrap_callback(self, cb: CallbackWrapper):
        def wrapped_cb(*args, **kwargs):
            cur_t = time.time()
            if self.prev_t is not None and cur_t - self.prev_t < MIN_CALLBACK_INTERVAL:
                # delay callback execution, and ensure it's newest
                # this args will be executed after acquiring data
                self.delay_args = (args, kwargs)
                return

            # don't raise exception in callback
            try:
                cb._pyroTimeout = 1.0  # 1s timeout for callback
                cb.oneway_callback(*args, **kwargs)
                self.prev_t = time.time()
            except Exception as e:
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
        self._before_run_program(prog, kwargs)
        try:
            # call original method from MyProgram instead of subclass method
            # in case of multiple execution of overridden method
            ret = prog._local_acquire(self.soc, **kwargs)
        finally:
            self._after_run_program()
        return ret

    @Pyro4.expose
    def run_program_decimated(self, name: str, cfg: dict, **kwargs):
        prog = self._make_prog(name, cfg)
        self._before_run_program(prog, kwargs)
        try:
            # call original method from MyProgram instead of subclass method
            # in case of multiple execution of overridden method
            ret = prog._local_acquire_decimated(self.soc, **kwargs)
        finally:
            self._after_run_program()
        return ret

    @Pyro4.expose
    def test_callback(self, cb: CallbackWrapper):
        print("Server received callback")
        print("executing callback...")
        self._wrap_callback(cb)()
        print("Finished processing callback")
