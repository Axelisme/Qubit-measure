import time
from typing import Optional
from copy import deepcopy

import Pyro4
import Pyro4.errors

from . import pyro  # noqa , 初始化Pyro4.config
from .wrapper import RemoteCallback

MIN_CALLBACK_INTERVAL = 0.5  # seconds


class ProgramServer:
    def __init__(self, soc, zp):
        self.soc = soc
        self.zp = zp  # zcu_tools.program.v1 or v2

        self.last_prog = None  # last running program
        self.acquiring = False
        self.orig_cb = None  # delayed callback
        self.delay_args = None  # arguments for delayed program execution
        self.prev_t = None  # previous successful callback time

    def _make_prog(self, name: str, cfg: dict):
        return getattr(self.zp, name)(self.soc, cfg)

    def _before_run_program(self, prog, kwargs):
        if self.acquiring:
            raise RuntimeError("Only one program can be run at a time")
        self.last_prog = prog
        self.acquiring = True

        kwargs["progress"] = False  # disable progress bar
        kwargs["round_callback"] = self._wrap_callback(kwargs.get("round_callback"))

        self.delay_args = None
        self.prev_t = None

    def _after_run_program(self):
        self.acquiring = False
        self.prev_t = None
        self.orig_cb = None
        self.delay_args = None

    def _run_remote_callback(self, *args, **kwargs):
        cb = self.orig_cb

        assert isinstance(cb, RemoteCallback), "Invalid callback object"

        # timeout is set to prevent blocking of the server
        timeout = max(MIN_CALLBACK_INTERVAL - 0.1, 0.1)
        cb._pyroTimeout, old = timeout, cb._pyroTimeout  # type: ignore
        try:
            cb.oneway_callback(*args, **kwargs)
        finally:
            cb._pyroTimeout = old  # type: ignore

    def _wrap_callback(self, cb: Optional[RemoteCallback]):
        if cb is None:
            return None  # do nothing

        self.orig_cb = cb

        # wrap callback obj to callable function
        # also drop some calling to reduce network traffic
        def wrapped_cb(*args, **kwargs):
            cur_t = time.time()

            # immediate callback first time
            if self.prev_t is None:
                self.prev_t = -MIN_CALLBACK_INTERVAL - 1

            if cur_t - self.prev_t < MIN_CALLBACK_INTERVAL:
                # delay callback execution to end of acquiring, and ensure it's newest
                # this args will be executed after acquiring data
                self.delay_args = deepcopy((args, kwargs))
                return

            # update tracking info
            self.prev_t = cur_t
            self.delay_args = None  # drop old delayed args

            # don't raise exception in callback
            try:
                self._run_remote_callback(*args, **kwargs)
            except Exception as e:
                print(f"Error during callback execution: {e}")
                self.delay_args = (args, kwargs)  # try again later

        return wrapped_cb

    @Pyro4.expose
    @Pyro4.oneway
    def set_interrupt(self, err="Unknown error"):
        if self.last_prog is not None:
            self.last_prog, prog = None, self.last_prog
            prog.set_interrupt(err)  # set interrupt flag in program
        else:
            print("Warning: no program is running but received KeyboardInterrupt")

    @Pyro4.expose
    def run_program(self, name: str, cfg: dict, decimated: bool, **kwargs):
        prog = self._make_prog(name, cfg)
        self._before_run_program(prog, kwargs)
        try:
            # call original method from MyProgram instead of subclass method
            # in case of multiple execution of overridden method
            ret = prog._local_acquire(self.soc, decimated=decimated, **kwargs)

            # execute delayed callback
            if self.delay_args is not None:
                args, kwargs = self.delay_args  # type: ignore
                try:
                    self._run_remote_callback(*args, **kwargs)
                except Exception as e:
                    print(f"Error during delayed callback execution: {e}")
        finally:
            self._after_run_program()
        return ret

    @Pyro4.expose
    def get_acc_buf(self, prog_name: str):
        if self.last_prog is None:
            raise RuntimeError("No program is running")
        if prog_name != self.last_prog.__class__.__name__:
            raise ValueError(
                f"Program name mismatch: {prog_name} != {self.last_prog.__class__.__name__}"
            )
        return self.last_prog.acc_buf

    @Pyro4.expose
    def test_callback(self, cb: RemoteCallback):
        print("Server received callback test...")
        self._before_run_program((), {})
        self._wrap_callback(cb)(0)  # type: ignore
        self._after_run_program()
        print("Finished callback test")
