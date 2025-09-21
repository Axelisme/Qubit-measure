from types import ModuleType
from typing import Optional, Tuple
from unittest.mock import Mock

import Pyro4

from zcu_tools.program.base import MyProgram
from zcu_tools.utils import AsyncFunc

from .wrapper import RemoteCallback, unwrap_callback


class ProgramServer:
    def __init__(self, soc, zp: ModuleType) -> None:
        self.soc = soc
        self.zp = zp  # zcu_tools.program.v1 or v2

        self.last_prog: Optional[MyProgram] = None  # last running program
        self.acquiring = False

    def _make_prog(self, name: str, cfg: dict) -> MyProgram:
        return getattr(self.zp, name)(self.soc, cfg)

    def _before_run_program(self, prog: MyProgram, kwargs: dict) -> None:
        if self.acquiring:
            raise RuntimeError("Only one program can be run at a time")
        self.last_prog = prog
        self.acquiring = True

        kwargs["progress"] = False  # disable progress bar

    def _after_run_program(self) -> None:
        self.acquiring = False

    @Pyro4.expose
    @Pyro4.oneway
    def set_early_stop(self, silent: bool = False) -> None:
        if self.last_prog is not None:
            # set interrupt flag in program
            self.last_prog.set_early_stop(silent=silent)
        else:
            print("Warning: no program is running but received early stop signal")

    @Pyro4.expose
    def run_program(self, name: str, cfg: dict, decimated: bool, **kwargs) -> list:
        prog = self._make_prog(name, cfg)
        self._before_run_program(prog, kwargs)
        callback = unwrap_callback(kwargs.get("callback"))
        try:
            # use async callback to prevent blocking by connection
            with AsyncFunc(callback) as cb:
                kwargs["callback"] = cb

                # call original method from MyProgram instead of subclass method
                # in case of multiple execution of overridden method
                if decimated:
                    return prog.local_acquire_decimated(self.soc, **kwargs)
                else:
                    return prog.local_acquire(self.soc, **kwargs)
        finally:
            self._after_run_program()

    @Pyro4.expose
    def get_raw(self) -> Optional[list]:
        if self.acquiring:
            raise RuntimeError("Program is still running")

        if self.last_prog is None:
            raise RuntimeError("No program has been run")

        return self.last_prog.get_raw()

    @Pyro4.expose
    def get_shots(self) -> Optional[list]:
        if self.acquiring:
            raise RuntimeError("Program is still running")

        if self.last_prog is None:
            raise RuntimeError("No program has been run")

        return self.last_prog.get_shots()

    @Pyro4.expose
    def get_round_data(self) -> Tuple[Optional[list], Optional[list]]:
        if self.acquiring:
            raise RuntimeError("Program is still running")

        if self.last_prog is None:
            raise RuntimeError("No program has been run")

        return self.last_prog.get_rounds(), self.last_prog.get_stderr_raw()

    @Pyro4.expose
    def test_callback(self, cb: RemoteCallback) -> None:
        print("Server received callback test...")
        self._before_run_program(Mock(), {})
        try:
            callback = unwrap_callback(cb)
            assert callback is not None  # of course
            callback(0)  # test callback
        finally:
            self._after_run_program()
        print("Finished callback test")
