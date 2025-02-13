import Pyro4

from qick.qick_asm import AcquireMixin

from . import pyro  # noqa , 初始化Pyro4.config


class ProgramServer:
    def __init__(self, soc, zp):
        self.soc = soc
        self.cur_prog = None
        self.zp = zp

    def _get_prog(self, name, cfg) -> AcquireMixin:
        if self.cur_prog is not None:
            raise RuntimeError("Only one program can be run at a time")

        prog = getattr(self.zp, name)(self.soc, cfg)
        self.cur_prog = prog
        return prog

    @Pyro4.expose
    @Pyro4.oneway
    def set_interrupt(self, err="Unknown error"):
        if self.cur_prog is not None:
            self.cur_prog, prog = None, self.cur_prog
            prog.set_interrupt(err)
        else:
            print("Warning: no program is running but received KeyboardInterrupt")

    @Pyro4.expose
    def run_program(self, name: str, cfg: dict, *args, **kwargs):
        prog = self._get_prog(name, cfg)
        kwargs["progress"] = False
        # call original method from MyProgram instead of subclass method
        # in case of multiple execution of overridden method
        try:
            return prog._local_acquire(self.soc, *args, **kwargs)
        finally:
            self.cur_prog = None

    @Pyro4.expose
    def run_program_decimated(self, name: str, cfg: dict, *args, **kwargs):
        prog = self._get_prog(name, cfg)
        kwargs["progress"] = False
        # call original method from MyProgram instead of subclass method
        # in case of multiple execution of overridden method
        try:
            return prog._local_acquire_decimated(self.soc, *args, **kwargs)
        finally:
            self.cur_prog = None

    @Pyro4.expose
    def test_callback(self, cb):
        print("Server received callback")
        print("executing callback...")
        try:
            cb._pyroTimeout = 1.0  # s
            cb.oneway_callback()
        except Exception as e:
            print(f"Error during callback execution: {e}")
        else:
            print("callback executed successfully")
        finally:
            print("Finished processing callback")
