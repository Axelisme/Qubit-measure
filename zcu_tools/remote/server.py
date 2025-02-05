import Pyro4

import zcu_tools.program as zp

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = set(["pickle"])
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4


SUPPORTED_PROGRAMS = [
    "OneToneProgram",
    "RGainOneToneProgram",
    "TwoToneProgram",
    "RGainTwoToneProgram",
    "RFreqTwoToneProgram",
    "RFreqTwoToneProgramWithRedReset",
    "PowerDepProgram",
    "T1Program",
    "T2RamseyProgram",
    "T2EchoProgram",
    "SingleShotProgram",
]


class ProgramServer:
    def __init__(self, soc):
        self.soc = soc
        self.cur_prog = None

    def _get_prog(self, name, cfg):
        if name not in SUPPORTED_PROGRAMS:
            raise ValueError(f"Program {name} is not supported")

        if self.cur_prog is not None:
            raise RuntimeError("Only one program can be run at a time")

        prog = getattr(zp, name)(self.soc, cfg)
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
        result = prog.acquire(self.soc, *args, **kwargs)
        self.cur_prog = None
        return result

    @Pyro4.expose
    def run_program_decimated(self, name: str, cfg: dict, *args, **kwargs):
        prog = self._get_prog(name, cfg)
        kwargs["progress"] = False
        result = prog.acquire_decimated(self.soc, *args, **kwargs)
        self.cur_prog = None
        return result
