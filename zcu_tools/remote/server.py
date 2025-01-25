import Pyro4

import zcu_tools.program as zp

SUPPORTED_PROGRAMS = ["OneToneProgram", "TwoToneProgram"]


class ProgramServer:
    def __init__(self, soc):
        self.soc = soc

    @Pyro4.expose
    def run_program(self, name: str, cfg: dict, *args, **kwargs):
        if name not in SUPPORTED_PROGRAMS:
            raise ValueError(f"Program {name} is not supported")

        prog = getattr(zp, name)(self.soc, cfg)
        return prog.acquire(
            self.soc, *args, **kwargs, progress=False, round_callback=None
        )
