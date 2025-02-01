import Pyro4

import zcu_tools.program as zp

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

    def _get_prog(self, name, cfg):
        if name not in SUPPORTED_PROGRAMS:
            raise ValueError(f"Program {name} is not supported")

        return getattr(zp, name)(self.soc, cfg)

    @Pyro4.expose
    def run_program(self, name: str, cfg: dict, *args, **kwargs):
        prog = self._get_prog(name, cfg)
        return prog.acquire(
            self.soc, *args, **kwargs, progress=False, round_callback=None
        )

    @Pyro4.expose
    def run_program_decimated(self, name: str, cfg: dict, *args, **kwargs):
        prog = self._get_prog(name, cfg)
        return prog.acquire_decimated(
            self.soc, *args, **kwargs, progress=False, round_callback=None
        )
