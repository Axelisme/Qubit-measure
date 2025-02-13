from qick import AveragerProgram, NDAveragerProgram, RAveragerProgram
from zcu_tools.program.base import MyProgram

from .readout import make_readout
from .reset import make_reset

SYNC_TIME = 200  # cycles


class MyProgramV1(MyProgram):
    def parse_modules(self, cfg):
        self.resetM = make_reset(cfg["dac"]["reset"])
        self.readoutM = make_readout(cfg["dac"]["readout"])


class MyAveragerProgram(MyProgramV1, AveragerProgram):
    pass


class MyRAveragerProgram(MyProgramV1, RAveragerProgram):
    pass


class MyNDAveragerProgram(MyProgramV1, NDAveragerProgram):
    pass
