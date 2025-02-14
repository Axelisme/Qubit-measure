from qick import AveragerProgram, NDAveragerProgram, RAveragerProgram
from zcu_tools.program.base import MyProgram

from .readout import make_readout
from .reset import make_reset

SYNC_TIME = 200  # cycles


class MyProgramV1(MyProgram):
    def _parse_cfg(self, cfg):
        self.resetM = make_reset(cfg["dac"]["reset"])
        self.readoutM = make_readout(cfg["dac"]["readout"])
        return super()._parse_cfg(cfg)

    def initialize(self):
        self.resetM.init(self)  # type: ignore
        self.readoutM.init(self)  # type: ignore


class MyAveragerProgram(MyProgramV1, AveragerProgram):
    pass


class MyRAveragerProgram(MyProgramV1, RAveragerProgram):
    pass


class MyNDAveragerProgram(MyProgramV1, NDAveragerProgram):
    pass
