from qick import AveragerProgram, NDAveragerProgram, RAveragerProgram
from zcu_tools.program.base import MyProgram

from .readout import make_readout
from .reset import make_reset

SYNC_TIME = 200  # cycles


class MyProgramV1(MyProgram):
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.resetM = make_reset(cfg["dac"]["reset"])
        self.readoutM = make_readout(cfg["dac"]["readout"])

    def initialize(self):
        self.resetM.init(self)  # type: ignore
        self.readoutM.init(self)  # type: ignore


class MyAveragerProgram(MyProgramV1, AveragerProgram):
    pass


class MyRAveragerProgram(MyProgramV1, RAveragerProgram):
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.cfg["start"] = self.sweep_cfg["start"]
        self.cfg["step"] = self.sweep_cfg["step"]
        self.cfg["expts"] = self.sweep_cfg["expts"]


class MyNDAveragerProgram(MyProgramV1, NDAveragerProgram):
    pass
