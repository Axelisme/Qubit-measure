from collections import defaultdict
from typing import Any, Dict

from qick import AveragerProgram, NDAveragerProgram, RAveragerProgram
from zcu_tools.program.base import MyProgram

from .readout import make_readout
from .reset import make_reset

SYNC_TIME = 200  # cycles


class MyProgramV1(MyProgram):
    def _parse_cfg(self, cfg: Dict[str, Any]) -> None:
        super()._parse_cfg(cfg)

        # dac pulse channel check
        self.ch_count = defaultdict(int)
        nqzs = dict()
        for name, pulse in self.dac.items():
            if "ch" not in pulse or "nqz" not in pulse:
                continue
            ch, nqz = pulse["ch"], pulse["nqz"]
            self.ch_count[ch] += 1
            cur_nqz = nqzs.setdefault(ch, nqz)
            assert cur_nqz == nqz, "Found different nqz on the same channel"

        self._make_modules()

    def _make_modules(self) -> None:
        self.resetM = make_reset(self.cfg.get("reset"))
        self.readoutM = make_readout(self.cfg["readout"])

    def initialize(self) -> None:
        self.resetM.init(self)
        self.readoutM.init(self)


class MyAveragerProgram(MyProgramV1, AveragerProgram):
    pass


class MyRAveragerProgram(MyProgramV1, RAveragerProgram):
    def _parse_cfg(self, cfg: Dict[str, Any]) -> None:
        super()._parse_cfg(cfg)
        self.cfg["start"] = self.sweep_cfg["start"]
        self.cfg["step"] = self.sweep_cfg["step"]
        self.cfg["expts"] = self.sweep_cfg["expts"]


class MyNDAveragerProgram(MyProgramV1, NDAveragerProgram):
    pass
