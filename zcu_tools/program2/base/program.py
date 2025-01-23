from collections import defaultdict

from qick import AveragerProgram, RAveragerProgram, NDAveragerProgram

from .readout import make_readout
from .reset import make_reset


SYNC_TIME = 200  # cycles


class MyProgram:
    def __init__(self, soccfg, cfg):
        self.dac = cfg.get("dac", {})
        self.adc = cfg.get("adc", {})
        if "sweep" in cfg:
            self.sweep_cfg = cfg["sweep"]
            if isinstance(self.sweep_cfg, dict) and "start" in self.sweep_cfg:
                self.cfg["start"] = self.sweep_cfg["start"]
                self.cfg["step"] = self.sweep_cfg["step"]
                self.cfg["expts"] = self.sweep_cfg["expts"]

        self.resetM = make_reset(cfg["reset"])
        self.readoutM = make_readout(cfg["readout"])

        for name, pulse in self.dac.items():
            if hasattr(self, name):
                raise ValueError(f"Pulse name {name} already exists")
            setattr(self, name, pulse)

        self.ch_count = defaultdict(int)
        nqzs = dict()
        for pulse in self.dac.values():
            ch, nqz = pulse["ch"], pulse["nqz"]
            self.ch_count[ch] += 1
            cur_nqz = nqzs.setdefault(ch, nqz)
            assert cur_nqz == nqz, "Found different nqz on the same channel"

        super().__init__(soccfg, cfg)


class MyAveragerProgram(MyProgram, AveragerProgram):
    pass


class MyRAveragerProgram(MyProgram, RAveragerProgram):
    pass


class MyNDAveragerProgram(MyProgram, NDAveragerProgram):
    pass
