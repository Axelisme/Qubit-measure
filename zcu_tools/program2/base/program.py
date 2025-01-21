from qick import AveragerProgram, RAveragerProgram, NDAveragerProgram

from .readout import make_readout
from .reset import make_reset


SYNC_TIME = 200  # cycles


def parser_prog(prog, cfg):
    prog.dac = cfg["dac"]
    prog.adc = cfg["adc"]
    if "sweep" in cfg:
        prog.sweep_cfg = cfg["sweep"]

    prog.resetM = make_reset(cfg["reset"])
    prog.readoutM = make_readout(cfg["readout"])

    for name, pulse in prog.dac.items():
        if hasattr(prog, name):
            raise ValueError(f"Pulse name {name} already exists")
        setattr(prog, name, pulse)

    prog.ch_count = [0 for _ in range(16)]
    nqzs = dict()
    for pulse in prog.dac.values():
        prog.ch_count[pulse["ch"]] += 1
        cur_nqz = nqzs.setdefault(pulse["ch"], pulse["nqz"])
        assert cur_nqz == pulse["nqz"], "Found different nqz on the same channel"


class MyAveragerProgram(AveragerProgram):
    def __init__(self, soccfg, cfg):
        parser_prog(self, cfg)
        super().__init__(soccfg, cfg)


class MyRAveragerProgram(RAveragerProgram):
    def __init__(self, soccfg, cfg):
        parser_prog(self, cfg)
        super().__init__(soccfg, cfg)


class MyNDAveragerProgram(NDAveragerProgram):
    def __init__(self, soccfg, cfg):
        parser_prog(self, cfg)
        super().__init__(soccfg, cfg)
