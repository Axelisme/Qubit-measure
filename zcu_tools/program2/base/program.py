from qick import AveragerProgram, RAveragerProgram, NDAveragerProgram

from .readout import make_readout
from .reset import make_reset


SYNC_TIME = 200  # cycles


def parser_prog(prog):
    prog.dac = prog.cfg["dac"]
    prog.adc = prog.cfg["adc"]
    if "sweep" in prog.cfg:
        prog.sweep_cfg = prog.cfg["sweep"]

    prog.resetM = make_reset(prog.cfg["reset"])
    prog.readoutM = make_readout(prog.cfg["readout"])

    for name, pulse in prog.dac.items():
        if hasattr(prog, name):
            raise ValueError(f"Pulse name {name} already exists")
        setattr(prog, name, pulse)

    prog.ch_count = [0 for _ in range(16)]
    for pulse in prog.dac.values():
        prog.ch_count[pulse["ch"]] += 1


class MyAveragerProgram(AveragerProgram):
    def __init__(self, soc, soccfg):
        super().__init__(soc, soccfg)
        parser_prog(self)


class MyRAveragerProgram(RAveragerProgram):
    def __init__(self, soc, soccfg):
        super().__init__(soc, soccfg)
        parser_prog(self)


class MyNDAveragerProgram(NDAveragerProgram):
    def __init__(self, soc, soccfg):
        super().__init__(soc, soccfg)
        parser_prog(self)
