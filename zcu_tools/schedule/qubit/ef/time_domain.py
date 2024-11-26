from copy import deepcopy

from zcu_tools.program.ef import EFT1Program, EFT2EchoProgram, EFT2RamseyProgram


def measure_ef_t2ramsey(soc, soccfg, cfg):
    prog = EFT2RamseyProgram(soccfg, deepcopy(cfg))
    ts, avgi, avgq = prog.acquire(soc, progress=True)
    signals = avgi[0][0] + 1j * avgq[0][0]

    return ts, signals


def measure_ef_t1(soc, soccfg, cfg):
    prog = EFT1Program(soccfg, deepcopy(cfg))
    ts, avgi, avgq = prog.acquire(soc, progress=True)
    signals = avgi[0][0] + 1j * avgq[0][0]

    return ts, signals


def measure_ef_t2echo(soc, soccfg, cfg):
    prog = EFT2EchoProgram(soccfg, deepcopy(cfg))
    ts, avgi, avgq = prog.acquire(soc, progress=True)
    signals = avgi[0][0] + 1j * avgq[0][0]

    return ts, signals
