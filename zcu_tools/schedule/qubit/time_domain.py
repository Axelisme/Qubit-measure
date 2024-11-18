from copy import deepcopy

from zcu_tools.program import T1Program, T2EchoProgram, T2RamseyProgram


def measure_t2ramsey(soc, soccfg, cfg):
    prog = T2RamseyProgram(soccfg, deepcopy(cfg))
    ts, avgi, avgq = prog.acquire(soc, progress=True)
    signals = avgi[0][0] + 1j * avgq[0][0]

    return ts, signals


def measure_t1(soc, soccfg, cfg):
    prog = T1Program(soccfg, deepcopy(cfg))
    ts, avgi, avgq = prog.acquire(soc, progress=True)
    signals = avgi[0][0] + 1j * avgq[0][0]

    return ts, signals


def measure_t2echo(soc, soccfg, cfg):
    prog = T2EchoProgram(soccfg, deepcopy(cfg))
    ts, avgi, avgq = prog.acquire(soc, progress=True)
    signals = avgi[0][0] + 1j * avgq[0][0]

    return ts, signals
