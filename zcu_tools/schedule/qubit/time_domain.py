from copy import deepcopy

from zcu_tools.program import T1Program, T2EchoProgram, T2RamseyProgram

from ..flux import set_flux


def measure_t2ramsey(soc, soccfg, cfg):
    set_flux(cfg["flux_dev"], cfg["flux"])

    prog = T2RamseyProgram(soccfg, deepcopy(cfg))
    ts, avgi, avgq = prog.acquire(soc, progress=True)
    signals = avgi[0][0] + 1j * avgq[0][0]

    return ts, signals


def measure_t1(soc, soccfg, cfg):
    set_flux(cfg["flux_dev"], cfg["flux"])

    prog = T1Program(soccfg, deepcopy(cfg))
    ts, avgi, avgq = prog.acquire(soc, progress=True)
    signals = avgi[0][0] + 1j * avgq[0][0]

    return ts, signals


def measure_t2echo(soc, soccfg, cfg):
    set_flux(cfg["flux_dev"], cfg["flux"])

    prog = T2EchoProgram(soccfg, deepcopy(cfg))
    ts, avgi, avgq = prog.acquire(soc, progress=True)
    signals = avgi[0][0] + 1j * avgq[0][0]

    return 2 * ts, signals
