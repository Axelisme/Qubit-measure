from numpy.typing import NDArray

from zcu_tools.program import T2RamseyProgram, T1Program, T2EchoProgram


def measure_t2ramsey(soc, soccfg, cfg) -> tuple[NDArray, NDArray]:
    prog = T2RamseyProgram(soccfg, cfg)
    ts, avgi, avgq = prog.acquire(soc, progress=True)
    signals = avgi[0][0] + 1j * avgq[0][0]

    return ts, signals


def measure_t1(soc, soccfg, cfg) -> tuple[NDArray, NDArray]:
    prog = T1Program(soccfg, cfg)
    ts, avgi, avgq = prog.acquire(soc, progress=True)
    signals = avgi[0][0] + 1j * avgq[0][0]

    return ts, signals


def measure_t2echo(soc, soccfg, cfg) -> tuple[NDArray, NDArray]:
    prog = T2EchoProgram(soccfg, cfg)
    ts, avgi, avgq = prog.acquire(soc, progress=True)
    signals = avgi[0][0] + 1j * avgq[0][0]

    return ts, signals
