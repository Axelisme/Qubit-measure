from numpy.typing import NDArray

from zcu_tools.program import OnetoneProgram


def measure_lookback(soc, soccfg, cfg) -> tuple[NDArray, NDArray]:
    prog = OnetoneProgram(soccfg, cfg)
    IQlist = prog.acquire_decimated(soc, progress=True).sum(axis=1)

    return IQlist[0]
