from copy import deepcopy

from zcu_tools.program import OnetoneProgram


def measure_lookback(soc, soccfg, cfg):
    prog = OnetoneProgram(soccfg, deepcopy(cfg))
    IQlist = prog.acquire_decimated(soc, progress=True).sum(axis=1)

    return IQlist[0]
