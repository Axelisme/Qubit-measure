from copy import deepcopy

from zcu_tools.program import OnetoneProgram


def measure_lookback(soc, soccfg, cfg):
    assert cfg.get("reps", 1) == 1, "Only one rep is allowed for lookback"
    cfg["reps"] = 1

    prog = OnetoneProgram(soccfg, deepcopy(cfg))
    IQlist = prog.acquire_decimated(soc, progress=True)

    return IQlist[0]
