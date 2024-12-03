from zcu_tools import make_cfg
from zcu_tools.program import OneToneProgram


def measure_lookback(soc, soccfg, cfg):
    assert cfg.get("reps", 1) == 1, "Only one rep is allowed for lookback"

    prog = OneToneProgram(soccfg, make_cfg(cfg, reps=1))
    IQlist = prog.acquire_decimated(soc, progress=True)

    return IQlist[0]
