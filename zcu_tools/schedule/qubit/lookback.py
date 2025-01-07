import numpy as np

from zcu_tools import make_cfg
from zcu_tools.program import TwoToneProgram

from ..flux import set_flux


def measure_excited_lookback(soc, soccfg, cfg, progress=True):
    assert cfg.get("reps", 1) == 1, "Only one rep is allowed for lookback"

    set_flux(cfg["flux_dev"], cfg["flux"])

    prog = TwoToneProgram(soccfg, make_cfg(cfg, reps=1))
    IQlist = prog.acquire_decimated(soc, progress=progress)
    Is, Qs = IQlist[0]
    Ts = prog.cycles2us(1, ro_ch=cfg["adc"]["chs"][0]) * np.arange(len(Is))

    return Ts, Is, Qs
