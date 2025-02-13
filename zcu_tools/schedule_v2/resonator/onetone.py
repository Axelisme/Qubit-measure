from copy import deepcopy

from qick.asm_v2 import QickSweep1D

from zcu_tools import make_cfg
from zcu_tools.program_v2 import OneToneProgram, DEFAULT_LOOP_NAME

from ..flux import set_flux


def measure_res_freq(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    cfg["dac"]["res_pulse"]["freq"] = QickSweep1D(
        DEFAULT_LOOP_NAME, cfg["sweep"]["start"], cfg["sweep"]["stop"]
    )
    cfg = make_cfg(cfg)
    prog = OneToneProgram(soccfg, cfg)
    fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)
    IQlist = prog.acquire(soc, soft_avgs=cfg["soft_avgs"], progress=True)

    return fpts, IQlist[0][0].dot([1, 1j])  # pyright: ignore
