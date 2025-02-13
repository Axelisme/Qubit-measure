from copy import deepcopy

from zcu_tools import make_cfg
from zcu_tools.program.v2 import OneToneProgram

from ..flux import set_flux
from ..tools import format_sweep, sweep2param


def measure_res_freq(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    sweep = cfg["sweep"]

    cfg["sweep"] = format_sweep(sweep, "res_freq")
    res_pulse["freq"] = sweep2param(cfg["sweep"])
    cfg = make_cfg(cfg)

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    prog = OneToneProgram(soccfg, cfg)
    fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)
    IQlist = prog.acquire(soc, soft_avgs=cfg["soft_avgs"], progress=True)

    return fpts, IQlist[0][0].dot([1, 1j])  # pyright: ignore
