from copy import deepcopy

import numpy as np

from zcu_tools import make_cfg
from zcu_tools.program.v1 import PowerDepProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import close_show, init_show2d, update_show2d
from zcu_tools.schedule.tools import sweep2array


def measure_qub_pdr_dep(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    freq_cfg = cfg["sweep"]["freq"]
    pdr_cfg = cfg["sweep"]["gain"]
    fpts = sweep2array(freq_cfg, False, "Custom frequency sweep only for soft loop")
    pdrs = sweep2array(pdr_cfg, False, "Custom power sweep only for soft loop")

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    if instant_show:
        fig, ax, dh, im = init_show2d(fpts, pdrs, "Frequency (MHz)", "Power (a.u.)")

    signals2D = np.full((len(pdrs), len(fpts)), np.nan, dtype=np.complex128)
    try:
        prog = PowerDepProgram(soccfg, make_cfg(cfg))
        fpt_pdr, avgi, avgq = prog.acquire(soc, progress=True)  # type: ignore
        signals2D = avgi[0][0] + 1j * avgq[0][0]  # type: ignore
        fpts, pdrs = fpt_pdr[0], fpt_pdr[1]  # type: ignore

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, early stopping the program")
    except Exception as e:
        print("Error during measurement:", e)
    finally:
        if instant_show:
            update_show2d(fig, ax, dh, im, np.abs(signals2D))
            close_show(fig, dh)

    return fpts, pdrs, signals2D  # (pdrs, freqs)
