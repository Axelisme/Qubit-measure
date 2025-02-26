from copy import deepcopy

import numpy as np

from zcu_tools import make_cfg
from zcu_tools.program.v1 import PowerDepProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import InstantShow
from zcu_tools.schedule.tools import sweep2array


def measure_qub_pdr_dep(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    freq_cfg = cfg["sweep"]["freq"]
    pdr_cfg = cfg["sweep"]["gain"]
    fpts = sweep2array(freq_cfg)
    pdrs = sweep2array(pdr_cfg)

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    signals2D = np.full((len(pdrs), len(fpts)), np.nan, dtype=np.complex128)
    if instant_show:
        viewer = InstantShow(
            pdrs, fpts, x_label="Power (a.u.)", y_label="Frequency (MHz)"
        )

        def callback(ir, sum_d):
            nonlocal signals2D
            signals2D = sum_d[0][0].dot([1, 1j]) / (ir + 1)
            viewer.update_show(signals2D)
    else:
        callback = None

    try:
        prog = PowerDepProgram(soccfg, make_cfg(cfg))
        fpt_pdr, avgi, avgq = prog.acquire(soc, progress=True, callback=callback)
        signals2D = avgi[0][0] + 1j * avgq[0][0]
        fpts, pdrs = fpt_pdr[0], fpt_pdr[1]

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, early stopping the program")
    except Exception as e:
        print("Error during measurement:", e)
    finally:
        if instant_show:
            viewer.update_show(signals2D)
            viewer.close_show()

    return fpts, pdrs, signals2D  # (pdrs, freqs)
