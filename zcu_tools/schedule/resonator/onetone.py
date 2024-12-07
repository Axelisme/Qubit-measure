from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program import OneToneProgram

from ..flux import set_flux


def measure_res_freq(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    set_flux(cfg["flux_dev"], cfg["flux"])

    sweep_cfg = cfg["sweep"]
    fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    if instant_show:
        import matplotlib.pyplot as plt
        from IPython.display import clear_output, display

        fig, ax = plt.subplots()
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Resonator frequency measurement")
        curve = ax.plot(fpts, np.zeros_like(fpts))[0]
        dh = display(fig, display_id=True)

    res_pulse = cfg["dac"]["res_pulse"]

    signals = np.full(len(fpts), np.nan, dtype=np.complex128)
    for i, fpt in enumerate(tqdm(fpts)):
        res_pulse["freq"] = fpt
        prog = OneToneProgram(soccfg, make_cfg(cfg))
        avgi, avgq = prog.acquire(soc, progress=False)
        signals[i] = avgi[0][0] + 1j * avgq[0][0]

        if instant_show:
            curve.set_data(fpts, np.abs(signals))
            ax.relim()
            ax.autoscale(axis="y")
            dh.update(fig)

    if instant_show:
        clear_output()

    return fpts, signals
