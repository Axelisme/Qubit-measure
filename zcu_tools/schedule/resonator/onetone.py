from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program import OneToneProgram


def measure_res_freq(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    sweep_cfg = cfg["sweep"]
    fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    signals = np.zeros(len(fpts), dtype=np.complex128)

    if instant_show:
        import matplotlib.pyplot as plt
        from IPython.display import clear_output, display

        fig, ax = plt.subplots()
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Resonator frequency measurement")
        curve = ax.plot(fpts, np.ma.masked_all_like(fpts))[0]
        dh = display(fig, display_id=True)

    res_pulse = cfg["res_pulse"]
    for i, fpt in enumerate(tqdm(fpts)):
        res_pulse["freq"] = fpt
        prog = OneToneProgram(soccfg, make_cfg(cfg))
        avgi, avgq = prog.acquire(soc, progress=False)
        signals[i] = avgi[0][0] + 1j * avgq[0][0]

        if instant_show:
            amps = np.ma.masked_equal(np.abs(signals), 0.0, copy=False)
            curve.set_ydata(amps)
            dh.update(fig)

    if instant_show:
        clear_output()

    return fpts, signals
