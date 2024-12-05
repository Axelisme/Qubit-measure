from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program import TwoToneProgram


def measure_dispersive(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification
    sweep_cfg = cfg["sweep"]
    fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

    res_pulse = cfg["res_pulse"]
    qub_pulse = cfg["qub_pulse"]

    pi_gain = qub_pulse["gain"]

    if instant_show:
        import matplotlib.pyplot as plt
        from IPython.display import clear_output, display

        fig, ax = plt.subplots()
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Dispersive measurement")
        curve_g = ax.plot(fpts, np.zeros_like(fpts))[0]
        curve_e = ax.plot(fpts, np.zeros_like(fpts))[0]
        dh = display(fig, display_id=True)

    qub_pulse["gain"] = 0
    g_signals = np.full(len(fpts), np.nan, dtype=np.complex128)
    for i, f in enumerate(tqdm(fpts)):
        res_pulse["freq"] = f
        prog = TwoToneProgram(soccfg, make_cfg(cfg))
        avgi, avgq = prog.acquire(soc, progress=False)
        signal = avgi[0][0] + 1j * avgq[0][0]
        g_signals[i] = signal

        if instant_show:
            curve_g.set_ydata(np.abs(g_signals))
            ax.relim()
            ax.set_xlim(fpts[0], fpts[-1])
            ax.autoscale_view()
            dh.update(fig)

    qub_pulse["gain"] = pi_gain
    e_signals = np.full(len(fpts), np.nan, dtype=np.complex128)
    for i, f in enumerate(tqdm(fpts)):
        res_pulse["freq"] = f
        prog = TwoToneProgram(soccfg, make_cfg(cfg))
        avgi, avgq = prog.acquire(soc, progress=False)
        signal = avgi[0][0] + 1j * avgq[0][0]
        e_signals[i] = signal

        if instant_show:
            curve_e.set_ydata(np.abs(e_signals))
            ax.relim()
            ax.set_xlim(fpts[0], fpts[-1])
            ax.autoscale_view()
            dh.update(fig)

    if instant_show:
        clear_output()

    return fpts, g_signals, e_signals
