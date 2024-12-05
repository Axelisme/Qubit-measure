from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools.auto import make_cfg
from zcu_tools.program import AmpRabiProgram, TwoToneProgram


def measure_lenrabi(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)

    sweep_cfg = cfg["sweep"]
    lens = np.arange(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["step"])

    if instant_show:
        import matplotlib.pyplot as plt
        from IPython.display import clear_output, display

        fig, ax = plt.subplots()
        ax.set_xlabel("Length (ns)")
        ax.set_ylabel("Signal (a.u.)")
        ax.set_title("Length-dependent measurement")
        curve = ax.plot(lens, np.zeros_like(lens))[0]
        dh = display(fig, display_id=True)

    qub_pulse = cfg["qub_pulse"]

    signals = np.full(len(lens), np.nan, dtype=np.complex128)
    for i, length in enumerate(tqdm(lens)):
        qub_pulse["length"] = length
        prog = TwoToneProgram(soccfg, make_cfg(cfg))
        avgi, avgq = prog.acquire(soc, progress=False)
        signals[i] = avgi[0][0] + 1j * avgq[0][0]

        if instant_show:
            curve.set_ydata(np.abs(signals))
            ax.relim()
            ax.set_xlim(lens[0], lens[-1])
            ax.autoscale_view()
            dh.update(fig)

    if instant_show:
        clear_output()

    return lens, signals


def measure_amprabi(soc, soccfg, cfg, instant_show=False, soft_loop=False):
    cfg = deepcopy(cfg)

    if soft_loop:
        print("Use TwoToneProgram for soft loop")

        sweep_cfg = cfg["sweep"]
        pdrs = np.arange(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["step"])

        qub_pulse = cfg["qub_pulse"]

        if instant_show:
            import matplotlib.pyplot as plt
            from IPython.display import clear_output, display

            fig, ax = plt.subplots()
            ax.set_xlabel("Power (a.u.)")
            ax.set_ylabel("Signal (a.u.)")
            ax.set_title("Power-dependent measurement")
            curve = ax.plot(pdrs, np.zeros_like(pdrs))[0]
            dh = display(fig, display_id=True)

        signals = np.full(len(pdrs), np.nan, dtype=np.complex128)
        for i, pdr in enumerate(tqdm(pdrs)):
            qub_pulse["gain"] = pdr
            prog = TwoToneProgram(soccfg, make_cfg(cfg))
            avgi, avgq = prog.acquire(soc, progress=False)
            signals[i] = avgi[0][0] + 1j * avgq[0][0]

            if instant_show:
                curve.set_ydata(np.abs(signals))
                ax.relim()
                ax.set_xlim(pdrs[0], pdrs[-1])
                ax.autoscale_view()
                dh.update(fig)

        if instant_show:
            clear_output()

    else:
        print("Use AmpRabiProgram for hard loop")

        if instant_show:
            print("Instant show is not supported for hard loop, ignored.")

        prog = AmpRabiProgram(soccfg, cfg)
        pdrs, avgi, avgq = prog.acquire(soc, progress=True)
        signals = avgi[0][0] + 1j * avgq[0][0]

    return pdrs, signals
