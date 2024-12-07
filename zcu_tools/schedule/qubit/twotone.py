from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program import TwoToneProgram, RFreqTwoToneProgram


def measure_qub_freq(soc, soccfg, cfg, instant_show=False, soft_loop=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    if soft_loop:
        print("Use TwoToneProgram for soft loop")

        sweep_cfg = cfg["sweep"]
        fpts = np.linspace(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["expts"])

        if instant_show:
            import matplotlib.pyplot as plt
            from IPython.display import clear_output, display

            fig, ax = plt.subplots()
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Qubit frequency measurement")
            curve = ax.plot(fpts, np.zeros_like(fpts))[0]
            dh = display(fig, display_id=True)

        qub_pulse = cfg["qub_pulse"]
        signals = np.full(len(fpts), np.nan, dtype=np.complex128)
        for i, fpt in enumerate(tqdm(fpts)):
            qub_pulse["freq"] = fpt
            prog = TwoToneProgram(soccfg, make_cfg(cfg))
            avgi, avgq = prog.acquire(soc, progress=False)
            signals[i] = avgi[0][0] + 1j * avgq[0][0]

            if instant_show:
                curve.set_ydata(np.abs(signals))
                ax.relim()
                ax.autoscale(axis="y")
                dh.update(fig)

        if instant_show:
            clear_output()
    else:
        print("Use RFreqTwoToneProgram for hard loop")

        if instant_show:
            print("Instant show is not supported for hard loop, ignored.")

        prog = RFreqTwoToneProgram(soccfg, make_cfg(cfg))
        fpts, avgi, avgq = prog.acquire(soc, progress=True)
        signals = avgi[0][0] + 1j * avgq[0][0]

    return fpts, signals
