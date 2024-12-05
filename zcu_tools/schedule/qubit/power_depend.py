from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.analysis import NormalizeData
from zcu_tools.program import TwoToneProgram, RGainTwoToneProgram


def measure_qub_pdr_dep(soc, soccfg, cfg, instant_show=False, soft_loop=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    freq_cfg = cfg["sweep"]["freq"]
    pdr_cfg = cfg["sweep"]["pdr"]
    fpts = np.linspace(freq_cfg["start"], freq_cfg["stop"], freq_cfg["expts"])
    pdrs = np.arange(pdr_cfg["start"], pdr_cfg["stop"], pdr_cfg["step"])

    qub_pulse = cfg["qub_pulse"]

    freq_tqdm = tqdm(fpts)
    if soft_loop:
        # print("Use soft loop")
        print("Use TwoToneProgram for soft loop")
        pdr_tqdm = tqdm(pdrs)
    else:
        print("Use RGainTwoToneProgram for hard loop")
        cfg["sweep"] = pdr_cfg
    signals2D = np.zeros((len(pdrs), len(fpts)), dtype=np.complex128)
    if instant_show:
        import matplotlib.pyplot as plt
        from IPython.display import clear_output, display

        fig, ax = plt.subplots()
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Power (a.u.)")
        ax.set_title("Power-dependent measurement")
        ax.pcolormesh(fpts, pdrs, np.abs(signals2D))
        dh = display(fig, display_id=True)

    for i, fpt in enumerate(fpts):
        qub_pulse["freq"] = fpt

        if soft_loop:
            pdr_tqdm.reset()
            pdr_tqdm.refresh()
            for j, pdr in enumerate(pdrs):
                qub_pulse["gain"] = pdr
                prog = TwoToneProgram(soccfg, make_cfg(cfg))
                avgi, avgq = prog.acquire(soc, progress=False)
                signals2D[j, i] = avgi[0][0] + 1j * avgq[0][0]
                pdr_tqdm.update()
        else:
            qub_pulse["gain"] = pdrs[0]  # initial gain
            prog = RGainTwoToneProgram(soccfg, make_cfg(cfg))
            pdrs, avgi, avgq = prog.acquire(soc, progress=False)
            signals2D[:, i] = avgi[0][0] + 1j * avgq[0][0]

        freq_tqdm.update()

        if instant_show:
            amps = np.abs(signals2D)
            amps = NormalizeData(np.ma.masked_where(amps == 0, amps))
            ax.pcolormesh(fpts, pdrs, amps)
            dh.update(fig)
    if instant_show:
        clear_output()

    return fpts, pdrs, signals2D  # (pdrs, freqs)
