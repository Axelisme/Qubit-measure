from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program_v2 import OneToneProgram

from .flux import set_flux
from .instant_show import clear_show, init_show, update_show


def measure_one(soc, soccfg, cfg, progress):
    cfg = make_cfg(cfg, reps=1)
    prog = OneToneProgram(soccfg, cfg)
    IQlist = prog.acquire_decimated(soc, progress=progress, soft_avgs=cfg["soft_avgs"])

    Ts = soccfg.cycles2us(np.arange(len(IQlist[0])), ro_ch=cfg["adc"]["chs"][0])
    Ts += cfg["adc"]["trig_offset"]

    return Ts, IQlist[0].dot([1, 1j])


def measure_lookback(soc, soccfg, cfg, progress=True, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification
    assert cfg.get("reps", 1) == 1, "Only one rep is allowed for lookback"

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    MAX_LEN = 3.32  # us

    if cfg["adc"]["ro_length"] <= MAX_LEN:
        Ts, signals = measure_one(soc, soccfg, cfg, progress=progress)
    else:
        # measure multiple times
        trig_offset = cfg["adc"]["trig_offset"]
        total_len = trig_offset + cfg["adc"]["ro_length"]
        cfg["adc"]["ro_length"] = MAX_LEN

        bar = tqdm(
            total=int(total_len / MAX_LEN + 0.999),
            desc="Readout",
            smoothing=0,
            disable=not progress,
        )

        if instant_show:
            total_num = soccfg.us2cycles(total_len, ro_ch=cfg["adc"]["chs"][0])
            fig, ax, dh, curve = init_show(
                np.linspace(0, total_len, total_num, endpoint=False),
                "Time (us)",
                "Amplitude",
                linestyle="-",
                marker=None,
            )

        Ts = []
        signals = []
        while trig_offset < total_len:
            cfg["adc"]["trig_offset"] = trig_offset

            Ts_, singals_ = measure_one(soc, soccfg, cfg, progress=False)

            Ts.append(Ts_)
            signals.append(singals_)

            if instant_show:
                update_show(
                    fig,
                    ax,
                    dh,
                    curve,
                    np.abs(np.concatenate(signals)),
                    np.concatenate(Ts),
                )

            trig_offset += MAX_LEN
            bar.update()

        bar.close()
        Ts = np.concatenate(Ts)
        signals = np.concatenate(signals)

        sort_idxs = np.argsort(Ts, kind="stable")
        Ts = Ts[sort_idxs]
        signals = signals[sort_idxs]

        if instant_show:
            update_show(fig, ax, dh, curve, np.abs(signals), Ts)
            clear_show()

    return Ts, signals
