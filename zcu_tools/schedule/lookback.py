from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program import OneToneProgram, TwoToneProgram

from .flux import set_flux
from .instant_show import clear_show, init_show, update_show


def measure_one(soc, soccfg, cfg, progress, qub_pulse):
    if qub_pulse:
        prog = TwoToneProgram(soccfg, make_cfg(cfg, reps=1))
    else:
        prog = OneToneProgram(soccfg, make_cfg(cfg, reps=1))
    IQlist = prog.acquire_decimated(soc, progress=progress)
    Is, Qs = IQlist[0]

    Ts = soccfg.cycles2us(np.arange(len(Is)), ro_ch=cfg["adc"]["chs"][0])
    Ts += cfg["adc"]["trig_offset"]

    return Ts, np.array(Is), np.array(Qs)


def measure_lookback(
    soc, soccfg, cfg, progress=True, instant_show=False, qub_pulse=False
):
    cfg = deepcopy(cfg)  # prevent in-place modification
    assert cfg.get("reps", 1) == 1, "Only one rep is allowed for lookback"

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    MAX_LEN = 3.32  # us

    if cfg["adc"]["ro_length"] <= MAX_LEN:
        Ts, Is, Qs = measure_one(
            soc, soccfg, cfg, progress=progress, qub_pulse=qub_pulse
        )
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
        Is, Qs = [], []
        while trig_offset < total_len:
            cfg["adc"]["trig_offset"] = trig_offset

            Ts_, Is_, Qs_ = measure_one(
                soc, soccfg, cfg, progress=False, qub_pulse=qub_pulse
            )

            Ts.append(Ts_)
            Is.append(Is_)
            Qs.append(Qs_)

            if instant_show:
                _Ts = np.concatenate(Ts)
                _Is = np.concatenate(Is)
                _Qs = np.concatenate(Qs)
                update_show(fig, ax, dh, curve, np.abs(_Is + 1j * _Qs), _Ts)

            trig_offset += MAX_LEN
            bar.update()

        bar.close()
        Ts = np.concatenate(Ts)
        Is = np.concatenate(Is)
        Qs = np.concatenate(Qs)

        sort_idxs = np.argsort(Ts, kind="stable")
        Ts = Ts[sort_idxs]
        Is = Is[sort_idxs]
        Qs = Qs[sort_idxs]

        if instant_show:
            update_show(fig, ax, dh, curve, np.abs(Is + 1j * Qs), Ts)
            clear_show()

    return Ts, Is, Qs
