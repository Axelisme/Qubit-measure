from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program import OneToneProgram, TwoToneProgram

from .flux import set_flux


def measure_one(soc, soccfg, cfg, progress, qub_pulse):
    if qub_pulse:
        prog = TwoToneProgram(soccfg, make_cfg(cfg, reps=1))
    else:
        prog = OneToneProgram(soccfg, make_cfg(cfg, reps=1))
    IQlist = prog.acquire_decimated(soc, progress=progress)
    Is, Qs = IQlist[0]
    Ts = prog.cycles2us(np.arange(len(Is)), ro_ch=cfg["adc"]["chs"][0])
    Ts += cfg["adc"]["trig_offset"]
    return Ts, Is, Qs


def measure_lookback(soc, soccfg, cfg, qub_pulse=False):
    cfg = deepcopy(cfg)  # prevent in-place modification
    assert cfg.get("reps", 1) == 1, "Only one rep is allowed for lookback"

    set_flux(cfg["flux_dev"], cfg["flux"])

    MAX_LEN = 3.32  # us

    if cfg["adc"]["ro_length"] <= MAX_LEN:
        Ts, Is, Qs = measure_one(soc, soccfg, cfg, True, qub_pulse)
    else:
        # measure multiple times
        trig_offset = cfg["adc"]["trig_offset"]
        total_len = trig_offset + cfg["adc"]["ro_length"]
        cfg["adc"]["ro_length"] = MAX_LEN

        bar = tqdm(total=int(total_len / MAX_LEN + 0.999), desc="Readout", smoothing=0)

        Ts = []
        Is, Qs = [], []
        while trig_offset < total_len:
            cfg["adc"]["trig_offset"] = trig_offset

            Ts_, Is_, Qs_ = measure_one(soc, soccfg, cfg, False, qub_pulse)

            Ts.append(Ts_)
            Is.append(Is_)
            Qs.append(Qs_)

            trig_offset += MAX_LEN
            bar.update()
        bar.close()
        Ts = np.concatenate(Ts)
        Is = np.concatenate(Is)
        Qs = np.concatenate(Qs)

        sort_idxs = np.argsort(Ts)
        Ts = Ts[sort_idxs]
        Is = Is[sort_idxs]
        Qs = Qs[sort_idxs]

    return Ts, Is, Qs
