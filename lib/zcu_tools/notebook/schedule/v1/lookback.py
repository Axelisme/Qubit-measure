from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm
from zcu_tools import make_cfg
from zcu_tools.program.v1 import OneToneProgram, TwoToneProgram

from ..flux import set_flux

from zcu_tools.liveplot.jupyter import LivePlotter1D


def measure_one(soc, soccfg, cfg, progress, qub_pulse):
    if qub_pulse:
        prog = TwoToneProgram(soccfg, make_cfg(cfg, reps=1))
    else:
        prog = OneToneProgram(soccfg, make_cfg(cfg, reps=1))
    IQlist = prog.acquire_decimated(soc, progress=progress)
    Is, Qs = IQlist[0]
    signals = np.array(Is) + 1j * np.array(Qs)

    Ts = prog.get_time_axis(ro_index=0)
    Ts += cfg["adc"]["trig_offset"]

    return Ts, signals


def measure_lookback(soc, soccfg, cfg, progress=True, qub_pulse=False):
    cfg = deepcopy(cfg)  # prevent in-place modification
    assert cfg.get("reps", 1) == 1, "Only one rep is allowed for lookback"

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    MAX_LEN = 3.32  # us

    if cfg["adc"]["ro_length"] <= MAX_LEN:
        Ts, signals = measure_one(
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

        with LivePlotter1D("Time (us)", "Amplitude") as viewer:
            Ts = []
            signals = []
            while trig_offset < total_len:
                cfg["adc"]["trig_offset"] = trig_offset

                Ts_, signals_ = measure_one(
                    soc, soccfg, cfg, progress=False, qub_pulse=qub_pulse
                )

                Ts.append(Ts_)
                signals.append(signals_)

                viewer.update(np.concatenate(Ts), np.abs(np.concatenate(signals)))

                trig_offset += MAX_LEN
                bar.update()

            bar.close()
            Ts = np.concatenate(Ts)
            signals = np.concatenate(signals)

            sort_idxs = np.argsort(Ts, kind="stable")
            Ts = Ts[sort_idxs]
            signals = signals[sort_idxs]

            viewer.update(Ts, np.abs(signals))

    return Ts, signals
