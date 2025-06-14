from typing import Tuple

import numpy as np
from tqdm.auto import tqdm
from zcu_tools.auto import make_cfg
from zcu_tools.liveplot.jupyter import LivePlotter1D
from zcu_tools.program.v2 import OneToneProgram, TwoToneProgram

from ..flux import set_flux


def onetone_demimated(
    soc, soccfg, cfg, progress=True, qub_pulse=False
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = make_cfg(cfg, reps=1)

    prog = TwoToneProgram(soccfg, cfg) if qub_pulse else OneToneProgram(soccfg, cfg)
    IQlist = prog.acquire_decimated(soc, progress=progress)

    Ts = prog.get_time_axis(ro_index=0)
    Ts += cfg["adc"]["trig_offset"]

    return Ts, IQlist[0].dot([1, 1j])


def measure_lookback(
    soc, soccfg, cfg, progress=True, qub_pulse=False
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = make_cfg(cfg)

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    MAX_LEN = 3.32  # us

    if cfg["adc"]["ro_length"] <= MAX_LEN:
        Ts, signals = onetone_demimated(
            soc, soccfg, cfg, progress=progress, qub_pulse=qub_pulse
        )
    else:
        # measure multiple times
        trig_offset = cfg["adc"]["trig_offset"]
        total_len = trig_offset + cfg["adc"]["ro_length"]
        cfg["adc"]["ro_length"] = MAX_LEN

        bar = tqdm(
            total=int((total_len - trig_offset) / MAX_LEN + 0.999),
            desc="Readout",
            smoothing=0,
            disable=not progress,
        )

        Ts = []
        signals = []
        with LivePlotter1D("Time (us)", "Amplitude", title="Readout") as viewer:
            while trig_offset < total_len:
                cfg["adc"]["trig_offset"] = trig_offset

                Ts_, singals_ = onetone_demimated(
                    soc, soccfg, cfg, progress=False, qub_pulse=qub_pulse
                )

                Ts.append(Ts_)
                signals.append(singals_)

                viewer.update(np.concatenate(Ts), np.concatenate(signals))

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
