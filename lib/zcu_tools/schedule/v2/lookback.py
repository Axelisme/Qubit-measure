from typing import Tuple

import numpy as np
from tqdm.auto import tqdm
from zcu_tools import make_cfg
from zcu_tools.program.v2 import OneToneProgram, TwoToneProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import InstantShow1D


def onetone_demimated(soc, soccfg, cfg, progress=True, qub_pulse=False):
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

        total_num = soccfg.us2cycles(total_len, ro_ch=cfg["adc"]["chs"][0])
        viewer = InstantShow1D(
            np.linspace(0, total_len, total_num, endpoint=False),
            x_label="Time (us)",
            y_label="Amplitude",
            title="Readout",
            linestyle="-",
            marker=None,
        )

        Ts = []
        signals = []
        while trig_offset < total_len:
            cfg["adc"]["trig_offset"] = trig_offset

            Ts_, singals_ = onetone_demimated(
                soc, soccfg, cfg, progress=False, qub_pulse=qub_pulse
            )

            Ts.append(Ts_)
            signals.append(singals_)

            viewer.update_show(
                np.abs(np.concatenate(signals)), ticks=np.concatenate(Ts)
            )

            trig_offset += MAX_LEN
            bar.update()

        bar.close()
        Ts = np.concatenate(Ts)
        signals = np.concatenate(signals)

        sort_idxs = np.argsort(Ts, kind="stable")
        Ts = Ts[sort_idxs]
        signals = signals[sort_idxs]

        viewer.update_show(np.abs(signals), ticks=Ts)
        viewer.close_show()

    return Ts, signals
