import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program.v2 import OneToneProgram, TwoToneProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import close_show, init_show1d, update_show1d


def onetone_demimated(soc, soccfg, cfg, progress=True, qub_pulse=False):
    cfg = make_cfg(cfg, reps=1)

    prog = TwoToneProgram(soccfg, cfg) if qub_pulse else OneToneProgram(soccfg, cfg)
    IQlist = prog.acquire_decimated(soc, progress=progress)

    Ts = soccfg.cycles2us(np.arange(len(IQlist[0])), ro_ch=cfg["adc"]["chs"][0])
    Ts += cfg["adc"]["trig_offset"]

    return Ts, IQlist[0].dot([1, 1j])


def measure_lookback(
    soc, soccfg, cfg, progress=True, instant_show=False, qub_pulse=False
):
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
            total=int(total_len / MAX_LEN + 0.999),
            desc="Readout",
            smoothing=0,
            disable=not progress,
        )

        if instant_show:
            total_num = soccfg.us2cycles(total_len, ro_ch=cfg["adc"]["chs"][0])
            fig, ax, dh, curve = init_show1d(
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

            Ts_, singals_ = onetone_demimated(
                soc, soccfg, cfg, progress=False, qub_pulse=qub_pulse
            )

            Ts.append(Ts_)
            signals.append(singals_)

            if instant_show:
                update_show1d(
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
            update_show1d(fig, ax, dh, curve, np.abs(signals), Ts)
            close_show(fig, dh)

    return Ts, signals
