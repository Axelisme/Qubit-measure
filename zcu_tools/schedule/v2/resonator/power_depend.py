import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.analysis import NormalizeData
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import clear_show, init_show2d, update_show2d
from zcu_tools.schedule.tools import map2adcfreq, sweep2array
from zcu_tools.schedule.v2.resonator.onetone import sweep_onetone


def measure_res_pdr_dep(
    soc,
    soccfg,
    cfg,
    instant_show=False,
    dynamic_reps=False,
    gain_ref=0.1,
):
    cfg = make_cfg(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    reps_ref = cfg["reps"]

    pdrs = sweep2array(cfg["sweep"]["gain"])
    fpts = sweep2array(cfg["sweep"]["freq"])
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    del cfg["sweep"]["gain"]  # use for loop here

    if instant_show:
        fig, ax, dh, im = init_show2d(fpts, pdrs, "Frequency (MHz)", "Power (a.u.)")

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    signals2D = np.full((len(pdrs), len(fpts)), np.nan, dtype=np.complex128)
    try:
        pdr_tqdm = tqdm(pdrs, desc="Power", smoothing=0)
        avgs_tqdm = tqdm(total=cfg["soft_avgs"], desc="Soft_avgs", smoothing=0)
        for i, pdr in enumerate(pdr_tqdm):
            res_pulse["gain"] = pdr

            if dynamic_reps:
                cfg["reps"] = int(reps_ref * gain_ref / max(pdr, 1e-6))
                if cfg["reps"] < 0.1 * reps_ref:
                    cfg["reps"] = int(0.1 * reps_ref + 0.99)
                elif cfg["reps"] > 10 * reps_ref:
                    cfg["reps"] = int(10 * reps_ref)

            avgs_tqdm.reset()
            avgs_tqdm.refresh()

            _signals2D = signals2D.copy()  # prevent overwrite

            def callback(ir, sum_d, *, xs):
                avgs_tqdm.update(max(ir + 1 - avgs_tqdm.n, 0))
                avgs_tqdm.refresh()
                if instant_show:
                    _signals2D[i] = sum_d[0][0].dot([1, 1j]) / (ir + 1)
                    amps = NormalizeData(np.abs(_signals2D), axis=1)
                    update_show2d(fig, ax, dh, im, amps)

            fpts, signals2D[i] = sweep_onetone(
                soc,
                soccfg,
                cfg,
                loop="freq",
                p_attr="freq",
                progress=False,
                callback=callback,
            )

            avgs_tqdm.update(avgs_tqdm.total - avgs_tqdm.n)

            if instant_show:
                amps = NormalizeData(np.abs(signals2D), axis=1)
                update_show2d(fig, ax, dh, im, amps, (fpts, pdrs))

        pdr_tqdm.close()
        avgs_tqdm.close()

        if instant_show:
            clear_show(fig, dh)
    except BaseException as e:
        print("Error during measurement:", e)

    return fpts, pdrs, signals2D  # (pdrs, freqs)
