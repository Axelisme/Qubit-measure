import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.analysis import NormalizeData
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import clear_show, init_show2d, update_show2d
from zcu_tools.schedule.tools import map2adcfreq, sweep2array, sweep2param
from zcu_tools.schedule.v2.resonator.onetone import sweep_onetone


def measure_res_flux_dep(soc, soccfg, cfg, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    res_pulse["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

    if cfg["dev"]["flux_dev"] == "none":
        raise ValueError("Flux sweep but get flux_dev == 'none'")

    flxs = sweep2array(cfg["sweep"]["flux"])
    fpts = sweep2array(cfg["sweep"]["freq"])
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    del cfg["sweep"]["flux"]  # use for loop here

    set_flux(cfg["dev"]["flux_dev"], flxs[0])  # set initial flux

    flux_tqdm = tqdm(flxs, desc="Flux", smoothing=0)
    avgs_tqdm = tqdm(total=cfg["soft_avgs"], desc="Soft_avgs", smoothing=0)
    if instant_show:
        fig, ax, dh, im = init_show2d(flxs, fpts, "Flux", "Frequency (MHz)")

    signals2D = np.full((len(flxs), len(fpts)), np.nan, dtype=np.complex128)
    try:
        for i, flx in enumerate(flux_tqdm):
            cfg["dev"]["flux"] = flx
            set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

            avgs_tqdm.reset()
            avgs_tqdm.refresh()

            _signals2D = signals2D.copy()  # prevent overwrite

            def callback(ir, sum_d):
                avgs_tqdm.update(max(ir + 1 - avgs_tqdm.n, 0))
                avgs_tqdm.refresh()
                if instant_show:
                    _signals2D[i] = sum_d[0][0].dot([1, 1j]) / (ir + 1)
                    amps = NormalizeData(np.abs(_signals2D), axis=1, rescale=False)
                    update_show2d(fig, ax, dh, im, amps.T)

            fpts, signals2D[i] = sweep_onetone(
                soc,
                soccfg,
                cfg,
                p_attr="freq",
                progress=False,
                callback=callback,
            )

            avgs_tqdm.update(avgs_tqdm.total - avgs_tqdm.n)
            avgs_tqdm.refresh()

            if instant_show:
                amps = NormalizeData(np.abs(signals2D), axis=1, rescale=False)
                update_show2d(fig, ax, dh, im, amps.T, (flxs, fpts))
        else:
            if instant_show:
                clear_show(fig, dh)

        flux_tqdm.close()
        avgs_tqdm.close()

    except BaseException as e:
        print("Error during measurement:", e)

    return flxs, fpts, signals2D
