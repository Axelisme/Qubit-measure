from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.analysis import NormalizeData
from zcu_tools.program.v1 import RFreqTwoToneProgram, RFreqTwoToneProgramWithRedReset
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import close_show, init_show2d, update_show2d
from zcu_tools.schedule.tools import sweep2array


def measure_qub_flux_dep(soc, soccfg, cfg, instant_show=False, reset_rf=None):
    cfg = deepcopy(cfg)  # prevent in-place modification

    if reset_rf is not None:
        cfg["r_f"] = reset_rf
        assert cfg.get("reset") == "pulse", "Need reset=pulse for conjugate reset"
        assert "reset_pulse" in cfg["dac"], "Need reset_pulse for conjugate reset"

    if cfg["dev"]["flux_dev"] == "none":
        raise ValueError("Flux sweep but get flux_dev == 'none'")

    freq_cfg = cfg["sweep"]["freq"]
    flux_cfg = cfg["sweep"]["flux"]
    fpts = sweep2array(freq_cfg, soft_loop=False)
    flxs = sweep2array(flux_cfg)

    cfg["sweep"] = cfg["sweep"]["freq"]

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
                    amps = NormalizeData(_signals2D, axis=1, rescale=True) ** 1.5
                    update_show2d(fig, ax, dh, im, amps.T)

            prog_cls = (
                RFreqTwoToneProgram
                if reset_rf is None
                else RFreqTwoToneProgramWithRedReset
            )
            prog = prog_cls(soccfg, make_cfg(cfg))
            fpts, avgi, avgq = prog.acquire(
                soc, progress=False, round_callback=callback
            )
            signals2D[i] = avgi[0][0] + 1j * avgq[0][0]

            avgs_tqdm.update(avgs_tqdm.total - avgs_tqdm.n)
            avgs_tqdm.refresh()

            if instant_show:
                amps = NormalizeData(signals2D, axis=1, rescale=True) ** 1.5
                update_show2d(fig, ax, dh, im, amps.T, (flxs, fpts))
        else:
            if instant_show:
                close_show(fig, dh)

        flux_tqdm.close()
        avgs_tqdm.close()

    except BaseException as e:
        print("Error during measurement:", e)

    return flxs, fpts, signals2D
