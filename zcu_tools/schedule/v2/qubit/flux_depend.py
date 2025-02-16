import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.analysis import NormalizeData
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import close_show, init_show2d, update_show2d
from zcu_tools.schedule.tools import sweep2array, sweep2param


def measure_qub_flux_dep(soc, soccfg, cfg, instant_show=False, reset_rf=None):
    cfg = make_cfg(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]
    qub_pulse["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

    if reset_rf is not None:
        assert cfg["dac"]["reset"] == "pulse", "Need reset=pulse for conjugate reset"
        assert "reset_pulse" in cfg["dac"], "Need reset_pulse for conjugate reset"
        cfg["dac"]["reset_pulse"]["freq"] = reset_rf - qub_pulse["freq"]

    if cfg["dev"]["flux_dev"] == "none":
        raise ValueError("Flux sweep but get flux_dev == 'none'")

    flxs = sweep2array(cfg["sweep"]["flux"])
    fpts = sweep2array(cfg["sweep"]["freq"])

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
                    amps = NormalizeData(_signals2D, axis=1) ** 1.5
                    update_show2d(fig, ax, dh, im, amps)

            prog = TwoToneProgram(soccfg, cfg)

            IQlist = prog.acquire(soc, progress=False, round_callback=callback)
            signals2D[i] = IQlist[0][0].dot([1, 1j])
            print(np.nanmax(np.abs(signals2D)))

            avgs_tqdm.update(avgs_tqdm.total - avgs_tqdm.n)
            avgs_tqdm.refresh()

            if instant_show:
                amps = NormalizeData(signals2D, axis=1) ** 1.5
                update_show2d(fig, ax, dh, im, amps, (flxs, fpts))
        else:
            if instant_show:
                close_show(fig, dh)

        flux_tqdm.close()
        avgs_tqdm.close()

    except BaseException as e:
        if instant_show:
            close_show(fig, dh)
        print("Error during measurement:", e)

    return flxs, fpts, signals2D
