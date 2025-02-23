import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.analysis import NormalizeData
from zcu_tools.program.v2 import OneToneProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import InstantShow
from zcu_tools.schedule.tools import map2adcfreq, sweep2array, sweep2param


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
        viewer = InstantShow(
            flxs, fpts, x_label="Flux (a.u.)", y_label="Frequency (MHz)"
        )

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
                    viewer.update_show(amps)

            prog = OneToneProgram(soccfg, cfg)
            IQlist = prog.acquire(soc, progress=False, round_callback=callback)

            fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)
            signals2D[i] = IQlist[0][0].dot([1, 1j])  # type: ignore

            avgs_tqdm.update(avgs_tqdm.total - avgs_tqdm.n)
            avgs_tqdm.refresh()

            if instant_show:
                amps = NormalizeData(np.abs(signals2D), axis=1, rescale=False)
                viewer.update_show(amps, (flxs, fpts))

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, early stopping the program")
    except Exception as e:
        print("Error during measurement:", e)
    finally:
        if instant_show:
            amps = NormalizeData(np.abs(signals2D), axis=1, rescale=False)
            viewer.update_show(amps, (flxs, fpts))
            viewer.close_show()
        flux_tqdm.close()
        avgs_tqdm.close()

    return flxs, fpts, signals2D
