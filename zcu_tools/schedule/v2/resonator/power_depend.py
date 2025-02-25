import numpy as np
from tqdm.auto import tqdm

from zcu_tools import make_cfg
from zcu_tools.program.v2 import OneToneProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import InstantShow
from zcu_tools.schedule.tools import map2adcfreq, sweep2array, sweep2param


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
    res_pulse["freq"] = sweep2param("freq", cfg["sweep"]["freq"])
    reps_ref = cfg["reps"]

    pdrs = sweep2array(cfg["sweep"]["gain"], allow_array=True)
    fpts = sweep2array(cfg["sweep"]["freq"])
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    del cfg["sweep"]["gain"]  # use for loop here

    pdr_tqdm = tqdm(pdrs, desc="Power", smoothing=0)
    avgs_tqdm = tqdm(total=cfg["soft_avgs"], desc="Soft_avgs", smoothing=0)
    if instant_show:
        viewer = InstantShow(
            fpts, pdrs, x_label="Frequency (MHz)", y_label="Power (a.u.)"
        )

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    signals2D = np.full((len(pdrs), len(fpts)), np.nan, dtype=np.complex128)
    try:
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

            def callback(ir, sum_d):
                avgs_tqdm.update(max(ir + 1 - avgs_tqdm.n, 0))
                avgs_tqdm.refresh()
                if instant_show:
                    _signals2D[i] = sum_d[0][0].dot([1, 1j]) / (ir + 1)
                    viewer.update_show(_signals2D.T)

            prog = OneToneProgram(soccfg, cfg)
            IQlist = prog.acquire(soc, progress=False, round_callback=callback)

            fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)
            signals2D[i] = IQlist[0][0].dot([1, 1j])  # type: ignore

            avgs_tqdm.update(avgs_tqdm.total - avgs_tqdm.n)
            avgs_tqdm.refresh()

            if instant_show:
                viewer.update_show(signals2D.T, (fpts, pdrs))

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, early stopping the program")
    except Exception as e:
        print("Error during measurement:", e)
    finally:
        if instant_show:
            viewer.close_show()
        pdr_tqdm.close()
        avgs_tqdm.close()

    return pdrs, fpts, signals2D  # (pdrs, freqs)
