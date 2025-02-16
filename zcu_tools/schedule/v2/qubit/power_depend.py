from zcu_tools import make_cfg
from zcu_tools.analysis import NormalizeData
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import close_show, init_show2d, update_show2d
from zcu_tools.schedule.tools import sweep2array, sweep2param

from .twotone import sweep_twotone


def measure_qub_pdr_dep(soc, soccfg, cfg, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    # make sure gain is the outer loop
    if list(cfg["sweep"].keys())[0] == "freq":
        cfg["sweep"] = {
            "gain": cfg["sweep"]["gain"],
            "freq": cfg["sweep"]["freq"],
        }

    qub_pulse = cfg["dac"]["qub_pulse"]
    qub_pulse["gain"] = sweep2param("gain", cfg["sweep"]["gain"])
    qub_pulse["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

    if instant_show:
        pdrs = sweep2array(cfg["sweep"]["gain"])
        fpts = sweep2array(cfg["sweep"]["freq"])
        fig, ax, dh, im = init_show2d(fpts, pdrs, "Frequency (MHz)", "Pulse Gain")

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    if instant_show:

        def callback(ir, sum_d):
            if instant_show:
                signals2D = sum_d[0][0].dot([1, 1j]) / (ir + 1)
                amps = NormalizeData(signals2D, axis=0, rescale=False)
                update_show2d(fig, ax, dh, im, amps.T)
    else:
        callback = None  # type: ignore

    pdrs, fpts, signals2D = sweep_twotone(  # type: ignore
        soc,
        soccfg,
        cfg,
        p_attr=["gain", "freq"],
        progress=True,
        callback=callback,
    )

    if instant_show:
        amps = NormalizeData(signals2D, axis=0, rescale=False)
        update_show2d(fig, ax, dh, im, amps.T, (fpts, pdrs))
        close_show(fig, dh)

    return pdrs, fpts, signals2D
