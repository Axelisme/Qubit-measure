import numpy as np

from zcu_tools import make_cfg
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.instant_show import clear_show, init_show, update_show
from zcu_tools.schedule.tools import format_sweep1D, sweep2array, sweep2param


def sweep_twotone(soc, soccfg, cfg, p_attr, progress=True, callback=None, **kwargs):
    prog = TwoToneProgram(soccfg, cfg)

    IQlist = prog.acquire(soc, progress=progress, round_callback=callback, **kwargs)
    signals = IQlist[0][0].dot([1, 1j])

    if isinstance(p_attr, str):
        xs = prog.get_pulse_param("qub_pulse", p_attr, as_array=True)
        return xs, signals
    elif isinstance(p_attr, (list, tuple)):
        xss = [prog.get_pulse_param("qub_pulse", r, as_array=True) for r in p_attr]
        return *xss, signals
    else:
        raise ValueError(f"Invalid p_attr: {p_attr}")


def measure_qub_freq(soc, soccfg, cfg, progress=True, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

    sweep_cfg = cfg["sweep"]["freq"]
    cfg["dac"]["qub_pulse"]["freq"] = sweep2param("freq", sweep_cfg)

    if instant_show:
        # predict fpts
        fpts = sweep2array(sweep_cfg, False)
        fig, ax, dh, curve = init_show(fpts, "Frequency (MHz)", "Amplitude")

        def callback(ir, sum_d):
            amps = np.abs(sum_d[0][0].dot([1, 1j]) / (ir + 1))
            update_show(fig, ax, dh, curve, amps)
    else:
        callback = None  # type: ignore

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    fpts, signals = sweep_twotone(
        soc,
        soccfg,
        cfg,
        p_attr="freq",
        progress=progress,
        callback=callback,
    )

    if instant_show:
        update_show(fig, ax, dh, curve, np.abs(signals), fpts)
        clear_show(fig, dh)

    return fpts, signals


def measure_qub_freq_with_reset(
    soc, soccfg, cfg, r_f, progress=True, instant_show=False
):
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

    assert cfg["dac"].get("reset") == "pulse", "Need reset=pulse for conjugate reset"
    assert "reset_pulse" in cfg["dac"], "Need reset_pulse for conjugate reset"

    sweep_cfg = cfg["sweep"]["freq"]
    params = sweep2param("freq", sweep_cfg)
    cfg["dac"]["qub_pulse"]["freq"] = params
    cfg["dac"]["reset_pulse"]["freq"] = r_f - params  # conjugate reset

    if instant_show:
        # predict fpts
        fpts = sweep2array(cfg["sweep"]["freq"], False)
        fig, ax, dh, curve = init_show(fpts, "Frequency (MHz)", "Amplitude")

        def callback(ir, sum_d):
            amps = np.abs(sum_d[0][0].dot([1, 1j]) / (ir + 1))
            update_show(fig, ax, dh, curve, amps)
    else:
        callback = None  # type: ignore

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    fpts, signals = sweep_twotone(
        soc,
        soccfg,
        cfg,
        p_attr="freq",
        progress=progress,
        callback=callback,
    )

    if instant_show:
        update_show(fig, ax, dh, curve, np.abs(signals), fpts)
        clear_show(fig, dh)

    return fpts, signals
