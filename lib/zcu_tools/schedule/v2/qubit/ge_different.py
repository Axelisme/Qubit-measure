import numpy as np
from zcu_tools import make_cfg
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.tools import (
    format_sweep1D,
    map2adcfreq,
    sweep2array,
    sweep2param,
)
from zcu_tools.schedule.v1.template import sweep2D_maximize_template
from zcu_tools.schedule.v2.template import sweep1D_soft_template, sweep_hard_template


def calc_snr(avg_d, std_d):
    contrast = avg_d[..., 1] - avg_d[..., 0]  # (*sweep)
    noise2_i = np.sum(std_d.real**2, axis=-1)  # (*sweep)
    noise2_q = np.sum(std_d.imag**2, axis=-1)  # (*sweep)
    noise = np.sqrt(noise2_i * contrast.real**2 + noise2_q * contrast.imag**2) / np.abs(
        contrast
    )

    return contrast / noise


def ge_result2signals(avg_d, std_d):
    avg_d = avg_d[0][0].dot([1, 1j])  # (*sweep, ge)
    std_d = std_d[0][0].dot([1, 1j])  # (*sweep, ge)

    return calc_snr(avg_d, std_d).T, None


def measure_ge_pdr_dep(soc, soccfg, cfg):
    cfg = make_cfg(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    qub_pulse = cfg["dac"]["qub_pulse"]

    # make sure gain is the outer loop
    if list(cfg["sweep"].keys())[0] == "freq":
        cfg["sweep"] = {
            "gain": cfg["sweep"]["gain"],
            "freq": cfg["sweep"]["freq"],
        }

    # append ge sweep to inner loop
    cfg["sweep"]["ge"] = {"start": 0, "stop": qub_pulse["gain"], "expts": 2}

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    res_pulse["gain"] = sweep2param("gain", cfg["sweep"]["gain"])
    res_pulse["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

    pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted pulse gains
    fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points
    fpts = map2adcfreq(soc, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    prog, snr2D = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        ticks=(fpts, pdrs),
        xlabel="Frequency (MHz)",
        ylabel="Readout Gain",
        result2signals=ge_result2signals,
    )

    # get the actual pulse gains and frequency points
    pdrs = prog.get_pulse_param("res_pulse", "gain", as_array=True)
    fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    return pdrs, fpts, snr2D  # (pdrs, fpts)


def measure_ge_pdr_dep_auto(soc, soccfg, cfg, method="Nelder-Mead"):
    cfg = make_cfg(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    qub_pulse = cfg["dac"]["qub_pulse"]

    # append ge sweep to inner loop
    cfg["sweep"]["ge"] = {"start": 0, "stop": qub_pulse["gain"], "expts": 2}
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted pulse gains
    fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points
    fpts = map2adcfreq(soc, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    del cfg["sweep"]["gain"]  # program should not use this
    del cfg["sweep"]["freq"]  # program should not use this

    # set again in case of change
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    def measure_signals(pdr, fpt):
        cfg["dac"]["res_pulse"]["gain"] = int(pdr)
        cfg["dac"]["res_pulse"]["freq"] = float(fpt)

        prog = TwoToneProgram(soccfg, cfg)
        avg_d, std_d = prog.acquire(soc, progress=False)

        return ge_result2signals(avg_d, std_d)

    return sweep2D_maximize_template(
        measure_signals,
        xs=pdrs,
        ys=fpts,
        xlabel="Power (a.u)",
        ylabel="Frequency (MHz)",
        method=method,
    )


def measure_ge_ro_dep(soc, soccfg, cfg):
    cfg = make_cfg(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

    # append ge sweep to inner loop
    cfg["sweep"]["ge"] = {"start": 0, "stop": qub_pulse["gain"], "expts": 2}

    # set with / without pi length for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    lens = sweep2array(cfg["sweep"]["length"])  # predicted readout lengths

    cfg["adc"]["ro_length"] = lens[0]
    cfg["dac"]["res_pulse"]["length"] = lens[0] + cfg["adc"]["trig_offset"] + 1.0

    del cfg["sweep"]["length"]  # program should not use this

    def updateCfg(cfg, _, ro_len):
        cfg["adc"]["ro_length"] = ro_len
        cfg["dac"]["res_pulse"]["length"] = ro_len + cfg["adc"]["trig_offset"] + 1.0

    lens, snrs = sweep1D_soft_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        xs=lens,
        progress=True,
        updateCfg=updateCfg,
        xlabel="Readout Length (us)",
        ylabel="Amplitude",
        result2signals=ge_result2signals,
    )

    return lens, snrs


def measure_ge_trig_dep(soc, soccfg, cfg):
    cfg = make_cfg(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "offset")

    # append ge sweep to inner loop
    cfg["sweep"]["ge"] = {"start": 0, "stop": qub_pulse["gain"], "expts": 2}

    # set with / without pi length for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    offsets = sweep2array(cfg["sweep"]["offset"])  # predicted trigger offsets

    del cfg["sweep"]["offset"]  # program should not use this

    res_len = cfg["dac"]["res_pulse"]["length"]
    orig_offset = cfg["adc"]["trig_offset"]

    def updateCfg(cfg, _, offset):
        cfg["adc"]["trig_offset"] = offset
        cfg["dac"]["res_pulse"]["length"] = res_len + offset - orig_offset

    offsets, snrs = sweep1D_soft_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        xs=offsets,
        progress=True,
        updateCfg=updateCfg,
        xlabel="Trigger Offset (us)",
        ylabel="Amplitude",
        result2signals=ge_result2signals,
    )

    return offsets, snrs
