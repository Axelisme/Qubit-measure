import numpy as np

from zcu_tools import make_cfg
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.schedule.tools import (
    format_sweep1D,
    map2adcfreq,
    sweep2array,
    sweep2param,
)
from zcu_tools.schedule.v2.template import sweep1D_soft_template, sweep_hard_template


def calc_snr(avg_d, std_d):
    contrast = avg_d[..., 1] - avg_d[..., 0]  # (*sweep)
    noise2_i = np.sum(std_d.real**2, axis=-1)  # (*sweep)
    noise2_q = np.sum(std_d.imag**2, axis=-1)  # (*sweep)
    noise = np.sqrt(noise2_i * contrast.real**2 + noise2_q * contrast.imag**2) / np.abs(
        contrast
    )

    return contrast / noise


def ge_raw2signals(ir, sum_d, sum2_d):
    sum_d = sum_d[0][0].dot([1, 1j])  # (*sweep, ge)
    sum2_d = sum2_d[0][0].dot([1, 1j])  # (*sweep, ge)

    avg_d = sum_d / (ir + 1)
    std_d = np.sqrt(sum2_d / (ir + 1) - avg_d**2)

    return calc_snr(avg_d, std_d)


def ge_result2signals(result):
    avg_d, std_d = result
    avg_d = avg_d[0][0].dot([1, 1j])  # (*sweep, ge)
    std_d = std_d[0][0].dot([1, 1j])  # (*sweep, ge)

    return calc_snr(avg_d, std_d)


def measure_ge_pdr_dep(soc, soccfg, cfg, instant_show=False):
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
        init_signals=np.full((len(pdrs), len(fpts)), np.nan, dtype=complex),
        ticks=(fpts, pdrs),
        progress=True,
        instant_show=instant_show,
        xlabel="Frequency (MHz)",
        ylabel="Readout Gain",
        raw2signals=ge_raw2signals,
        result2signals=ge_result2signals,
        ret_std=True,
    )

    # get the actual pulse gains and frequency points
    pdrs = prog.get_pulse_param("res_pulse", "gain", as_array=True)
    fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    return pdrs, fpts, snr2D  # (pdrs, fpts)


def measure_ge_ro_dep(soc, soccfg, cfg, instant_show=False):
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
        init_signals=np.full(len(lens), np.nan, dtype=complex),
        progress=True,
        instant_show=instant_show,
        updateCfg=updateCfg,
        xlabel="Readout Length (us)",
        ylabel="Amplitude",
        result2signals=ge_result2signals,
        ret_std=True,
    )

    return lens, snrs


def measure_ge_trig_dep(soc, soccfg, cfg, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "offset")

    # append ge sweep to inner loop
    cfg["sweep"]["ge"] = {"start": 0, "stop": qub_pulse["gain"], "expts": 2}

    # set with / without pi length for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    offsets = sweep2array(cfg["sweep"]["offset"])  # predicted trigger offsets

    del cfg["sweep"]["offset"]  # program should not use this

    orig_offset = cfg["adc"]["trig_offset"]

    def updateCfg(cfg, _, offset):
        cfg["adc"]["trig_offset"] = offset
        cfg["adc"]["ro_length"] = (
            cfg["adc"]["ro_length"] - cfg["adc"]["trig_offset"] + orig_offset
        )

        if cfg["adc"]["ro_length"] < 0:
            raise ValueError("Readout length cannot be negative")

    offsets, snrs = sweep1D_soft_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        xs=offsets,
        init_signals=np.full(len(offsets), np.nan, dtype=complex),
        progress=True,
        instant_show=instant_show,
        updateCfg=updateCfg,
        xlabel="Trigger Offset (us)",
        ylabel="Amplitude",
        result2signals=ge_result2signals,
        ret_std=True,
    )

    return offsets, snrs
