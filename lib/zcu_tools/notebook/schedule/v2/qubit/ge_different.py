from typing import Optional, Tuple

import numpy as np
from numpy import ndarray
from zcu_tools import make_cfg
from zcu_tools.program.v2 import TwoToneProgram

from ...tools import format_sweep1D, map2adcfreq, sweep2array, sweep2param
from ..template import sweep1D_soft_template, sweep_hard_template


def calc_snr(avg_d: ndarray, std_d: ndarray) -> ndarray:
    contrast = avg_d[1, ...] - avg_d[0, ...]  # (*sweep)
    noise2_i = np.sum(std_d.real**2, axis=0)  # (*sweep)
    noise2_q = np.sum(std_d.imag**2, axis=0)  # (*sweep)
    noise = np.sqrt(noise2_i * contrast.real**2 + noise2_q * contrast.imag**2) / np.abs(
        contrast
    )

    return contrast / noise


def ge_result2signals(avg_d: list, std_d: list) -> Tuple[ndarray, Optional[ndarray]]:
    avg_d = avg_d[0][0].dot([1, 1j])  # (ge, *sweep)
    std_d = std_d[0][0].dot([1, 1j])  # (ge, *sweep)

    return calc_snr(avg_d, std_d), None


def measure_ge_freq_dep(soc, soccfg, cfg):
    """
    Measure ground/excited state discrimination as a function of frequency.

    Parameters
    ----------
    soc : object
        System-on-chip object for hardware control
    soccfg : object
        System-on-chip configuration
    cfg : dict
        Configuration dictionary with DAC, sweep, and other settings

    Returns
    -------
    tuple
        (frequency_points, snrs)
        Returns frequency points, and the resulting SNR matrix
    """
    cfg = make_cfg(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    qub_pulse = cfg["dac"]["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

    # prepend ge sweep to inner loop
    cfg["sweep"] = {
        "ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2},
        "freq": cfg["sweep"]["freq"],
    }

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    res_pulse["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

    fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points
    fpts = map2adcfreq(soc, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    prog, snrs = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        ticks=(fpts,),
        xlabel="Frequency (MHz)",
        ylabel="SNR",
        result2signals=ge_result2signals,
    )

    # get the actual pulse gains and frequency points
    fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    return fpts, snrs  # fpts


def measure_ge_pdr_dep(soc, soccfg, cfg):
    """
    Measure ground/excited state discrimination as a function of pulse drive.

    Parameters
    ----------
    soc : object
        System-on-chip object for hardware control
    soccfg : object
        System-on-chip configuration
    cfg : dict
        Configuration dictionary with DAC, sweep, and other settings

    Returns
    -------
    tuple
        (pdrs, snrs)
        Returns pulse drive, and the resulting SNR matrix
    """
    cfg = make_cfg(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    qub_pulse = cfg["dac"]["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")

    # prepend ge sweep to inner loop
    cfg["sweep"] = {
        "ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2},
        "gain": cfg["sweep"]["gain"],
    }

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    res_pulse["gain"] = sweep2param("gain", cfg["sweep"]["gain"])

    pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted pulse gains

    prog, snrs = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        ticks=(pdrs,),
        xlabel="Power (a.u)",
        ylabel="SNR",
        result2signals=ge_result2signals,
    )

    # get the actual pulse gains and frequency points
    pdrs = prog.get_pulse_param("res_pulse", "gain", as_array=True)

    return pdrs, snrs  # pdrs


def measure_ge_pdr_dep2D(soc, soccfg, cfg):
    """
    Measure ground/excited state discrimination as a function of pulse drive and frequency.

    Parameters
    ----------
    soc : object
        System-on-chip object for hardware control
    soccfg : object
        System-on-chip configuration
    cfg : dict
        Configuration dictionary with DAC, sweep, and other settings

    Returns
    -------
    tuple
        (pulse_drive_values, frequency_points, snr_2D_matrix)
        Returns actual pulse drive values, frequency points, and the resulting SNR matrix
    """
    cfg = make_cfg(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    qub_pulse = cfg["dac"]["qub_pulse"]

    # prepend ge sweep to inner loop
    cfg["sweep"] = {
        "ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2},
        "gain": cfg["sweep"]["gain"],
        "freq": cfg["sweep"]["freq"],
    }

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
        ticks=(pdrs, fpts),
        xlabel="Readout Gain",
        ylabel="Frequency (MHz)",
        result2signals=ge_result2signals,
    )

    # get the actual pulse gains and frequency points
    pdrs = prog.get_pulse_param("res_pulse", "gain", as_array=True)
    fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    return pdrs, fpts, snr2D  # (pdrs, fpts)


def measure_ge_ro_dep(soc, soccfg, cfg):
    """
    Measure ground/excited state discrimination as a function of readout length.

    Parameters
    ----------
    soc : object
        System-on-chip object for hardware control
    soccfg : object
        System-on-chip configuration
    cfg : dict
        Configuration dictionary with DAC, sweep, and other settings

    Returns
    -------
    tuple
        (readout_lengths, signal_to_noise_ratios)
        Returns the readout length values and corresponding SNR values
    """
    cfg = make_cfg(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    length_sweep = cfg["sweep"]["length"]

    # replace ge sweep to sweep, and use soft loop for length
    cfg["sweep"] = {"ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2}}

    # set with / without pi length for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    lens = sweep2array(length_sweep)  # predicted readout lengths

    cfg["adc"]["ro_length"] = lens[0]
    cfg["dac"]["res_pulse"]["length"] = lens.max() + cfg["adc"]["trig_offset"] + 0.1

    def updateCfg(cfg, _, ro_len):
        cfg["adc"]["ro_length"] = ro_len

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
    """
    Measure ground/excited state discrimination as a function of trigger offset.

    Parameters
    ----------
    soc : object
        System-on-chip object for hardware control
    soccfg : object
        System-on-chip configuration
    cfg : dict
        Configuration dictionary with DAC, sweep, and other settings

    Returns
    -------
    tuple
        (trigger_offsets, signal_to_noise_ratios)
        Returns the trigger offset values and corresponding SNR values
    """
    cfg = make_cfg(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "offset")
    offset_sweep = cfg["sweep"]["offset"]

    # replace ge sweep to loop, and use soft loop for offset
    cfg["sweep"] = {"ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2}}

    # set with / without pi length for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    offsets = sweep2array(offset_sweep)  # predicted trigger offsets

    cfg["adc"]["trig_offset"] = offsets[0]
    cfg["dac"]["res_pulse"]["length"] = offsets.max() + cfg["adc"]["ro_length"] + 0.1

    def updateCfg(cfg, _, offset):
        cfg["adc"]["trig_offset"] = offset

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
