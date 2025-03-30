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
    """
    Calculate signal-to-noise ratio from average and standard deviation data.

    Parameters
    ----------
    avg_d : ndarray
        Average data with shape (*sweep, 2) where last dimension represents ground and excited states
    std_d : ndarray
        Standard deviation data with same shape as avg_d

    Returns
    -------
    ndarray
        Signal-to-noise ratio calculated as contrast divided by noise
    """
    contrast = avg_d[..., 1] - avg_d[..., 0]  # (*sweep)
    noise2_i = np.sum(std_d.real**2, axis=-1)  # (*sweep)
    noise2_q = np.sum(std_d.imag**2, axis=-1)  # (*sweep)
    noise = np.sqrt(noise2_i * contrast.real**2 + noise2_q * contrast.imag**2) / np.abs(
        contrast
    )

    return contrast / noise


def ge_result2signals(avg_d, std_d):
    """
    Convert raw measurement results to signal-to-noise ratio for ground/excited state discrimination.

    Parameters
    ----------
    avg_d : ndarray
        Average data from measurement
    std_d : ndarray
        Standard deviation data from measurement

    Returns
    -------
    tuple
        (SNR matrix transposed, None)
    """
    avg_d = avg_d[0][0].dot([1, 1j])  # (*sweep, ge)
    std_d = std_d[0][0].dot([1, 1j])  # (*sweep, ge)

    return calc_snr(avg_d, std_d).T, None


def measure_ge_pdr_dep(soc, soccfg, cfg):
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
    """
    Automatically optimize pulse drive and frequency for ground/excited state discrimination.

    Uses optimization algorithm to find the best parameters for maximizing SNR.

    Parameters
    ----------
    soc : object
        System-on-chip object for hardware control
    soccfg : object
        System-on-chip configuration
    cfg : dict
        Configuration dictionary with DAC, sweep, and other settings
    method : str, optional
        Optimization method, default is "Nelder-Mead"

    Returns
    -------
    tuple
        Result from sweep2D_maximize_template, containing optimal parameters and SNR values
    """
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
