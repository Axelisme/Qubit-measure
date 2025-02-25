import numpy as np

from zcu_tools import make_cfg
from zcu_tools.program.v2 import GEProgram
from zcu_tools.schedule.tools import (
    format_sweep1D,
    map2adcfreq,
    sweep2array,
    sweep2param,
)
from zcu_tools.schedule.v2.template import sweep_hard_template


def measure_ge_pdr_dep(soc, soccfg, cfg, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    # make sure gain is the outer loop
    if list(cfg["sweep"].keys())[0] == "freq":
        cfg["sweep"] = {
            "gain": cfg["sweep"]["gain"],
            "freq": cfg["sweep"]["freq"],
        }

    res_pulse = cfg["dac"]["res_pulse"]
    res_pulse["gain"] = sweep2param("gain", cfg["sweep"]["gain"])
    res_pulse["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

    pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted pulse gains
    fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points
    fpts = map2adcfreq(soc, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    prog, snr2D = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        GEProgram,
        init_signals=np.full((len(pdrs), len(fpts)), np.nan, dtype=complex),
        ticks=(fpts, pdrs),
        progress=True,
        instant_show=instant_show,
        xlabel="Frequency (MHz)",
        ylabel="Readout Gain",
    )

    # get the actual pulse gains and frequency points
    pdrs = prog.get_pulse_param("res_pulse", "gain", as_array=True)
    fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    return pdrs, fpts, snr2D  # (pdrs, fpts)


def measure_ge_ro_dep(soc, soccfg, cfg, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

    cfg["adc"]["ro_length"] = sweep2param("gain", cfg["sweep"]["length"])
    cfg["dac"]["res_pulse"]["length"] = (
        cfg["adc"]["ro_length"] + cfg["adc"]["trig_offset"] + 1.0
    )

    lens = sweep2array(cfg["sweep"]["length"])  # predicted readout lengths

    prog, snrs = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        GEProgram,
        init_signals=np.full(len(lens), np.nan, dtype=complex),
        ticks=(lens,),
        progress=True,
        instant_show=instant_show,
        xlabel="Readout Length (us)",
        ylabel="Amplitude",
    )

    # get the actual readout lengths
    lens = prog.get_pulse_param("readout_adc", "length", as_array=True)

    return lens, snrs


def measure_ge_trig_dep(soc, soccfg, cfg, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "offset")

    orig_offset = cfg["adc"]["trig_offset"]
    cfg["adc"]["trig_offset"] = sweep2param("offset", cfg["sweep"]["offset"])
    cfg["adc"]["ro_length"] = (
        cfg["adc"]["ro_length"] - cfg["adc"]["trig_offset"] + orig_offset
    )

    offsets = sweep2array(cfg["sweep"]["offset"])  # predicted trigger offsets

    prog, snrs = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        GEProgram,
        init_signals=np.full(len(offsets), np.nan, dtype=complex),
        ticks=(offsets,),
        progress=True,
        instant_show=instant_show,
        xlabel="Trigger Offset (us)",
        ylabel="Amplitude",
    )

    # get the actual trigger offsets
    offsets = prog.get_time_param("trig_offset", "t", as_array=True)

    return offsets, snrs
