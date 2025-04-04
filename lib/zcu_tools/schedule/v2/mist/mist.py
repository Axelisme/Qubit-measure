from typing import Tuple

import numpy as np
from numpy import ndarray
from zcu_tools import make_cfg
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.schedule.tools import format_sweep1D, sweep2array, sweep2param
from zcu_tools.schedule.v2.template import (
    sweep_hard_template,
    sweep2D_soft_hard_template,
)


def mist_len_result2signal(avg_d: list, std_d: list):
    avg_d = avg_d[0][0].dot([1, 1j])  # (ge, *sweep)
    std_d = std_d[0][0].dot([1, 1j])  # (ge, *sweep)

    avg_d = avg_d[1, ...] - avg_d[0, ...]  # (*sweep)
    std_d = np.sqrt(np.sum(np.abs(std_d) ** 2, axis=0))  # (*sweep)

    return avg_d, std_d


def measure_mist_len_dep(soc, soccfg, cfg) -> Tuple[ndarray, ndarray]:
    cfg = make_cfg(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    len_sweep = cfg["sweep"]["length"]

    cfg["sweep"] = {
        "w/o": {"start": 0, "stop": qub_pulse["gain"], "expts": 2},
        "length": len_sweep,
    }

    qub_pulse["gain"] = sweep2param("w/o", cfg["sweep"]["w/o"])
    qub_pulse["length"] = sweep2param("length", len_sweep)

    max_length = max(
        len_sweep["start"], len_sweep["stop"], qub_pulse.get("pre_delay", 0.0)
    )
    qub_pulse["pre_delay"] = max_length - qub_pulse["length"]

    lens = sweep2array(len_sweep)  # predicted lengths
    prog, signals = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        ticks=(lens,),
        progress=True,
        xlabel="Length (us)",
        ylabel="MIST",
        result2signals=mist_len_result2signal,
    )

    # get the actual lengths
    lens: ndarray = prog.get_pulse_param("qub_pulse", "length", as_array=True)  # type: ignore

    return lens, signals


def mist_pdr_result2signal(avg_d: list, std_d: list):
    avg_d = avg_d[0][0].dot([1, 1j])  # (pdr, *)
    std_d = std_d[0][0].dot([1, 1j])  # (pdr, *)

    avg_d -= avg_d[0]  # (pdr, *)
    std_d = np.sqrt(np.abs(std_d) ** 2 + np.abs(std_d)[0] ** 2)  # (pdr, *)

    return avg_d, std_d


def measure_mist_pdr_dep(soc, soccfg, cfg) -> Tuple[ndarray, ndarray]:
    cfg = make_cfg(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    pdr_sweep = cfg["sweep"]["gain"]

    qub_pulse["gain"] = sweep2param("gain", pdr_sweep)

    amps = sweep2array(pdr_sweep)  # predicted amplitudes
    prog, signals = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        ticks=(amps,),
        progress=True,
        xlabel="Pulse gain",
        ylabel="MIST",
        result2signals=mist_pdr_result2signal,
    )

    # get the actual amplitudes
    amps: ndarray = prog.get_pulse_param("qub_pulse", "gain", as_array=True)  # type: ignore

    return amps, signals


def measure_mist_pdr_len_dep2D(soc, soccfg, cfg) -> Tuple[ndarray, ndarray, ndarray]:
    cfg = make_cfg(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]

    pdr_sweep = cfg["sweep"]["gain"]
    len_sweep = cfg["sweep"]["length"]

    # force gain be the first sweep
    cfg["sweep"] = {"gain": pdr_sweep, "length": len_sweep}

    qub_pulse["gain"] = sweep2param("gain", pdr_sweep)
    qub_pulse["length"] = sweep2param("length", len_sweep)

    max_length = max(
        len_sweep["start"], len_sweep["stop"], qub_pulse.get("pre_delay", 0.0)
    )
    qub_pulse["pre_delay"] = max_length - qub_pulse["length"]

    pdrs = sweep2array(pdr_sweep)  # predicted gains
    lens = sweep2array(len_sweep)  # predicted lengths
    prog, signals2D = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        ticks=(pdrs, lens),
        progress=True,
        xlabel="Drive power (a.u.)",
        ylabel="Length (us)",
        result2signals=mist_pdr_result2signal,
    )

    # get the actual lengths
    pdrs: ndarray = prog.get_pulse_param("qub_pulse", "gain", as_array=True)  # type: ignore
    lens: ndarray = prog.get_pulse_param("qub_pulse", "length", as_array=True)  # type: ignore

    return pdrs, lens, signals2D


def measure_mist_pdr_fpt_dep2D(soc, soccfg, cfg) -> Tuple[ndarray, ndarray, ndarray]:
    cfg = make_cfg(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]

    pdr_sweep = cfg["sweep"]["gain"]
    fpt_sweep = cfg["sweep"]["freq"]

    # force gain be the first sweep
    cfg["sweep"] = {"gain": pdr_sweep, "freq": fpt_sweep}

    qub_pulse["gain"] = sweep2param("gain", pdr_sweep)
    qub_pulse["freq"] = sweep2param("freq", fpt_sweep)

    pdrs = sweep2array(pdr_sweep)  # predicted gains
    fpts = sweep2array(fpt_sweep)  # predicted frequencies
    prog, signals2D = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        ticks=(pdrs, fpts),
        progress=True,
        xlabel="Drive power (a.u.)",
        ylabel="Frequency (MHz)",
        result2signals=mist_pdr_result2signal,
    )

    # get the actual lengths
    pdrs: ndarray = prog.get_pulse_param("qub_pulse", "gain", as_array=True)  # type: ignore
    fpts: ndarray = prog.get_pulse_param("qub_pulse", "freq", as_array=True)  # type: ignore

    return pdrs, fpts, signals2D


def measure_mist_flx_pdr_dep2D(soc, soccfg, cfg) -> Tuple[ndarray, ndarray, ndarray]:
    cfg = make_cfg(cfg)  # prevent in-place modification

    pdr_sweep = cfg["sweep"]["gain"]
    flx_sweep = cfg["sweep"]["flux"]

    del cfg["sweep"]["flux"]  # use soft loop
    cfg["dac"]["qub_pulse"]["gain"] = sweep2param("gain", pdr_sweep)

    mAs = sweep2array(flx_sweep, allow_array=True)  # predicted currents
    pdrs = sweep2array(pdr_sweep)  # predicted gains

    cfg["dev"]["flux"] = mAs[0]  # set initial flux

    def updateCfg(cfg, _, mA):
        cfg["dev"]["flux"] = mA * 1e-3  # convert to A

    prog, signals2D = sweep2D_soft_hard_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        xs=1e3 * mAs,
        ys=pdrs,
        xlabel="Flux (mA)",
        ylabel="Drive power (a.u.)",
        updateCfg=updateCfg,
        result2signals=mist_pdr_result2signal,
    )

    # get the actual lengths
    pdrs: ndarray = prog.get_pulse_param("qub_pulse", "gain", as_array=True)  # type: ignore

    return mAs, pdrs, signals2D
