from typing import Tuple, Optional

import numpy as np
from numpy import ndarray
from zcu_tools import make_cfg
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.schedule.tools import format_sweep1D, sweep2array, sweep2param
from zcu_tools.schedule.v2.template import sweep_hard_template


def mist_len_result2signal(
    avg_d: list, std_d: list
) -> Tuple[ndarray, Optional[ndarray]]:
    avg_d = avg_d[0][0].dot([1, 1j])  # (ge, *sweep)
    std_d = std_d[0][0].dot([1, 1j])  # (ge, *sweep)

    avg_d = avg_d[1, ...] - avg_d[0, ...]  # (*sweep)
    std_d = np.sqrt(np.sum(std_d**2, axis=0))  # (*sweep)

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

    qub_pulse["gain"] = sweep2param("w/o", cfg["sweep"]["ge"])
    qub_pulse["length"] = sweep2param("length", len_sweep)

    max_length = max(len_sweep["start"], len_sweep["stop"], qub_pulse["pre_delay"])
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
        signal2real=mist_len_result2signal,
    )

    # get the actual lengths
    lens: ndarray = prog.get_pulse_param("qub_pulse", "length", as_array=True)  # type: ignore

    return lens, signals


def mist_pdr_result2signal(
    avg_d: list, std_d: list
) -> Tuple[ndarray, Optional[ndarray]]:
    avg_d = avg_d[0][0].dot([1, 1j])  # (pdr, )
    std_d = std_d[0][0].dot([1, 1j])  # (pdr, )

    avg_d -= avg_d[0]  # (pdr, )
    std_d = np.sqrt(std_d**2 + std_d[0] ** 2)  # (pdr, )

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
        signal2real=mist_pdr_result2signal,
    )

    # get the actual amplitudes
    amps: ndarray = prog.get_pulse_param("qub_pulse", "gain", as_array=True)  # type: ignore

    return amps, signals
