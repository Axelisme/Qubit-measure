from copy import deepcopy
from typing import List, Tuple

import numpy as np
from zcu_tools.liveplot.jupyter import LivePlotter1D
from zcu_tools.notebook.single_qubit.process import minus_background, rotate2real
from zcu_tools.program.v2 import (
    MuxResetRabiProgram,
    OneToneProgram,
    ResetRabiProgram,
    TwoToneProgram,
)
from zcu_tools.program.v2.base.simulate import SimulateProgramV2
from zcu_tools.tools import print_traceback

from ...flux import set_flux
from ...tools import format_sweep1D, sweep2array, sweep2param
from ..template import sweep_hard_template


def qub_signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(minus_background(signals))


def measure_reset_freq(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification
    reset_pulse = cfg["dac"]["reset_pulse"]

    if cfg["dac"]["reset"] != "pulse":
        raise ValueError("Reset pulse must be one pulse")

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

    sweep_cfg = cfg["sweep"]["freq"]
    reset_pulse["freq"] = sweep2param("freq", sweep_cfg)

    fpts = sweep2array(sweep_cfg)  # predicted frequency points

    prog, signals = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        OneToneProgram,
        ticks=(fpts,),
        progress=True,
        xlabel="Frequency (MHz)",
        ylabel="Amplitude",
        signal2real=qub_signal2real,
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("reset", "freq", as_array=True)

    return fpts, signals


def measure_mux_reset_freq(
    soc, soccfg, cfg
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    reset_pulse1 = cfg["dac"]["reset_pulse1"]
    reset_pulse2 = cfg["dac"]["reset_pulse2"]

    if cfg["dac"]["reset"] != "mux_dual_pulse":
        raise ValueError("Reset pulse must be two pulse")

    # force freq1 to be the outer loop
    cfg["sweep"] = {"freq1": cfg["sweep"]["freq1"], "freq2": cfg["sweep"]["freq2"]}

    reset_pulse1["freq"] = sweep2param("freq1", cfg["sweep"]["freq1"])
    reset_pulse2["freq"] = sweep2param("freq2", cfg["sweep"]["freq2"])

    fpts1 = sweep2array(cfg["sweep"]["freq1"])  # predicted frequency points
    fpts2 = sweep2array(cfg["sweep"]["freq2"])  # predicted frequency points

    prog, signals = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        OneToneProgram,
        ticks=(fpts1, fpts2),
        progress=True,
        xlabel="Frequency1 (MHz)",
        ylabel="Frequency2 (MHz)",
        signal2real=qub_signal2real,
    )

    # get the actual frequency points
    fpts1 = prog.get_pulse_param("reset1", "freq", as_array=True)
    fpts2 = prog.get_pulse_param("reset2", "freq", as_array=True)

    return fpts1, fpts2, signals


def measure_reset_time(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

    # prepend ge sweep to inner loop
    cfg["sweep"] = {
        "ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2},
        "length": cfg["sweep"]["length"],
    }

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    len_params = sweep2param("length", cfg["sweep"]["length"])
    if cfg["dac"]["reset"] == "pulse":
        cfg["dac"]["reset_pulse"]["length"] = len_params
    elif cfg["dac"]["reset"] == "mux_dual_pulse":
        cfg["dac"]["reset_pulse1"]["length"] = len_params
        cfg["dac"]["reset_pulse2"]["length"] = len_params
    else:
        raise ValueError(f"Reset type {cfg['dac']['reset']} not supported")

    lens = sweep2array(cfg["sweep"]["length"])  # predicted pulse gains

    def result2signals(avg_d: list, std_d: list) -> Tuple[np.ndarray, np.ndarray]:
        avg_d = avg_d[0][0].dot([1, 1j])  # (ge, *sweep)
        std_d = std_d[0][0].dot([1, 1j])  # (ge, *sweep)
        avg_d = avg_d[1, ...] - avg_d[0, ...]  # (*sweep)
        std_d = np.sqrt(std_d[1, ...] ** 2 + std_d[0, ...] ** 2)  # (*sweep)

        return avg_d, std_d

    prog, signals = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        ticks=(lens,),
        xlabel="Length (us)",
        ylabel="Amplitude",
        result2signals=result2signals,
    )

    # get the actual pulse gains and frequency points
    pulse_name = "reset" if cfg["dac"]["reset"] == "pulse" else "reset1"
    real_lens = prog.get_pulse_param(pulse_name, "length", as_array=True)
    real_lens += lens[0] - real_lens[0]  # adjust to the first length

    return real_lens, signals  # lens


def measure_reset_amprabi(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    gain_sweep = cfg["sweep"]["gain"]

    cfg["dac"]["qub_pulse"]["gain"] = sweep2param("gain", gain_sweep)

    amps = sweep2array(gain_sweep)  # predicted amplitudes
    signals_all = np.full((2, len(amps)), np.nan, dtype=complex)

    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"], progress=True)

    def raw2result(ir, sum_d, sum2_d) -> Tuple[List, List]:
        avg_d = [d / (ir + 1) for d in sum_d]
        std_d = [np.sqrt(d2 / (ir + 1) - d**2) for d, d2 in zip(avg_d, sum2_d)]
        return avg_d, std_d

    def result2signals(avg_d: list, std_d: list) -> Tuple[np.ndarray, np.ndarray]:
        avg_d = avg_d[0][0].dot([1, 1j])  # (*sweep)
        std_d = np.max(std_d[0][0], axis=-1)  # (*sweep)

        return avg_d, std_d

    def signal2real(signals) -> np.ndarray:
        return rotate2real(signals).real

    if cfg["dac"]["reset_test"] == "pulse":
        reset_pulse = cfg["dac"]["reset_test_pulse"]
        reset_gain = reset_pulse["gain"]
    elif cfg["dac"]["reset_test"] == "mux_dual_pulse":
        reset_pulse1 = cfg["dac"]["reset_test_pulse1"]
        reset_pulse2 = cfg["dac"]["reset_test_pulse2"]
        reset_gain1 = reset_pulse1["gain"]
        reset_gain2 = reset_pulse2["gain"]
    else:
        raise ValueError(f"Reset type {cfg['dac']['reset_test']} not supported")

    prog = None
    with LivePlotter1D("Pulse gain", "Amplitude", num_line=2) as viewer:
        try:
            for i in range(2):
                if cfg["dac"]["reset_test"] == "pulse":
                    reset_pulse["gain"] = i * reset_gain
                elif cfg["dac"]["reset_test"] == "mux_dual_pulse":
                    reset_pulse1["gain"] = i * reset_gain1
                    reset_pulse2["gain"] = i * reset_gain2

                def callback(ir, sum_d, sum2_d) -> None:
                    signals_all[i, :], _ = result2signals(
                        *raw2result(ir, sum_d, sum2_d)
                    )
                    viewer.update(amps, signal2real(signals_all))

                if cfg["dac"]["reset_test"] == "pulse":
                    prog = ResetRabiProgram(soccfg, cfg)
                elif cfg["dac"]["reset_test"] == "mux_dual_pulse":
                    prog = MuxResetRabiProgram(soccfg, cfg)
                else:
                    raise ValueError(
                        f"Reset type {cfg['dac']['reset_test']} not supported"
                    )

                avg_d, std_d = prog.acquire(soc, progress=True, callback=callback)
                signals_all[i, :], _ = result2signals(avg_d, std_d)
        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
            viewer.update(amps, signal2real(signals_all))
        except Exception as e:
            if prog is None:
                raise e  # the error is happen in initialize of program
            print("Error during measurement:")
            print_traceback()

    # get the actual amplitudes
    amps: np.ndarray = prog.get_pulse_param("qub_pulse", "gain", as_array=True)  # type: ignore

    return amps, signals_all


def visualize_reset_time(soccfg, cfg, *, time_fly=0.0) -> None:
    cfg = deepcopy(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

    # prepend ge sweep to inner loop
    cfg["sweep"] = {
        "ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2},
        "length": cfg["sweep"]["length"],
    }

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    len_params = sweep2param("length", cfg["sweep"]["length"])
    if cfg["dac"]["reset"] == "pulse":
        cfg["dac"]["reset_pulse"]["length"] = len_params
    elif cfg["dac"]["reset"] == "mux_dual_pulse":
        cfg["dac"]["reset_pulse1"]["length"] = len_params
        cfg["dac"]["reset_pulse2"]["length"] = len_params
    else:
        raise ValueError(f"Reset type {cfg['dac']['reset']} not supported")

    visualizer = SimulateProgramV2(TwoToneProgram, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)


def visualize_reset_amprabi(soccfg, cfg, *, time_fly=0.0) -> None:
    cfg = deepcopy(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    gain_sweep = cfg["sweep"]["gain"]

    cfg["dac"]["qub_pulse"]["gain"] = sweep2param("gain", gain_sweep)

    if cfg["dac"]["reset_test"] == "pulse":
        progCls = ResetRabiProgram
    elif cfg["dac"]["reset_test"] == "mux_dual_pulse":
        progCls = MuxResetRabiProgram
    else:
        raise ValueError(f"Reset type {cfg['dac']['reset_test']} not supported")

    visualizer = SimulateProgramV2(progCls, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)
