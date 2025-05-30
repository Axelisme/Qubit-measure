from copy import deepcopy
from typing import Tuple

import numpy as np
from zcu_tools.program.v2 import OneToneProgram, TwoToneProgram
from zcu_tools.notebook.single_qubit.process import minus_background
from zcu_tools.notebook.single_qubit.process import rotate2real

from ...tools import format_sweep1D, sweep2array, sweep2param
from ..template import sweep_hard_template
from ...instant_show import InstantShow1D
from ...flux import set_flux
from zcu_tools.tools import print_traceback


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

    if cfg["dac"]["reset"] != "two_pulse":
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


def measure_mux_reset_amprabi(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    gain_sweep = cfg["sweep"]["gain"]

    cfg["dac"]["qub_pulse"]["gain"] = sweep2param("gain", gain_sweep)

    amps = sweep2array(gain_sweep)  # predicted amplitudes
    signals_all = np.full((2, len(amps)), np.nan, dtype=complex)

    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"], progress=True)

    def raw2result(ir, sum_d, sum2_d):
        avg_d = [d / (ir + 1) for d in sum_d]
        std_d = [np.sqrt(d2 / (ir + 1) - d**2) for d, d2 in zip(avg_d, sum2_d)]
        return avg_d, std_d

    def result2signals(avg_d: list, std_d: list):
        avg_d = avg_d[0][0].dot([1, 1j])  # (*sweep)
        std_d = np.max(std_d[0][0], axis=-1)  # (*sweep)

        return avg_d, std_d

    def signal2real(signals):
        return rotate2real(signals).real

    with InstantShow1D(amps, "Pulse gain", "Amplitude", num_line=2) as viewer:
        for i in range(2):
            if i == 1:
                cfg["dac"]["reset_pulse1"]["gain"] = 0.0
                cfg["dac"]["reset_pulse2"]["gain"] = 0.0

            def callback(ir, sum_d, sum2_d):
                signals_all[i, :], _ = result2signals(*raw2result(ir, sum_d, sum2_d))
                viewer.update_show(signal2real(signals_all))

            try:
                prog = TwoToneProgram(soccfg, cfg)

                avg_d, std_d = prog.acquire(soc, progress=True, callback=callback)
                signals_all[i, :], _ = result2signals(avg_d, std_d)
            except KeyboardInterrupt:
                print("Received KeyboardInterrupt, early stopping the program")
            except Exception:
                print("Error during measurement:")
                print_traceback()

            viewer.update_show(signal2real(signals_all))

    # get the actual amplitudes
    amps: np.ndarray = prog.get_pulse_param("qub_pulse", "gain", as_array=True)  # type: ignore

    return amps, signals_all
