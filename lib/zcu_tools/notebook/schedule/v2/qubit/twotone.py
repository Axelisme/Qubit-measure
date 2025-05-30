from typing import Tuple

import numpy as np
from zcu_tools.auto import make_cfg
from zcu_tools.notebook.single_qubit.process import (
    calculate_noise,
    peak_n_avg,
    rotate2real,
)
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.program.v2.base.simulate import SimulateProgramV2

from ...tools import format_sweep1D, sweep2array, sweep2param
from ..template import sweep_hard_template


def qub_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


def qub_signal2snr(signals: np.ndarray) -> float:
    noise, m_signals = calculate_noise(signals)

    m_amps = qub_signal2real(m_signals)
    contrast = peak_n_avg(m_amps, n=3, mode="max")

    return contrast / noise


def measure_qub_freq(
    soc, soccfg, cfg, remove_bg=False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a frequency sweep measurement of a qubit using two-tone spectroscopy.

    Args:
        soc (object): Socket object for communication with hardware.
        soccfg (object): Socket configuration object.
        cfg (dict): Configuration dictionary containing measurement parameters.
        reset_rf (float, optional): Reset frequency for conjugate reset pulse. If None,
                                   conjugate reset is not used. Defaults to None.
        remove_bg (bool, optional): Whether to remove background from signals.
                                   Defaults to False.

    Returns:
        tuple:
            - fpts (ndarray): Array of frequency points used in the measurement.
            - signals (ndarray): Measured signals at each frequency point.

    Notes:
        The function sets up a frequency sweep for two-tone spectroscopy and
        can optionally implement a conjugate reset protocol when reset_rf is provided.
    """
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

    sweep_cfg = cfg["sweep"]["freq"]
    params = sweep2param("freq", sweep_cfg)
    cfg["dac"]["qub_pulse"]["freq"] = params

    fpts = sweep2array(sweep_cfg)  # predicted frequency points

    kwargs = {"xlabel": "Frequency (MHz)", "ylabel": "Amplitude"}
    if remove_bg:
        kwargs["signal2real"] = qub_signal2real

    prog, signals = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        ticks=(fpts,),
        **kwargs,
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("qub_pulse", "freq", as_array=True)

    return fpts, signals


def visualize_qub_freq(soccfg, cfg, time_fly=0.0) -> None:
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

    sweep_cfg = cfg["sweep"]["freq"]
    params = sweep2param("freq", sweep_cfg)
    cfg["dac"]["qub_pulse"]["freq"] = params

    visualizer = SimulateProgramV2(TwoToneProgram, soccfg, cfg)
    visualizer.visualize(time_fly=time_fly)
