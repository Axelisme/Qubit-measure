from typing import Tuple

import numpy as np
from zcu_tools import make_cfg
from zcu_tools.analysis import minus_background
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.schedule.tools import format_sweep1D, sweep2array, sweep2param
from zcu_tools.schedule.v2.template import sweep_hard_template


def qub_signals2reals(signals):
    return np.abs(minus_background(signals, method="mean"))


def measure_lenrabi(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    """Measure Rabi oscillation by sweeping pulse length.

    This function performs a Rabi measurement experiment by varying the length of the qubit pulse
    and measuring the amplitude response. It allows for characterization of qubit control through
    the observation of Rabi oscillations.

    Parameters
    ----------
    soc : object
        The system-on-chip object that controls the hardware.
    soccfg : object
        Configuration for the system-on-chip.
    cfg : dict
        Configuration dictionary containing experiment parameters.
        Must include:
        - sweep: dict with length sweep parameters
        - dac: dict with qubit pulse settings

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - First element: Array of actual pulse lengths used (in microseconds)
        - Second element: Array of measured amplitude responses
    """
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

    sweep_cfg = cfg["sweep"]["length"]
    cfg["dac"]["qub_pulse"]["length"] = sweep2param("length", sweep_cfg)

    lens = sweep2array(sweep_cfg)  # predicted lengths

    prog, signals = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        ticks=(lens,),
        progress=True,
        xlabel="Length (us)",
        ylabel="Amplitude",
        result2signals=qub_signals2reals,
    )

    # get the actual lengths
    lens: np.ndarray = prog.get_pulse_param("qub_pulse", "length", as_array=True)  # type: ignore

    return lens, signals


def measure_amprabi(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    """Measure Rabi oscillation by sweeping pulse amplitude (gain).

    This function performs a Rabi measurement experiment by varying the amplitude/gain of the
    qubit pulse and measuring the response. It allows for characterization of qubit control
    by observing how the system responds to different pulse powers.

    Parameters
    ----------
    soc : object
        The system-on-chip object that controls the hardware.
    soccfg : object
        Configuration for the system-on-chip.
    cfg : dict
        Configuration dictionary containing experiment parameters.
        Must include:
        - sweep: dict with gain sweep parameters
        - dac: dict with qubit pulse settings

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - First element: Array of actual pulse gains/amplitudes used
        - Second element: Array of measured amplitude responses
    """
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")

    sweep_cfg = cfg["sweep"]["gain"]
    cfg["dac"]["qub_pulse"]["gain"] = sweep2param("gain", sweep_cfg)

    amps = sweep2array(sweep_cfg)  # predicted amplitudes

    prog, signals = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        ticks=(amps,),
        progress=True,
        xlabel="Pulse gain",
        ylabel="Amplitude",
        result2signals=qub_signals2reals,
    )

    # get the actual amplitudes
    amps: np.ndarray = prog.get_pulse_param("qub_pulse", "gain", as_array=True)  # type: ignore

    return amps, signals
