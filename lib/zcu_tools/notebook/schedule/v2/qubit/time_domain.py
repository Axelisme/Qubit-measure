# type: ignore

from typing import Tuple

import numpy as np
from zcu_tools.auto import make_cfg
from zcu_tools.notebook.single_qubit.process import rotate2real
from zcu_tools.program.v2 import T1Program, T2EchoProgram, T2RamseyProgram

from ...tools import format_sweep1D, sweep2array, sweep2param
from ..template import sweep_hard_template


def qub_signals2reals(signals):
    return rotate2real(signals).real


def measure_t2ramsey(
    soc, soccfg, cfg, detune: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform a T2* Ramsey measurement on a qubit.

    This function measures the phase coherence time (T2*) of a qubit using the Ramsey
    protocol, which involves applying two π/2 pulses separated by variable delay times.

    Parameters
    ----------
    soc : object
        The socket object for communication with the hardware.
    soccfg : object
        The socket configuration object.
    cfg : dict
        Configuration dictionary containing measurement settings.
        Must include 'sweep' with 'length' parameter.
    detune : float, optional
        Frequency detune value for the Ramsey experiment in MHz, by default 0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - ts: Array of time points in microseconds
        - signals: Array of measured amplitude values
    """
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    sweep_cfg = cfg["sweep"]["length"]

    cfg["detune"] = detune
    cfg["dac"]["t2r_length"] = sweep2param("length", sweep_cfg)

    ts = sweep2array(sweep_cfg)  # predicted times

    # linear hard sweep
    prog, signals = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        T2RamseyProgram,
        ticks=(ts,),
        xlabel="Time (us)",
        ylabel="Amplitude",
        signal2real=qub_signals2reals,
    )

    # get the actual times
    _ts: np.ndarray = prog.get_time_param("t2r_length", "t", as_array=True)
    # TODO: check if this is correct
    ts = _ts + ts[0] - _ts[0]  # adjust to start from the first time

    return ts, signals


def measure_t2echo(
    soc, soccfg, cfg, detune: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform a T2 Echo measurement on a qubit.

    This function measures the phase coherence time (T2) using the Hahn echo technique,
    which adds a π pulse between two π/2 pulses to refocus dephasing caused by low-frequency
    noise.

    Parameters
    ----------
    soc : object
        The socket object for communication with the hardware.
    soccfg : object
        The socket configuration object.
    cfg : dict
        Configuration dictionary containing measurement settings.
        Must include 'sweep' with 'length' parameter.
    detune : float, optional
        Frequency detune value for the T2 echo experiment in MHz, by default 0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - ts: Array of time points in microseconds
        - signals: Array of measured amplitude values
    """
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    sweep_cfg = cfg["sweep"]["length"]

    cfg["detune"] = detune
    cfg["dac"]["t2e_half"] = sweep2param("length", sweep_cfg)

    ts = 2 * sweep2array(sweep_cfg)  # predicted times

    # linear hard sweep
    prog, signals = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        T2EchoProgram,
        ticks=(ts,),
        xlabel="Time (us)",
        ylabel="Amplitude",
        signal2real=qub_signals2reals,
    )

    # get the actual times
    _ts: np.ndarray = 2 * prog.get_time_param("t2e_half", "t", as_array=True)  # type: ignore
    # TODO: check if this is correct
    ts = _ts + ts[0] - _ts[0]  # adjust to start from the first time

    return ts, signals


def measure_t1(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    """Perform a T1 measurement on a qubit.

    This function measures the energy relaxation time (T1) of a qubit by applying a π pulse
    to excite the qubit and measuring the decay after variable delay times.

    Parameters
    ----------
    soc : object
        The socket object for communication with the hardware.
    soccfg : object
        The socket configuration object.
    cfg : dict
        Configuration dictionary containing measurement settings.
        Must include 'sweep' with 'length' parameter.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - ts: Array of time points in microseconds
        - signals: Array of measured amplitude values
    """
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    sweep_cfg = cfg["sweep"]["length"]

    cfg["dac"]["t1_length"] = sweep2param("length", sweep_cfg)

    ts = sweep2array(sweep_cfg)  # predicted times

    # linear hard sweep
    prog, signals = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        T1Program,
        ticks=(ts,),
        xlabel="Time (us)",
        ylabel="Amplitude",
        signal2real=qub_signals2reals,
    )

    # get the actual times
    _ts: np.ndarray = prog.get_time_param("t1_length", "t", as_array=True)
    # TODO: check if this is correct
    ts = _ts + ts[0] - _ts[0]  # adjust to start from the first time

    return ts, signals
