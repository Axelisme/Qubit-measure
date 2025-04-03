# type: ignore

from typing import Tuple

import numpy as np
from zcu_tools import make_cfg
from zcu_tools.program.v2 import T1Program, T2EchoProgram, T2RamseyProgram
from zcu_tools.schedule.tools import format_sweep1D, sweep2array, sweep2param
from zcu_tools.schedule.v2.template import sweep1D_soft_template, sweep_hard_template


def measure_t2ramsey(
    soc, soccfg, cfg, soft_loop=False, detune: float = 0
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
    soft_loop : bool, optional
        If True, uses software loop instead of hardware sweep, by default False.
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

    cfg["detune"] = detune
    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

    sweep_cfg = cfg["sweep"]["length"]
    ts = sweep2array(sweep_cfg, allow_array=True)  # predicted times

    args = (soc, soccfg, cfg, T2RamseyProgram)
    kwargs = dict(
        xlabel="Time (us)",
        ylabel="Amplitude",
    )
    if isinstance(sweep_cfg, np.ndarray) or isinstance(sweep_cfg, list) or soft_loop:
        # custom soft sweep
        del cfg["sweep"]  # program should not use this

        cfg["dac"]["t2r_length"] = ts[0]  # initial value

        def updateCfg(cfg, _, t):
            cfg["dac"]["t2r_length"] = t

        ts, signals = sweep1D_soft_template(*args, xs=ts, updateCfg=updateCfg, **kwargs)

    elif isinstance(sweep_cfg, dict):
        # linear hard sweep
        cfg["dac"]["t2r_length"] = sweep2param("length", sweep_cfg)
        prog, signals = sweep_hard_template(*args, ticks=(ts,), **kwargs)

        # get the actual times
        _ts: np.ndarray = prog.get_time_param("t2r_length", "t", as_array=True)
        ts = _ts + ts[0] - _ts[0]  # adjust to start from the first time
    else:
        raise ValueError("Invalid sweep configuration")

    return ts, signals


def measure_t2echo(soc, soccfg, cfg, soft_loop=False) -> Tuple[np.ndarray, np.ndarray]:
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
    soft_loop : bool, optional
        If True, uses software loop instead of hardware sweep, by default False.

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
    ts = 2 * sweep2array(sweep_cfg)  # predicted times

    args = (soc, soccfg, cfg, T2EchoProgram)
    kwargs = dict(
        xlabel="Time (us)",
        ylabel="Amplitude",
    )
    if isinstance(sweep_cfg, np.ndarray) or isinstance(sweep_cfg, list) or soft_loop:
        # custom soft sweep
        del cfg["sweep"]  # program should not use this

        cfg["dac"]["t2e_half"] = float(ts[0]) / 2  # initial value

        def updateCfg(cfg, _, t):
            cfg["dac"]["t2e_half"] = float(t) / 2  # half the time

        ts, signals = sweep1D_soft_template(*args, xs=ts, updateCfg=updateCfg, **kwargs)
    elif isinstance(sweep_cfg, dict):
        # linear hard sweep
        cfg["dac"]["t2e_half"] = sweep2param("length", sweep_cfg)
        prog, signals = sweep_hard_template(*args, ticks=(ts,), **kwargs)

        # get the actual times
        _ts: np.ndarray = 2 * prog.get_time_param("t2e_half", "t", as_array=True)  # type: ignore
        ts = _ts + ts[0] - _ts[0]  # adjust to start from the first time
    else:
        raise ValueError("Invalid sweep configuration")

    return ts, signals


def measure_t1(soc, soccfg, cfg, soft_loop=False) -> Tuple[np.ndarray, np.ndarray]:
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
    soft_loop : bool, optional
        If True, uses software loop instead of hardware sweep, by default False.

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
    ts = sweep2array(sweep_cfg)  # predicted times

    args = (soc, soccfg, cfg, T1Program)
    kwargs = dict(
        xlabel="Time (us)",
        ylabel="Amplitude",
    )

    if isinstance(sweep_cfg, np.ndarray) or isinstance(sweep_cfg, list) or soft_loop:
        # custom soft sweep
        del cfg["sweep"]

        cfg["dac"]["t1_length"] = ts[0]  # initial value

        def updateCfg(cfg, _, t):
            cfg["dac"]["t1_length"] = t

        ts, signals = sweep1D_soft_template(*args, xs=ts, updateCfg=updateCfg, **kwargs)
    elif isinstance(sweep_cfg, dict):
        # linear hard sweep
        cfg["dac"]["t1_length"] = sweep2param("length", sweep_cfg)
        prog, signals = sweep_hard_template(*args, ticks=(ts,), **kwargs)

        # get the actual times
        _ts: np.ndarray = prog.get_time_param("t1_length", "t", as_array=True)
        ts = _ts + ts[0] - _ts[0]
    else:
        raise ValueError("Invalid sweep configuration")

    return ts, signals
