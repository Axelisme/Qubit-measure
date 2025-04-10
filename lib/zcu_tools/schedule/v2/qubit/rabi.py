from typing import Tuple, Literal

import numpy as np
from zcu_tools import make_cfg
from zcu_tools.analysis import calculate_noise, peak_n_avg, rotate2real
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.schedule.tools import format_sweep1D, sweep2array, sweep2param
from zcu_tools.schedule.v2.template import sweep_hard_template


def qub_signals2reals(signals):
    return rotate2real(signals).real


def qub_signal2snr(signals):
    noise, m_signals = calculate_noise(signals)

    m_real = rotate2real(m_signals).real
    contrast = peak_n_avg(m_real, n=3, mode="max") - peak_n_avg(m_real, n=3, mode="min")

    return contrast / noise


def measure_lenrabi(
    soc,
    soccfg,
    cfg,
    *,
    force_align=True,
    align_type: Literal["pre_delay", "post_delay"] = "post_delay",
    earlystop_snr=None,
) -> Tuple[np.ndarray, np.ndarray]:
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
    force_align : bool, optional
        If True, add alignment delay time before qub_pulse to ensure the total length is
        consistent.
    earlystop_snr : float, optional
        Early stop signal-to-noise ratio threshold. If provided, the measurement will stop
        when the SNR exceeds this value.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - First element: Array of actual pulse lengths used (in microseconds)
        - Second element: Array of measured amplitude responses
    """
    cfg = make_cfg(cfg)  # prevent in-place modification

    qub_pulse = cfg["dac"]["qub_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
    len_sweep = cfg["sweep"]["length"]

    qub_pulse["length"] = sweep2param("length", len_sweep)

    if force_align:
        max_length = max(
            len_sweep["start"], len_sweep["stop"], qub_pulse.get(align_type, 0.0)
        )
        qub_pulse[align_type] = max_length - qub_pulse["length"]

    if earlystop_snr is not None:

        def checker(signals):
            snr = qub_signal2snr(signals)
            return snr >= earlystop_snr, f"SNR: {snr:.2f}"

    else:
        checker = None

    lens = sweep2array(len_sweep)  # predicted lengths
    prog, signals = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        ticks=(lens,),
        progress=True,
        xlabel="Length (us)",
        ylabel="Amplitude",
        signal2real=qub_signals2reals,
        early_stop_checker=checker,
    )

    # get the actual lengths
    lens: np.ndarray = prog.get_pulse_param("qub_pulse", "length", as_array=True)  # type: ignore

    return lens, signals


def measure_amprabi(
    soc, soccfg, cfg, earlystop_snr=None
) -> Tuple[np.ndarray, np.ndarray]:
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
    earlystop_snr : float, optional
        Early stop signal-to-noise ratio threshold. If provided, the measurement will stop
        when the SNR exceeds this value.

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

    if earlystop_snr is not None:

        def checker(signals):
            snr = qub_signal2snr(signals)
            return snr >= earlystop_snr, f"SNR: {snr:.2g}"

    else:
        checker = None

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
        signal2real=qub_signals2reals,
        early_stop_checker=checker,
    )

    # get the actual amplitudes
    amps: np.ndarray = prog.get_pulse_param("qub_pulse", "gain", as_array=True)  # type: ignore

    return amps, signals
