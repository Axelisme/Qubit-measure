import numpy as np
from zcu_tools import make_cfg
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.schedule.tools import format_sweep1D, sweep2array, sweep2param
from zcu_tools.schedule.v2.template import sweep_hard_template


def qub_signals2reals(signals):
    """
    Convert complex qubit signals to real amplitudes by removing mean background.

    Args:
        signals (ndarray): Raw complex qubit measurement signals.

    Returns:
        ndarray: Absolute values of signals after subtracting the mean.
    """
    return np.abs(signals - np.mean(signals))


def measure_qub_freq(
    soc, soccfg, cfg, reset_rf=None, remove_bg=False, earlystop_snr=None
):
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
        earlystop_snr (float, optional): Early stop signal-to-noise ratio threshold.
                                         Defaults to None.

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

    if reset_rf is not None:
        assert cfg["dac"]["reset"] == "pulse", "Need reset=pulse for conjugate reset"
        assert "reset_pulse" in cfg["dac"], "Need reset_pulse for conjugate reset"

    sweep_cfg = cfg["sweep"]["freq"]
    params = sweep2param("freq", sweep_cfg)
    cfg["dac"]["qub_pulse"]["freq"] = params
    if reset_rf is not None:
        cfg["dac"]["reset_pulse"]["freq"] = reset_rf - params

    fpts = sweep2array(sweep_cfg)  # predicted frequency points

    kwargs = {"xlabel": "Frequency (MHz)", "ylabel": "Amplitude"}
    if remove_bg:
        kwargs["signal2real"] = qub_signals2reals

    prog, signals = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        ticks=(fpts,),
        earlystop_snr=earlystop_snr,
        **kwargs,
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("qub_pulse", "freq", as_array=True)

    return fpts, signals
