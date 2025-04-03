import numpy as np
from zcu_tools import make_cfg
from zcu_tools.analysis import minus_background
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.schedule.tools import sweep2array, sweep2param
from zcu_tools.schedule.v2.template import sweep2D_soft_hard_template
from zcu_tools.schedule.v2.qubit.twotone import qub_signal2snr


def qub_signals2reals(signals):
    return np.abs(minus_background(signals, axis=1))


def measure_qub_flux_dep(soc, soccfg, cfg, reset_rf=None, earlystop_snr=None):
    """
    Measure qubit frequency as a function of flux bias.

    This function performs a two-dimensional sweep over flux values and qubit frequencies
    using a two-tone spectroscopy technique. It measures the qubit response at different
    flux biases to characterize flux dependence of the qubit frequency.

    Args:
        soc: System-on-chip controller object.
        soccfg: System-on-chip configuration object.
        cfg (dict): Configuration dictionary containing measurement settings.
            Required keys:
            - dev: Device configuration with flux_dev and flux settings.
            - dac: DAC configuration with qub_pulse settings.
            - sweep: Sweep parameters for freq and flux.
        reset_rf (float, optional): Reset frequency for conjugate reset pulse.
            If provided, enables reset pulse with conjugate frequency. Defaults to None.
        earlystop_snr (float, optional): Early stop signal-to-noise ratio threshold.
            If provided, the measurement will stop if the SNR exceeds this value.
            Defaults to None.

    Returns:
        tuple: Three-element tuple containing:
            - flxs (numpy.ndarray): Array of flux values used in the sweep.
            - fpts (numpy.ndarray): Array of frequency points actually used in the measurement.
            - signals2D (numpy.ndarray): 2D array of measured signals, with shape (len(flxs), len(fpts)).

    Raises:
        ValueError: If cfg["dev"]["flux_dev"] is set to "none" while attempting a flux sweep.
    """
    cfg = make_cfg(cfg)  # prevent in-place modification

    if cfg["dev"]["flux_dev"] == "none":
        raise ValueError("Flux sweep but get flux_dev == 'none'")

    qub_pulse = cfg["dac"]["qub_pulse"]
    fpt_sweep = cfg["sweep"]["freq"]
    flx_sweep = cfg["sweep"]["flux"]
    qub_pulse["freq"] = sweep2param("freq", fpt_sweep)

    del cfg["sweep"]["flux"]  # use for loop here

    if reset_rf is not None:
        assert cfg["dac"]["reset"] == "pulse", "Need reset=pulse for conjugate reset"
        assert "reset_pulse" in cfg["dac"], "Need reset_pulse for conjugate reset"
        cfg["dac"]["reset_pulse"]["freq"] = reset_rf - qub_pulse["freq"]

    mAs = sweep2array(flx_sweep, allow_array=True)
    fpts = sweep2array(fpt_sweep)

    cfg["dev"]["flux"] = mAs[0]  # set initial flux

    def updateCfg(cfg, _, mA):
        cfg["dev"]["flux"] = mA * 1e-3  # convert to A

    if earlystop_snr is not None:

        def checker(signals):
            snr = qub_signal2snr(signals)
            return snr >= earlystop_snr, f"Current SNR: {snr:.2g}"

    else:
        checker = None

    prog, signals2D = sweep2D_soft_hard_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        xs=1e3 * mAs,
        ys=fpts,
        xlabel="Flux (mA)",
        ylabel="Frequency (MHz)",
        updateCfg=updateCfg,
        signal2real=qub_signals2reals,
        early_stop_checker=checker,
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("qub_pulse", "freq", as_array=True)

    return mAs, fpts, signals2D
