import numpy as np
from numpy import ndarray
from zcu_tools import make_cfg
from zcu_tools.notebook.single_qubit.process import minus_background
from zcu_tools.program.v2 import TwoToneProgram

from ...tools import sweep2array, sweep2param
from ..template import sweep2D_soft_hard_template


def qub_signals2reals(signals):
    return np.abs(minus_background(signals, axis=1))


def measure_qub_flux_dep(soc, soccfg, cfg):
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
    flx_sweep = cfg["sweep"]["flux"]
    fpt_sweep = cfg["sweep"]["freq"]

    qub_pulse = cfg["dac"]["qub_pulse"]

    if cfg["dev"]["flux_dev"] == "none":
        raise ValueError("Flux sweep but get flux_dev == 'none'")

    qub_pulse["freq"] = sweep2param("freq", fpt_sweep)

    del cfg["sweep"]["flux"]  # use for loop here

    As = sweep2array(flx_sweep, allow_array=True)
    fpts = sweep2array(fpt_sweep)

    cfg["dev"]["flux"] = As[0]  # set initial flux

    def updateCfg(cfg, _, mA):
        cfg["dev"]["flux"] = mA * 1e-3  # convert to A

    prog, signals2D = sweep2D_soft_hard_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        xs=1e3 * As,
        ys=fpts,
        xlabel="Flux (mA)",
        ylabel="Frequency (MHz)",
        updateCfg=updateCfg,
        signal2real=qub_signals2reals,
    )

    # get the actual frequency points
    fpts: ndarray = prog.get_pulse_param("qub_pulse", "freq", as_array=True)

    return As, fpts, signals2D
