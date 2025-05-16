import numpy as np
from zcu_tools.auto import make_cfg
from zcu_tools.notebook.analysis import minus_background
from zcu_tools.program.v2 import OneToneProgram

from ...tools import map2adcfreq, sweep2array, sweep2param
from ..template import sweep2D_soft_hard_template


def signal2real(signals):
    return minus_background(np.abs(signals), axis=1)


def measure_res_flux_dep(soc, soccfg, cfg):
    """
    Measures resonator frequency dependence on flux.

    This function performs a 2D sweep of resonator frequency vs flux to
    characterize the flux dependence of a resonator.

    Args:
        soc: System-on-chip object that controls the hardware.
        soccfg: Configuration object for the system-on-chip.
        cfg (dict): Configuration dictionary containing:
            - dac.res_pulse: Resonator pulse parameters.
            - sweep.freq: Frequency sweep parameters.
            - sweep.flux: Flux sweep parameters.
            - adc.chs: ADC channel array.
            - dev: Device configuration for flux control.

    Returns:
        tuple: A 3-tuple containing:
            - flxs (numpy.ndarray): Flux values used in the measurement.
            - fpts (numpy.ndarray): Frequency points used in the measurement.
            - signals2D (numpy.ndarray): 2D array of measurement results.
    """
    cfg = make_cfg(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    fpt_sweep = cfg["sweep"]["freq"]
    flx_sweep = cfg["sweep"]["flux"]

    del cfg["sweep"]["flux"]  # use soft for loop here

    res_pulse["freq"] = sweep2param("freq", fpt_sweep)

    flxs = sweep2array(flx_sweep, allow_array=True)
    fpts = sweep2array(fpt_sweep)  # predicted frequency points
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    cfg["dev"]["flux"] = flxs[0]  # set initial flux

    def updateCfg(cfg, i, flx):
        """
        Updates configuration with current flux value during the sweep.

        Args:
            cfg (dict): Configuration dictionary to update.
            i (int): Current index in the sweep.
            flx (float): Current flux value to set.
        """
        cfg["dev"]["flux"] = flx

    prog, signals2D = sweep2D_soft_hard_template(
        soc,
        soccfg,
        cfg,
        OneToneProgram,
        xs=flxs,
        ys=fpts,
        xlabel="Flux (a.u.)",
        ylabel="Frequency (MHz)",
        updateCfg=updateCfg,
        signal2real=signal2real,
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    return flxs, fpts, signals2D
