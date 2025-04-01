import numpy as np
from zcu_tools import make_cfg
from zcu_tools.analysis import minus_background, rescale
from zcu_tools.program.v2 import OneToneProgram
from zcu_tools.schedule.tools import map2adcfreq, sweep2array, sweep2param
from zcu_tools.schedule.v2.template import sweep2D_soft_hard_template


def signal2real(signals):
    """
    Process measurement signals by removing background and rescaling.

    Args:
        signals (ndarray): Raw measurement signals to process.

    Returns:
        ndarray: Processed signals after taking absolute value, removing background,
                and rescaling along axis 1.
    """
    return rescale(minus_background(np.abs(signals), axis=1), axis=1)


def measure_res_pdr_dep(soc, soccfg, cfg, dynamic_avg=False, gain_ref=0.1):
    """
    Measure the power dependency of a resonator by sweeping both power and frequency.

    This function performs a 2D sweep where power (gain) is swept in the outer loop
    and frequency in the inner loop. For each power level, a frequency sweep is performed
    to characterize the resonator response.

    Args:
        soc: System-on-chip object for hardware control.
        soccfg: SoC configuration object containing hardware settings.
        cfg (dict): Configuration dictionary containing measurement parameters:
            - dac.res_pulse: Resonator pulse settings
            - sweep.gain: Power/gain sweep settings
            - sweep.freq: Frequency sweep settings
            - reps: Number of measurement repetitions
            - adc.chs: ADC channels to use
        dynamic_avg (bool, optional): Whether to dynamically adjust avg count
                                     based on power level. Defaults to False.
        gain_ref (float, optional): Reference gain value for dynamic repetition
                                   adjustment. Defaults to 0.1.

    Returns:
        tuple: A 3-element tuple containing:
            - pdrs (ndarray): Power/gain values used in the sweep.
            - fpts (ndarray): Frequency points used in the sweep.
            - signals2D (ndarray): 2D array of measured signals with shape (len(pdrs), len(fpts)).
    """
    cfg = make_cfg(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    pdr_sweep = cfg["sweep"]["gain"]
    fpt_sweep = cfg["sweep"]["freq"]
    reps_ref = cfg["reps"]
    rounds_ref = cfg["rounds"]

    del cfg["sweep"]["gain"]  # use soft for loop here

    res_pulse["freq"] = sweep2param("freq", fpt_sweep)

    pdrs = sweep2array(pdr_sweep, allow_array=True)
    fpts = sweep2array(fpt_sweep)  # predicted frequency points
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    res_pulse["gain"] = pdrs[0]  # set initial power

    def updateCfg(cfg, i, pdr):
        """
        Update configuration for each step in the power sweep.

        Args:
            cfg (dict): Configuration dictionary to update.
            i (int): Current index in the power sweep.
            pdr (float): Current power/gain value.
        """
        cfg["dac"]["res_pulse"]["gain"] = pdr

        if dynamic_avg:
            dyn_factor = gain_ref / max(pdr, 1e-6)
            if dyn_factor > 1:
                # increase reps
                cfg["reps"] = int(reps_ref * dyn_factor)
                if cfg["reps"] > 100 * reps_ref:
                    cfg["reps"] = int(10 * reps_ref)
            else:
                # decrease rounds
                cfg["rounds"] = int(rounds_ref * dyn_factor)
                if cfg["rounds"] < 0.1 * rounds_ref:
                    cfg["rounds"] = int(0.1 * rounds_ref + 0.99)
                cfg["soft_avgs"] = cfg["rounds"]

    prog, signals2D = sweep2D_soft_hard_template(
        soc,
        soccfg,
        cfg,
        OneToneProgram,
        xs=pdrs,
        ys=fpts,
        xlabel="Power (a.u.)",
        ylabel="Frequency (MHz)",
        updateCfg=updateCfg,
        signal2real=signal2real,
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    return pdrs, fpts, signals2D
