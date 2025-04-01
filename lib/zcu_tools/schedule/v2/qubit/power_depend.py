import numpy as np
from zcu_tools import make_cfg
from zcu_tools.analysis import minus_background
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.schedule.tools import sweep2array, sweep2param
from zcu_tools.schedule.v2.template import sweep_hard_template


def signals2reals(signals: np.ndarray) -> np.ndarray:
    """
    Convert complex measurement signals to real-valued magnitudes by removing background noise.

    Args:
        signals (np.ndarray): Complex-valued measurement signals array.

    Returns:
        np.ndarray: Absolute values of the background-corrected signals.
    """
    return np.abs(minus_background(signals, axis=1))


def measure_qub_pdr_dep(soc, soccfg, cfg):
    """
    Measure qubit power dependency by performing a 2D sweep of pulse gain and frequency.

    This function prepares and executes a two-tone spectroscopy experiment that sweeps both
    the gain (power) and frequency of the qubit pulse. The gain sweep is set as the outer loop
    for efficient measurement.

    Args:
        soc: System-on-chip interface object for hardware control.
        soccfg: Configuration for the system-on-chip.
        cfg (dict): Configuration dictionary containing sweep parameters and pulse settings.
                    Must include 'sweep' with 'gain' and 'freq' parameters, and 'dac' with
                    'qub_pulse' settings.

    Returns:
        tuple: Three elements:
            - pdrs (np.ndarray): Actual pulse gain values used in the measurement.
            - fpts (np.ndarray): Actual frequency points used in the measurement.
            - signals (np.ndarray): Measurement signal data with dimensions matching the sweep.
    """
    cfg = make_cfg(cfg)  # prevent in-place modification

    # make sure gain is the outer loop
    if list(cfg["sweep"].keys())[0] == "freq":
        cfg["sweep"] = {
            "gain": cfg["sweep"]["gain"],
            "freq": cfg["sweep"]["freq"],
        }

    qub_pulse = cfg["dac"]["qub_pulse"]
    qub_pulse["gain"] = sweep2param("gain", cfg["sweep"]["gain"])
    qub_pulse["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

    pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted pulse gains
    fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points

    prog, signals = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        TwoToneProgram,
        ticks=(pdrs, fpts),
        progress=True,
        xlabel="Pulse Gain",
        ylabel="Frequency (MHz)",
        signal2real=signals2reals,
    )

    # get the actual pulse gains and frequency points
    pdrs = prog.get_pulse_param("qub_pulse", "gain", as_array=True)
    fpts = prog.get_pulse_param("qub_pulse", "freq", as_array=True)

    return pdrs, fpts, signals  # (pdrs, fpts)
