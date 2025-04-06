from copy import deepcopy
from warnings import warn

import numpy as np
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.schedule.flux import set_flux
from zcu_tools.schedule.tools import sweep2param


def measure_singleshot(soc, soccfg, cfg):
    """
    Perform single-shot measurements on a qubit to distinguish between quantum states.

    This function configures and executes a measurement that captures individual qubit
    state readouts without averaging, allowing for state discrimination analysis.
    The function modifies the configuration to set up a sweep that alternates between
    measuring the ground state (no qubit pulse) and excited state (with pi pulse).

    Parameters
    ----------
    soc : object
        The socket object for communication with the hardware.
    soccfg : object
        The socket configuration object containing hardware settings.
    cfg : dict
        Configuration dictionary containing measurement settings.
        Required keys:
        - shots: Number of single-shot measurements to perform
        - dac: Dictionary with qub_pulse settings
        - dev: Dictionary with flux_dev and flux settings

    Returns
    -------
    np.ndarray
        Complex array of shape (shots, 2) containing I+jQ measurement results.
        The first dimension corresponds to different shot measurements,
        while the second dimension distinguishes between ground (0) and excited (1) states.

    Notes
    -----
    - The function modifies cfg to ensure single-shot operation:
      * soft_avgs is set to 1
      * reps is set to the number of shots
      * sweep is configured to alternate between no pulse (0 gain) and pi pulse
    - The measurement results are returned as complex numbers (I+jQ values)
    - Flux bias is set according to the provided configuration before measurement
    """
    cfg = deepcopy(cfg)  # avoid in-place modification

    if cfg.setdefault("soft_avgs", 1) != 1:
        warn("soft_avgs will be overwritten to 1 for singleshot measurement")

    if "reps" in cfg:
        warn("reps will be overwritten by singleshot measurement shots")
    cfg["reps"] = cfg["shots"]

    if "sweep" in cfg:
        warn("sweep will be overwritten by singleshot measurement")

    qub_pulse = cfg["dac"]["qub_pulse"]

    # append ge sweep to inner loop
    cfg["sweep"] = {"ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2}}

    # set with / without pi gain for qubit pulse
    qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    prog = TwoToneProgram(soccfg, deepcopy(cfg))
    prog.acquire(soc, progress=True)
    acc_buf = prog.acc_buf[0]
    avgiq = acc_buf / prog.get_time_axis(0)[-1]  # (reps, 2, 1, 2)
    i0, q0 = avgiq[..., 0, 0].T, avgiq[..., 0, 1].T  # (reps, 2)

    return np.array(i0 + 1j * q0)
