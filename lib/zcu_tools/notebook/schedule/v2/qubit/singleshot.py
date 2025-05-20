from copy import deepcopy
from typing import Tuple
from warnings import warn

import numpy as np
from tqdm.auto import tqdm
from zcu_tools.notebook.single_qubit.process import rotate2real
from zcu_tools.program.v2 import TwoToneProgram
from zcu_tools.tools import AsyncFunc, print_traceback

from ...flux import set_flux
from ...instant_show import InstantShowHist
from ...tools import format_sweep1D, sweep2array, sweep2param


def acquire_singleshot(prog, soc):
    prog.acquire(soc, progress=False)
    acc_buf = prog.get_acc_buf()[0]  # use this method to support proxy program
    avgiq = acc_buf / list(prog.ro_chs.values())[0]["length"]  # (reps, *sweep, 1, 2)
    i0, q0 = avgiq[..., 0, 0], avgiq[..., 0, 1]  # (reps, *sweep)
    signals = np.array(i0 + 1j * q0)  # (reps, *sweep)

    # swap axes to (*sweep, reps)
    signals = np.swapaxes(signals, 0, -1)

    return signals  # (reps, *sweep)


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
    return acquire_singleshot(prog, soc)


def measure_amprabi_singleshot(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    if cfg.setdefault("soft_avgs", 1) != 1:
        warn("soft_avgs will be overwritten to 1 for singleshot measurement")

    if "reps" in cfg:
        warn("reps will be overwritten by singleshot measurement shots")
    cfg["reps"] = cfg["shots"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
    pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted amplitudes

    del cfg["sweep"]  # use soft loop here

    cfg["dac"]["qub_pulse"]["gain"] = pdrs[0]

    signals = np.full((len(pdrs), cfg["shots"]), np.nan, dtype=complex)  # (pdr, shots)

    # set flux first
    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"], progress=True)

    with InstantShowHist("rototed I", "Count", title="Rabi SingleShot") as viewer:
        try:
            pdr_tqdm = tqdm(pdrs, desc="Power", smoothing=0)
            with AsyncFunc(viewer.update_show, include_idx=False) as async_draw:
                for i, pdr in enumerate(pdr_tqdm):
                    cfg["dac"]["qub_pulse"]["gain"] = pdr

                    prog = TwoToneProgram(soccfg, cfg)
                    signals[i] = acquire_singleshot(prog, soc)
                    async_draw(i, rotate2real(signals)[i].real)

        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, early stopping the program")
        except Exception:
            print("Error during measurement:")
            print_traceback()

    return pdrs, signals
