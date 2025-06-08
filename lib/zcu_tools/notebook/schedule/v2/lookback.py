from typing import Tuple

import numpy as np
from tqdm.auto import tqdm
from zcu_tools.auto import make_cfg

# from ..instant_show import InstantShow1D
from zcu_tools.liveplot.jupyter import LivePlotter1D
from zcu_tools.program.v2 import OneToneProgram, TwoToneProgram

from ..flux import set_flux


def onetone_demimated(
    soc, soccfg, cfg, progress=True, qub_pulse=False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs a decimated one-tone or two-tone measurement.

    This function acquires a single-shot measurement using either OneToneProgram or
    TwoToneProgram depending on whether a qubit pulse is included.

    Parameters
    ----------
    soc : object
        Socket object for communication with the FPGA.
    soccfg : object
        Socket configuration object.
    cfg : dict
        Configuration dictionary for the measurement.
    progress : bool, optional
        Whether to show a progress bar during measurement, defaults to True.
    qub_pulse : bool, optional
        If True, uses TwoToneProgram (with qubit pulse); otherwise uses
        OneToneProgram (without qubit pulse), defaults to False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - Time axis (in microseconds)
        - Complex signal data (I + 1j*Q)
    """
    cfg = make_cfg(cfg, reps=1)

    prog = TwoToneProgram(soccfg, cfg) if qub_pulse else OneToneProgram(soccfg, cfg)
    IQlist = prog.acquire_decimated(soc, progress=progress)

    Ts = prog.get_time_axis(ro_index=0)
    Ts += cfg["adc"]["trig_offset"]

    return Ts, IQlist[0].dot([1, 1j])


def measure_lookback(
    soc, soccfg, cfg, progress=True, qub_pulse=False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs a lookback measurement, handling both short and long readout lengths.

    For readout lengths exceeding MAX_LEN (3.32 us), this function automatically splits
    the acquisition into multiple segments and combines them appropriately.

    Parameters
    ----------
    soc : object
        Socket object for communication with the FPGA.
    soccfg : object
        Socket configuration object.
    cfg : dict
        Configuration dictionary for the measurement, should contain:
        - 'dev': device settings with 'flux_dev' and 'flux' values
        - 'adc': ADC settings including 'ro_length', 'trig_offset', and 'chs'
    progress : bool, optional
        Whether to show a progress bar during measurement, defaults to True.
    qub_pulse : bool, optional
        If True, uses TwoToneProgram (with qubit pulse); otherwise uses
        OneToneProgram (without qubit pulse), defaults to False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - Time axis (in microseconds)
        - Complex signal data (I + 1j*Q)

    Notes
    -----
    For long measurements (ro_length > MAX_LEN), the function:
    1. Splits the acquisition into multiple segments
    2. Shows real-time data visualization using InstantShow1D
    3. Sorts and concatenates all measurements into a continuous data trace
    """
    cfg = make_cfg(cfg)

    set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"])

    MAX_LEN = 3.32  # us

    if cfg["adc"]["ro_length"] <= MAX_LEN:
        Ts, signals = onetone_demimated(
            soc, soccfg, cfg, progress=progress, qub_pulse=qub_pulse
        )
    else:
        # measure multiple times
        trig_offset = cfg["adc"]["trig_offset"]
        total_len = trig_offset + cfg["adc"]["ro_length"]
        cfg["adc"]["ro_length"] = MAX_LEN

        bar = tqdm(
            total=int((total_len - trig_offset) / MAX_LEN + 0.999),
            desc="Readout",
            smoothing=0,
            disable=not progress,
        )

        Ts = []
        signals = []
        with LivePlotter1D("Time (us)", "Amplitude", title="Readout") as viewer:
            while trig_offset < total_len:
                cfg["adc"]["trig_offset"] = trig_offset

                Ts_, singals_ = onetone_demimated(
                    soc, soccfg, cfg, progress=False, qub_pulse=qub_pulse
                )

                Ts.append(Ts_)
                signals.append(singals_)

                viewer.update(np.concatenate(Ts), np.concatenate(signals))

                trig_offset += MAX_LEN
                bar.update()

            bar.close()
            Ts = np.concatenate(Ts)
            signals = np.concatenate(signals)

            sort_idxs = np.argsort(Ts, kind="stable")
            Ts = Ts[sort_idxs]
            signals = signals[sort_idxs]

            viewer.update(Ts, np.abs(signals))

    return Ts, signals
