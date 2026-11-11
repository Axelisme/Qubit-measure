"""Actual activated detune shared by native Echo/Ramsey and workflows."""

import numpy as np
from qick import QickConfig
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import SweepCfg


def actual_detune(
    length_sweep: SweepCfg,
    *,
    detune: float,
    gen_ch: int,
    soccfg: QickConfig,
    time_scaler: float,
) -> float:
    """Return actual MHz detune after native time and phase quantization.

    length_sweep is the requested total delay in microseconds; detune is its
    nominal phase-ramp frequency in MHz. gen_ch selects the pi/2 generator and
    soccfg is the fixed QICK configuration used to compile this acquisition.
    time_scaler is required: Ramsey callers supply 1.0 for full-delay rounding,
    Echo callers 0.5 for their two half-delay waits. Native phase rounding always
    uses scaler=360*detune, independently of time_scaler.

    Zero nominal detune returns exactly 0.0. Otherwise return the nominal value
    multiplied by the mean phase-rounded/time-rounded delay ratio over positive
    delays. Actual and nominal values can differ even for the same requested
    grid. Preserve native quantization failures and NumPy diagnostics/NaN when
    no positive rounded delays exist; do not guess, clamp, retry or change cfg.
    No hardware, acquisition, fitting, plotting or file I/O occurs.
    """
    if detune == 0.0:
        return 0.0
    lengths = sweep2array(
        length_sweep, "time", {"soccfg": soccfg, "scaler": time_scaler}
    )
    phase_lengths = sweep2array(
        length_sweep,
        "phase",
        {"soccfg": soccfg, "gen_ch": gen_ch, "scaler": 360 * detune},
    )
    positive = lengths > 0
    ratio = np.mean(phase_lengths[positive] / lengths[positive]).item()
    return float(detune * ratio)
