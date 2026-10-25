"""Resolve this point's accepted physical pulses without building experiments."""

from dataclasses import dataclass

from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg

from zcu_lab.workflows.coherence_fluxdep.t1 import T1Plan
from zcu_lab.workflows.common.calibration import Calibrated, apply_readout_calibration


@dataclass(frozen=True)
class CalibratedPulses:
    """Detached pi and readout pulses for this point's T1 measurement.

    pi uses this point's accepted frequency, duration and gain, with the plan's
    routing/waveform. readout uses the accepted or retained readout calibration.
    No pi/2 is needed for T1. Consumers must copy pulses before changing them.
    """

    pi: PulseCfg
    readout: PulseReadoutCfg


def calibrated_pulses(plan: T1Plan, cal: Calibrated | None) -> CalibratedPulses | None:
    """Copy accepted pi/readout onto plan routing; return None without current pi.

    cal must come from current_pi_calibration, never a prior point's pi.
    None drive_gain retains the plan's gain. Inputs remain unchanged; native
    pulse construction errors propagate. No experiment cfg or history is used.
    """
    if cal is None:
        return None
    pi = plan.qub_pulse.model_copy(deep=True)
    pi.set_param("freq", cal.f01_mhz)
    pi.set_param("length", cal.pi_length_us)
    if cal.drive_gain is not None:
        pi.set_param("gain", cal.drive_gain)
    readout = plan.readout.model_copy(deep=True)
    apply_readout_calibration(readout, cal)
    return CalibratedPulses(pi=pi, readout=readout)
