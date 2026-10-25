"""Resolve this point's accepted physical pulses without building experiments."""

from dataclasses import dataclass

from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg

from zcu_lab.workflows.coherence_fluxdep.t1 import T1Plan
from zcu_lab.workflows.common.calibration import Calibrated, CalibratedPi2


@dataclass(frozen=True)
class CalibratedPulses:
    """Detached pi, optional pi2 and readout pulses for this point.

    pi uses this point's accepted frequency, duration and gain, with the plan's
    routing/waveform. readout uses the accepted or retained readout calibration.
    pi2 exists only with this point's trusted pair; T1/GE do not require it.
    Gains retain the requested values; the native cfg
    builder owns clipping and its diagnostics. Consumers copy before changing.
    """

    pi: PulseCfg
    readout: PulseReadoutCfg
    pi2: PulseCfg | None = None


def calibrated_pulses(plan: T1Plan, cal: Calibrated | None) -> CalibratedPulses | None:
    """Copy accepted pi/readout onto plan routing; return None without current pi.

    cal comes from current_pi_calibration or current_point_calibration, never
    a prior point's pi. Only CalibratedPi2 supplies the trusted pi2 pulse.
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
    readout.set_param("freq", cal.readout_freq_mhz)
    readout.set_param("gain", cal.readout_gain)
    pi2 = None
    if isinstance(cal, CalibratedPi2):
        pi2 = pi.model_copy(deep=True)
        pi2.set_param("length", cal.pi2_length_us)
        pi2_gain = (
            cal.pi2_drive_gain if cal.pi2_drive_gain is not None else cal.drive_gain
        )
        if pi2_gain is not None:
            pi2.set_param("gain", pi2_gain)
    return CalibratedPulses(pi=pi, readout=readout, pi2=pi2)
