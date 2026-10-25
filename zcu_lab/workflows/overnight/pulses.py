"""Fixed calibrated pulses borrowed by overnight measurement stages."""

from dataclasses import dataclass

from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg


@dataclass(frozen=True, kw_only=True)
class FixedPulses:
    """Fixed pi drive and pulse-backed readout captured in the run plan.

    pi retains calibrated frequency, waveform, duration and gain. Echo derives
    pi/2 with half this gain. Stages borrow these values without modifying them.
    No timing policy, point identity or reporting services belong here.
    """

    pi: PulseCfg
    readout: PulseReadoutCfg
