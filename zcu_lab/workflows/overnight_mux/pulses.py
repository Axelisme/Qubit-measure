"""Fixed mux pulses and their original readout routing."""

from dataclasses import dataclass

from zcu_tools.program.v2 import ResetCfg
from zcu_tools.program.v2.modules import MuxReadoutCfg, PulseCfg

from zcu_lab.v2.mux.t1.core import MuxT1ToneCfg


class OvernightMuxTone(MuxT1ToneCfg):
    """Original readout position: name, x180 drive and independent x90 pulse.

    Both pulses are required for participants by the native cores. Inactive
    positions may leave either pulse None. No pulse is inferred or calibrated.
    """

    pi2_pulse: PulseCfg | None = None


@dataclass(frozen=True, kw_only=True)
class FixedMuxPulses:
    """Fixed full tone order, readout, participant selection and optional reset.

    tones retains every original slot, including inactive ones. participants
    lists original tone indices in acquisition order. Each participant supplies
    calibrated pi and pi/2 pulses; readout keeps the full original tone order.
    reset is the shared native reset, or None. Stages do not modify these inputs.
    No timing policy, point identity or reporting services belong here.
    """

    tones: tuple[OvernightMuxTone, ...]
    readout: MuxReadoutCfg
    participants: tuple[int, ...]
    reset: ResetCfg | None
