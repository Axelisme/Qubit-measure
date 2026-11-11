"""Assemble charge-drive settings without imposing an envelope or duration."""

from zcu_tools.program.v2 import Pulse
from zcu_tools.program.v2.modules.waveform import EnvelopeWaveformCfg


def drive_pulse(
    ch: int, freq: float, gain: float, waveform: EnvelopeWaveformCfg
) -> Pulse.Cfg:
    """Return a zone-1 drive at freq/MHz with signed gain and the given envelope.

    ch is the generator index. The envelope owns its length in us, including
    asset-derived arbitrary duration. No pre/post delay is added.
    """
    return Pulse.Cfg(ch=ch, nqz=1, freq=freq, gain=gain, waveform=waveform)
