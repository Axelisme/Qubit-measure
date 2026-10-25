"""Ordinary two-line pulse configuration and parallel charge-line playback."""

from __future__ import annotations

import math
from typing import Annotated, Self

from pydantic import model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.program.v2 import Pulse, PulseCfg
from zcu_tools.program.v2.modules import Join


class DualLineCfg(ConfigBase):
    """Two ordinary scalar pulses on independent F1/F2 charge generators.

    f1/f2 explicitly carry each line's waveform, signed gain and phase (deg).
    Frequencies (MHz), envelope durations (us) and pre-delays (us) must match.
    Gains and relative phase remain independent; no ratio is optimized here.
    Post-delays may differ: parallel playback waits for both to finish.
    Frequency, gain, phase and delays must be finite scalars. Delays must be
    nonnegative and envelope duration positive.
    Host scans replace both frequencies together, never use QICK parameter sweeps.
    No coupler AC line, persistent gate module kind or logical frame is implied.
    """

    f1: Annotated[
        PulseCfg,
        FieldMeta(
            label="F1 drive",
            description="填入 F1 charge-line pulse：frequency MHz、signed gain a.u. within [-1, 1]、NQZ 與 envelope duration us。先做 F1 frequency 與 amplitude／length Rabi 取得 pulse 參考，高能階 drive 範圍及 CZ pulse slot 仍由操作者確認。",
        ),
    ]
    f2: Annotated[
        PulseCfg,
        FieldMeta(
            label="F2 drive",
            description="填入 F2 charge-line pulse：frequency MHz、signed gain a.u. within [-1, 1]、NQZ 與 envelope duration us。先做 F2 frequency 與 amplitude／length Rabi 取得 pulse 參考，高能階 drive 範圍及 CZ pulse slot 仍由操作者確認。",
        ),
    ]

    @model_validator(mode="after")
    def validate_lines(self) -> Self:
        """Reject non-scalar/mismatched timing or frequency and shared generators."""
        if self.f1.ch == self.f2.ch:
            raise ValueError("Dual-line drive requires independent charge generators")
        durations: list[float] = []
        for pulse in (self.f1, self.f2):
            for name, value in (
                ("freq", pulse.freq),
                ("gain", pulse.gain),
                ("phase", pulse.phase),
                ("pre_delay", pulse.pre_delay),
                ("post_delay", pulse.post_delay),
            ):
                if not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError(f"Dual-line {name} must be a finite scalar")
            if pulse.pre_delay < 0 or pulse.post_delay < 0:
                raise ValueError("Dual-line delays must be nonnegative")
            duration = pulse.waveform.build("duration").length
            if (
                not isinstance(duration, (float, int))
                or not math.isfinite(duration)
                or duration <= 0
            ):
                raise ValueError(
                    "Dual-line envelope duration must be finite and positive"
                )
            durations.append(float(duration))
        if self.f1.freq != self.f2.freq:
            raise ValueError("Dual-line frequencies must match")
        if durations[0] != durations[1]:
            raise ValueError("Dual-line envelope durations must match")
        if self.f1.pre_delay != self.f2.pre_delay:
            raise ValueError("Dual-line pulses must share the same time origin")
        return self


def dual_line_drive(
    drive: DualLineCfg, *, frequency: float, name: str = "drive"
) -> Join:
    """Build parallel ordinary pulses at a common finite MHz frequency.

    drive is a validated scalar pair; name prefixes its two program pulse names.
    Return a Join that starts both at one origin and blocks through both pulses,
    including their individual post-delays. Input pulses are never modified.
    Frequency scans preserve gain ratio, relative phase and both waveforms.
    Raise ValueError for nonfinite frequency; program/route failures propagate.
    """
    if not math.isfinite(frequency):
        raise ValueError("Common drive frequency must be finite")
    return Join(
        Pulse(f"{name}_f1", drive.f1.with_updates(freq=frequency)),
        Pulse(f"{name}_f2", drive.f2.with_updates(freq=frequency)),
    )
