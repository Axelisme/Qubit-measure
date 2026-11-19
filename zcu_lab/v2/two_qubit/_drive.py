"""Scalar two-line pulse configuration shared by two-qubit experiments."""

from __future__ import annotations

import math
from typing import Annotated, Self

from pydantic import model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.program.v2 import PulseCfg
from zcu_tools.program.v2.modules import ArbWaveformCfg


class DualLineCfg(ConfigBase):
    """Two ordinary scalar pulses on independent F1/F2 charge generators.

    f1/f2 explicitly carry each line's waveform, signed gain and phase (deg).
    Frequencies (MHz), envelope durations (us) and pre-delays (us) must match.
    Gains and relative phase remain independent; no ratio is optimized here.
    Post-delays may differ: parallel playback waits for both to finish.
    Frequency, gain, phase and delays must be finite scalars. Delays must be
    nonnegative and envelope duration positive. When both lines reference the
    same arb path, duration matches by construction and the file is validated
    when a run reads it, not when the cfg is restored.
    Runs apply typed Pulse overrides for hardware sweeps after validating this
    scalar baseline; each run writes its own parallel fork/emit/join sequence.
    No coupler AC line, persistent gate module kind or logical frame is implied.
    """

    f1: Annotated[
        PulseCfg,
        FieldMeta(
            label="F1 drive",
            description="F1 charge-line pulse。",
        ),
    ]
    f2: Annotated[
        PulseCfg,
        FieldMeta(
            label="F2 drive",
            description="F2 charge-line pulse。",
        ),
    ]

    @model_validator(mode="after")
    def validate_lines(self) -> Self:
        """Reject non-scalar/mismatched timing or frequency."""
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
        if self.f1.freq != self.f2.freq:
            raise ValueError("Dual-line frequencies must match")
        if self.f1.pre_delay != self.f2.pre_delay:
            raise ValueError("Dual-line pulses must share the same time origin")
        one, two = self.f1.waveform, self.f2.waveform
        if (
            isinstance(one, ArbWaveformCfg)
            and isinstance(two, ArbWaveformCfg)
            and one.path == two.path
        ):
            return self

        durations: list[float] = []
        for waveform in (one, two):
            duration = waveform.build("duration").length
            if (
                not isinstance(duration, (float, int))
                or not math.isfinite(duration)
                or duration <= 0
            ):
                raise ValueError(
                    "Dual-line envelope duration must be finite and positive"
                )
            durations.append(float(duration))
        if durations[0] != durations[1]:
            raise ValueError("Dual-line envelope durations must match")
        return self
