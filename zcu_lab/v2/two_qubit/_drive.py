"""Scalar two-line pulse configuration shared by two-qubit experiments."""

from __future__ import annotations

import math
from typing import Annotated, Self

from pydantic import model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.program.v2 import PulseCfg
from zcu_tools.program.v2.modules import ArbWaveformCfg


class QubitDriveCfg(ConfigBase):
    """Aligned scalar charge-line pulses on the two named qubit positions.

    qubit1/qubit2 carry independent signed gain and phase in degrees. The
    pulses must have equal MHz frequency, envelope duration and pre-delay in
    µs; their post-delays may differ. The run validates referenced waveform assets.
    """

    qubit1: Annotated[
        PulseCfg,
        FieldMeta(label="Qubit1 drive", description="Qubit1 charge-line pulse."),
    ]
    qubit2: Annotated[
        PulseCfg,
        FieldMeta(label="Qubit2 drive", description="Qubit2 charge-line pulse."),
    ]

    @model_validator(mode="after")
    def validate_lines(self) -> Self:
        """Reject non-scalar/mismatched timing or frequency."""
        _validate_pulse_pair(self.qubit1, self.qubit2)
        return self


def _validate_pulse_pair(first: PulseCfg, second: PulseCfg) -> None:
    for pulse in (first, second):
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
    if first.freq != second.freq:
        raise ValueError("Dual-line frequencies must match")
    if first.pre_delay != second.pre_delay:
        raise ValueError("Dual-line pulses must share the same time origin")
    one, two = first.waveform, second.waveform
    if (
        isinstance(one, ArbWaveformCfg)
        and isinstance(two, ArbWaveformCfg)
        and one.path == two.path
    ):
        return

    durations: list[float] = []
    for waveform in (one, two):
        duration = waveform.build("duration").length
        if (
            not isinstance(duration, (float, int))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError("Dual-line envelope duration must be finite and positive")
        durations.append(float(duration))
    if durations[0] != durations[1]:
        raise ValueError("Dual-line envelope durations must match")
