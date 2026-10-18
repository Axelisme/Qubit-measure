"""Numeric full-tone cfg admission shared by mux frequency and Amp Rabi."""

import math
from collections.abc import Sequence
from typing import Annotated, Self

from pydantic import model_validator
from qick.asm_v2 import QickParam
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.program.v2 import ProgramV2Cfg, PulseCfg, ResetCfg, SweepCfg
from zcu_tools.program.v2.modules import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    MuxReadoutCfg,
)


class MuxDriveModuleCfg(ConfigBase):
    """Shared numeric modules: optional reset and full ordered mux readout.

    reset defaults to None; readout is a fixed native MuxReadoutCfg. Both use
    microsecond timing and leave hardware validation to the program owner.
    """

    reset: ResetCfg | None = None
    readout: MuxReadoutCfg


class MuxToneCfg(ConfigBase):
    """One named original tone's required drive and independent pulse sweep.

    name is its nonempty capture-generated qubit name; full-list admission
    rejects duplicates. drive is required even when not participating. Its MHz
    frequency, normalized gain, degree phase and us delays must be fixed finite
    values, with nonnegative delays and unswept waveform duration. sweep uses
    MHz, normalized gain or us according to the leaf, with finite bounds/step.
    Native pulse/waveform schemas own shape and clock conversion.
    """

    name: Annotated[str, FieldMeta(readonly=True)]
    drive: PulseCfg
    sweep: SweepCfg

    @model_validator(mode="after")
    def validate_tone(self) -> Self:
        """Reject a nonfinite sweep or a nonfinite/swept drive for every element."""
        if not all(
            math.isfinite(v)
            for v in (self.sweep.start, self.sweep.stop, self.sweep.step)
        ):
            raise ValueError("sweep start/stop/step must be finite")
        drive = self.drive
        for name in ("freq", "gain", "phase", "pre_delay", "post_delay"):
            value = getattr(drive, name)
            if isinstance(value, QickParam) or not math.isfinite(value):
                raise ValueError(f"drive.{name} must be fixed and finite")
        if drive.pre_delay < 0 or drive.post_delay < 0:
            raise ValueError("drive delays must be nonnegative")
        if isinstance(
            drive.waveform, (ConstWaveformCfg, FlatTopWaveformCfg)
        ) and isinstance(drive.waveform.length, QickParam):
            raise ValueError("drive.waveform.length must be fixed")
        return self


def validate_mux_scan(
    cfg: ProgramV2Cfg,
    modules: MuxDriveModuleCfg,
    tones: Sequence[MuxToneCfg],
    participants: Sequence[str],
) -> None:
    """Validate complete named tones, participants, zip counts and drive channels.

    cfg provides shared positive reps/rounds and finite nonnegative delays in us.
    tones must match the full numeric readout length. Every sweep must have
    equal point counts, including nonparticipants; every drive channel must
    be distinct and different from readout.gen_ch. Raise ValueError with original
    indices/actual values; never read bindings or change inputs.
    """
    if cfg.reps <= 0 or cfg.rounds <= 0:
        raise ValueError("reps and rounds must be positive")
    if any(not math.isfinite(v) or v < 0 for v in (cfg.initial_delay, cfg.relax_delay)):
        raise ValueError("initial_delay and relax_delay must be finite nonnegative us")
    if len(tones) != len(modules.readout.tones):
        raise ValueError(
            f"tones length={len(tones)}; full readout count={len(modules.readout.tones)}"
        )
    participant_indices([tone.name for tone in tones], participants)
    count = tones[0].sweep.expts
    channels: dict[int, int] = {}
    for i, tone in enumerate(tones):
        if tone.sweep.expts != count:
            raise ValueError(
                f"tones[{i}].sweep.expts={tone.sweep.expts}; expected {count}"
            )
        drive = tone.drive
        if drive.ch == modules.readout.gen_ch or drive.ch in channels:
            raise ValueError(
                f"tones[{i}].drive.ch={drive.ch}: requires independent generator; prior tone={channels.get(drive.ch)}; readout.gen_ch={modules.readout.gen_ch}"
            )
        channels[drive.ch] = i


def participant_indices(names: Sequence[str], participants: Sequence[str]) -> list[int]:
    """Resolve participant names to original positions, retaining caller order.

    names is the complete capture-ordered tone identity list. Both lists must
    contain nonempty names without duplicates; participants must be nonempty
    and contain only names in names. Raise ValueError with the offending name
    or position. Never sort, mutate inputs or consult a capture/library.
    """
    index_of: dict[str, int] = {}
    for index, name in enumerate(names):
        if not name or name in index_of:
            raise ValueError(
                f"tones[{index}].name={name!r}: empty or duplicate name; prior index={index_of.get(name)}"
            )
        index_of[name] = index
    if not participants:
        raise ValueError("participants must be nonempty")
    seen: set[str] = set()
    indices: list[int] = []
    for position, name in enumerate(participants):
        if name not in index_of or name in seen:
            raise ValueError(
                f"participants[{position}]={name!r}: unknown or duplicate name"
            )
        seen.add(name)
        indices.append(index_of[name])
    return indices
