"""Numeric full-tone cfg admission shared by mux frequency and Amp Rabi."""

import math
from collections.abc import Sequence
from typing import Self

from pydantic import StrictBool, model_validator
from qick.asm_v2 import QickParam
from zcu_tools.cfg_model import ConfigBase
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
    """One original tone's enabled flag, optional drive and independent sweep.

    active is a strict bool. Only active tones require drive. sweep is SweepCfg
    in the leaf's units: MHz, normalized gain, or requested total pulse length
    in us for Len Rabi. Len Rabi's prepared coordinates use the time-grid owner's
    quantized requested lengths, not compiled durations or hardware traces.
    Finite sweep bounds/step are required.
    Active drives must have fixed finite numeric frequency/gain/phase/delays and
    nonnegative delays. Waveform durations must be fixed, not hardware sweeps.
    Native pulse/waveform schemas own their shape and conversion rules.
    """

    active: StrictBool
    drive: PulseCfg | None
    sweep: SweepCfg

    @model_validator(mode="after")
    def validate_tone(self) -> Self:
        """Reject missing active drive or a nonfinite/swept active configuration."""
        if not all(
            math.isfinite(v)
            for v in (self.sweep.start, self.sweep.stop, self.sweep.step)
        ):
            raise ValueError("sweep start/stop/step must be finite")
        if not self.active:
            return self
        if self.drive is None:
            raise ValueError("active tone requires drive")
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
    cfg: ProgramV2Cfg, modules: MuxDriveModuleCfg, tones: Sequence[MuxToneCfg]
) -> None:
    """Validate full-list size, active zip counts and independent drive channels.

    cfg provides shared positive reps/rounds and finite nonnegative delays in us.
    tones must match the full numeric readout length, with at least one active.
    Only active sweeps must have equal point counts; their drive channels must
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
    active = [(i, t) for i, t in enumerate(tones) if t.active]
    if not active:
        raise ValueError("tones must include at least one active element")
    count = active[0][1].sweep.expts
    channels: dict[int, int] = {}
    for i, tone in active:
        if tone.sweep.expts != count:
            raise ValueError(
                f"tones[{i}].sweep.expts={tone.sweep.expts}; expected {count}"
            )
        drive = tone.drive
        if drive is None:
            raise ValueError(f"tones[{i}].drive is required")
        if drive.ch == modules.readout.gen_ch or drive.ch in channels:
            raise ValueError(
                f"tones[{i}].drive.ch={drive.ch}: requires independent generator; prior tone={channels.get(drive.ch)}; readout.gen_ch={modules.readout.gen_ch}"
            )
        channels[drive.ch] = i
