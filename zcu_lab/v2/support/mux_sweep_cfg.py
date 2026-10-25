"""Validate named drive tones and participant order for mux measurements."""

import math
from collections.abc import Sequence
from typing import Annotated, Protocol

from pydantic import Field, ValidationError
from pydantic_core import PydanticCustomError
from qick.asm_v2 import QickParam
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.program.v2 import PulseCfg
from zcu_tools.program.v2.modules import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    MuxReadoutCfg,
)


class MuxAcquisition(Protocol):
    """Read shared numeric acquisition settings without a program cfg wrapper.

    Counts are positive averaging controls. Delays use finite nonnegative us.
    readout contains the complete original mux tone list, not the active subset.
    """

    @property
    def reps(self) -> int: ...

    @property
    def rounds(self) -> int: ...

    @property
    def initial_delay(self) -> float: ...

    @property
    def relax_delay(self) -> float: ...

    @property
    def readout(self) -> MuxReadoutCfg: ...


class ToneRangeCfg(ConfigBase):
    """Finite sweep endpoints; units are determined by the experiment.

    start and stop use MHz, gain in a.u. within [-1, 1] or us according to the leaf.
    The experiment supplies the shared point count, not individual tones.
    """

    start: float = Field(
        allow_inf_nan=False, description="First scan value in experiment units."
    )
    stop: float = Field(
        allow_inf_nan=False, description="Last scan value in experiment units."
    )


class MuxDriveToneCfg(ConfigBase):
    """One named original tone's nullable fixed drive, without a pulse sweep.

    name is its nonempty capture-generated qubit name; full-list admission
    rejects duplicates. drive is required only when participating. Its MHz
    frequency, gain in a.u. within [-1, 1], degree phase and us delays must be fixed finite
    values, with nonnegative delays and unswept waveform duration.
    Native pulse/waveform schemas own shape and clock conversion.
    """

    name: Annotated[str, FieldMeta(readonly=True)]
    drive: PulseCfg | None = None

    def require_drive(self) -> PulseCfg:
        """Return the active tone's drive, or raise ValueError naming this tone."""
        if self.drive is None:
            raise ValueError(f"{self.name}: active tone requires drive")
        return self.drive

    def validate_tone(self) -> None:
        """Reject a missing, nonfinite or swept drive when this tone participates."""
        drive = self.require_drive()
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


class MuxToneCfg(MuxDriveToneCfg):
    """One nullable drive and independent pulse-scan range, required when active.

    Inherited name and drive identify the original tone. sweep holds finite
    start/stop in MHz, gain in a.u. within [-1, 1] or us for twotone, amp or length scans.
    Coherence uses a shared delay instead and does not use this range schema.
    """

    sweep: ToneRangeCfg | None = None

    def require_sweep(self) -> ToneRangeCfg:
        """Return the active tone's bounds, or raise ValueError naming this tone."""
        if self.sweep is None:
            raise ValueError(f"{self.name}: active tone requires sweep")
        return self.sweep


def require_tone_value(value: object, path: tuple[str | int, ...]) -> None:
    """Reject None with a directed native cfg error asking to fill this setting.

    path identifies the original tone field, including enclosing frequency when
    needed. Non-null values retain their ordinary schema/scan validation.
    """
    if value is None:
        raise ValidationError.from_exception_data(
            "Mux tone",
            [
                {
                    "type": PydanticCustomError(
                        "required_tone", "填入 {setting}", {"setting": path[-1]}
                    ),
                    "loc": path,
                    "input": None,
                }
            ],
        )


def validate_mux_tones(
    cfg: MuxAcquisition,
    tones: Sequence[MuxDriveToneCfg],
    participants: Sequence[int],
) -> None:
    """Validate full names/readout dimensions and active drives/channels.

    cfg provides shared positive reps/rounds and finite nonnegative delays in us.
    tones must match the full numeric readout length. Each active drive channel must
    be distinct and different from readout.gen_ch. Raise ValueError with original
    indices/actual values; never read bindings or change inputs. No sweep is needed.
    """
    # Check shared averaging and readout dimensions before indexing full tones.
    if cfg.reps <= 0 or cfg.rounds <= 0:
        raise ValueError("reps and rounds must be positive")
    if any(not math.isfinite(v) or v < 0 for v in (cfg.initial_delay, cfg.relax_delay)):
        raise ValueError("initial_delay and relax_delay must be finite nonnegative us")
    if len(tones) != len(cfg.readout.tones):
        raise ValueError(
            f"tones length={len(tones)}; full readout count={len(cfg.readout.tones)}"
        )
    validate_participants([tone.name for tone in tones], participants)

    # Only participants consume drives; readout still contains every tone.
    channels: dict[int, int] = {}
    for i in participants:
        tone = tones[i]
        require_tone_value(tone.drive, ("tones", i, "drive"))
        tone.validate_tone()
        drive = tone.require_drive()
        if drive.ch == cfg.readout.gen_ch or drive.ch in channels:
            raise ValueError(
                f"tones[{i}].drive.ch={drive.ch}: requires independent generator; prior tone={channels.get(drive.ch)}; readout.gen_ch={cfg.readout.gen_ch}"
            )
        channels[drive.ch] = i


def validate_mux_scan(
    cfg: MuxAcquisition,
    tones: Sequence[MuxToneCfg],
    participants: Sequence[int],
    expts: int,
) -> None:
    """Validate fixed drives plus independent pulse ranges and a shared count.

    cfg, modules, tones and participants obey validate_mux_tones. Each tone has
    finite sweep start/stop in experiment units. expts is a positive integer
    shared by all tones. Raise ValueError without changing inputs.
    """
    validate_mux_tones(cfg, tones, participants)
    if type(expts) is not int or expts <= 0:
        raise ValueError("expts must be a positive integer")
    for i in participants:
        require_tone_value(tones[i].sweep, ("tones", i, "sweep"))
        sweep = tones[i].require_sweep()
        if not all(math.isfinite(v) for v in (sweep.start, sweep.stop)):
            raise ValueError(f"tones[{i}].sweep start/stop must be finite")


def validate_participants(names: Sequence[str], participants: Sequence[int]) -> None:
    """Check complete names and nonempty unique original indices, preserving order.

    names contains the full captured identity list. participants indexes that
    list directly, not a selected subset. Negative, bool, duplicate and out-of-
    range values raise ValueError with the participant input position.
    """
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("tones names must be nonempty and unique")
    if not participants:
        raise ValueError("participants must be nonempty")
    seen: set[int] = set()
    for position, index in enumerate(participants):
        if type(index) is not int or not 0 <= index < len(names) or index in seen:
            raise ValueError(
                f"participants[{position}]={index!r}: expected unique original tone index in [0, {len(names)})"
            )
        seen.add(index)
