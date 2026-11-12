"""Validate named drive tones and participant order for mux measurements."""

import math
from collections.abc import Sequence
from typing import Annotated, Protocol

from pydantic import Field
from qick.asm_v2 import QickParam
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.program.v2 import PulseCfg
from zcu_tools.program.v2.modules import MuxReadoutCfg


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
        if drive.waveform.style != "arb" and isinstance(
            drive.waveform.length, QickParam
        ):
            raise ValueError("drive.waveform.length must be fixed")


def validate_mux_tones(
    cfg: MuxAcquisition,
    tones: Sequence[MuxDriveToneCfg],
    participants: Sequence[int],
) -> None:
    """Validate full names/readout dimensions and active drives/channels.

    cfg provides shared positive reps/rounds and finite nonnegative delays in us.
    tones must match the full numeric readout length. Each active drive channel must
    differ from readout.gen_ch. Raise ValueError with original
    indices/actual values; never read bindings or change inputs. No sweep is needed.
    """
    # Check complete readout dimensions before indexing full tones.
    if len(tones) != len(cfg.readout.tones):
        raise ValueError(
            f"tones length={len(tones)}; full readout count={len(cfg.readout.tones)}"
        )
    validate_participants([tone.name for tone in tones], participants)

    # Only participants consume drives; readout still contains every tone.
    for i in participants:
        tone = tones[i]
        tone.validate_tone()
        drive = tone.require_drive()
        if drive.ch == cfg.readout.gen_ch:
            raise ValueError(
                f"tones[{i}].drive.ch={drive.ch}: drive and readout require different generators"
            )


def validate_mux_scan(
    cfg: MuxAcquisition,
    tones: Sequence[MuxDriveToneCfg],
    participants: Sequence[int],
    expts: int,
) -> None:
    """Validate fixed drives plus independent pulse ranges and a shared count.

    cfg, tones and participants obey validate_mux_tones. expts is a positive
    integer shared by all tones. Each leaf declares its physical range field
    using ToneRangeCfg, which owns finite endpoints. Raise without mutation.
    """
    validate_mux_tones(cfg, tones, participants)
    if type(expts) is not int or expts <= 0:
        raise ValueError("expts must be a positive integer")


def validate_participants(names: Sequence[str], participants: Sequence[int]) -> None:
    """Check complete names and nonempty unique original indices, preserving order.

    names contains the full captured identity list. participants indexes that
    list directly, not a selected subset. Negative, bool and duplicate values
    raise ValueError with the participant input position. Consumers index the
    complete list directly and reject positive out-of-range indices there.
    """
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("tones names must be nonempty and unique")
    if not participants:
        raise ValueError("participants must be nonempty")
    seen: set[int] = set()
    for position, index in enumerate(participants):
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError(
                f"participants[{position}]={index!r}: expected unique nonnegative original tone index"
            )
        seen.add(index)
