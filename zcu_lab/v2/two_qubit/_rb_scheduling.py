"""Shared per-qubit pulse timing and virtual frames for two-qubit RB."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from zcu_tools.program.v2 import (
    Pulse,
    TimelineBuilder,
    TimelineProgramV2,
    TProcMemoryError,
)

from ._cz_phase import CZPhasePair
from ._drive import QubitDriveCfg
from ._rb_sequence import RBArm, TwoQubitGate

_QUBIT_TIMELINES = ("rb_qubit1_time", "rb_qubit2_time")


def validate_x_pulses(
    x90: tuple[Pulse.Cfg, Pulse.Cfg],
    x180: tuple[Pulse.Cfg, Pulse.Cfg],
    drive: QubitDriveCfg,
) -> None:
    """Check the Qubit1/Qubit2 X pulses in each pair against the corresponding CZ line.

    Raise ValueError if a pulse uses another generator or its phase is not a
    finite scalar in degrees. Leave the supplied cfgs unchanged.
    """
    for role, line, half, full in zip(
        ("Qubit1", "Qubit2"), (drive.qubit1, drive.qubit2), x90, x180, strict=True
    ):
        for gate, pulse in (("X90", half), ("X180", full)):
            if line.ch != pulse.ch:
                raise ValueError(f"{role} CZ and {gate} must use the same generator")
            if not isinstance(pulse.phase, (int, float)) or not math.isfinite(
                pulse.phase
            ):
                raise ValueError(f"{role} {gate} phase must be a finite scalar")


def emit_depth_sweep(
    builder: TimelineBuilder,
    arm: RBArm,
    *,
    x90: tuple[Pulse.Cfg, Pulse.Cfg],
    x180: tuple[Pulse.Cfg, Pulse.Cfg],
    drive: QubitDriveCfg,
    phase_pair: CZPhasePair,
) -> None:
    """Play one arm at every requested depth; loop index k plays arm.depths[k].

    Emit reset or herald before this call, and terminal readout after it.
    Each shot plays its Clifford prefix, takes its recovery and jumps once to
    readout (the deepest shot falls through). x90 and x180 each supply the
    calibrated Qubit1/Qubit2 pulses, in that order. drive is the two-line CZ pulse and
    phase_pair gives its two virtual-Z corrections. Timing follows emit_sequence.
    Reserve the depth loop, rb_skip_* and rb_readout labels, and RB timelines.
    Compilation errors propagate at build.
    """
    builder.add_loop("depth", len(arm.depths))
    builder.fork("default", into=list(_QUBIT_TIMELINES))
    player = _GatePlayer(builder, x90, x180, drive, phase_pair)

    *shallower, deepest = arm.depths
    done = 0
    for k, depth in enumerate(shallower):
        player.play(arm.cliffords[done:depth])
        done = depth
        # Other loop indices skip the recovery and keep playing the prefix.
        builder.cond_jump(label=f"rb_skip_{k}", arg1="depth", test="NZ", op="-", arg2=k)
        player.recover(arm.recoveries[depth])
        # ADR-0083: sync each path to its own end before merging at readout.
        builder.delay_by_timeline(timeline="default")
        builder.jump("rb_readout")
        builder.label(f"rb_skip_{k}")

    player.play(arm.cliffords[done:deepest])
    player.recover(arm.recoveries[deepest])
    builder.delay_by_timeline(timeline="default")
    builder.label("rb_readout")


def emit_sequence(
    builder: TimelineBuilder,
    arm: RBArm,
    depth: int,
    *,
    x90: tuple[Pulse.Cfg, Pulse.Cfg],
    x180: tuple[Pulse.Cfg, Pulse.Cfg],
    cz: tuple[QubitDriveCfg, CZPhasePair],
) -> None:
    """Play the prefix and recovery for one of arm.depths without any jumps.

    Start both qubits at the current default end with zero virtual frames.
    x90 and x180 each give the calibrated Qubit1/Qubit2 pulses in that order. Their
    phases are the configured degrees minus the role's frame. cz supplies the
    two-line drive and its virtual-Z corrections in degrees. Those corrections
    and Z90 advance the frames. CZ plays both lines at their configured phases.
    Clifford and CZ boundaries align both qubits.
    Leave the recovery's end on default for the next module. Reserve
    rb_qubit1_time, rb_qubit2_time and gate-indexed CZ timeline names. An unrequested
    depth raises KeyError; compilation errors propagate at build.
    """
    recovery = arm.recoveries[depth]
    builder.fork("default", into=list(_QUBIT_TIMELINES))
    drive, phase_pair = cz
    player = _GatePlayer(builder, x90, x180, drive, phase_pair)
    player.play(arm.cliffords[:depth])
    player.recover(recovery)


def compile_depth_sweep(builder: TimelineBuilder, arm: RBArm) -> TimelineProgramV2:
    """Build a complete depth-sweep program with the caller's readout attached.

    Raise ValueError with the arm, scan size and word counts on PMEM overflow.
    Other build errors propagate unchanged.
    """
    try:
        return builder.build()
    except TProcMemoryError as error:
        if error.memory != "pmem":
            raise
        name = "interleaved" if arm.interleaved else "reference"
        raise ValueError(
            f"RB seed {arm.seed} {name} arm needs {error.used} PMEM words for "
            f"{len(arm.depths)} depths up to {arm.depths[-1]}; the board has "
            f"{error.capacity}. Reduce the largest depth or the number of depths."
        ) from error


@dataclass
class _GatePlayer:
    """Lower gate words to two timelines; retain prefix frames between words."""

    builder: TimelineBuilder
    x90: tuple[Pulse.Cfg, Pulse.Cfg]
    x180: tuple[Pulse.Cfg, Pulse.Cfg]
    drive: QubitDriveCfg
    phase_pair: CZPhasePair
    frames: tuple[float, float] = (0.0, 0.0)
    next_gate: int = 0

    def play(self, cliffords: Sequence[tuple[TwoQubitGate, ...]]) -> None:
        """Play Cliffords in order, carrying forward each qubit's frame."""
        for gates in cliffords:
            self.frames = self._play_clifford(gates)

    def recover(self, gates: tuple[TwoQubitGate, ...]) -> None:
        """Play recovery from the current frames without changing them."""
        self._play_clifford(gates)

    def _play_clifford(self, gates: tuple[TwoQubitGate, ...]) -> tuple[float, float]:
        frames = list(self.frames)
        for gate in gates:
            index = self.next_gate
            self.next_gate += 1
            match gate:
                case TwoQubitGate.Z90_Q0 | TwoQubitGate.Z90_Q1:
                    role = 0 if gate is TwoQubitGate.Z90_Q0 else 1
                    frames[role] = (frames[role] + 90.0) % 360.0
                case TwoQubitGate.X90_Q0 | TwoQubitGate.X90_Q1:
                    role = 0 if gate is TwoQubitGate.X90_Q0 else 1
                    self._play_x(
                        f"gate_{index}_x90", self.x90[role], role, frames[role]
                    )
                case TwoQubitGate.X180_Q0 | TwoQubitGate.X180_Q1:
                    role = 0 if gate is TwoQubitGate.X180_Q0 else 1
                    self._play_x(
                        f"gate_{index}_x180", self.x180[role], role, frames[role]
                    )
                case TwoQubitGate.CZ:
                    self._play_cz(index)
                    frames[0] = (
                        frames[0] + self.phase_pair.correction_qubit1_deg
                    ) % 360.0
                    frames[1] = (
                        frames[1] + self.phase_pair.correction_qubit2_deg
                    ) % 360.0
        self._align_qubits()
        return frames[0], frames[1]

    def _play_x(self, name: str, pulse: Pulse.Cfg, role: int, frame: float) -> None:
        phase = (float(pulse.phase) - frame) % 360.0
        self.builder.emit(
            Pulse(f"{name}_f{role + 1}", pulse, phase=phase),
            timeline=_QUBIT_TIMELINES[role],
        )

    def _play_cz(self, index: int) -> None:
        """Align both qubits, play both CZ lines together, restart at the CZ end."""
        line_qubit1, line_qubit2 = (
            f"gate_{index}_cz_qubit1_time",
            f"gate_{index}_cz_qubit2_time",
        )
        self.builder.join(*_QUBIT_TIMELINES, into="default")
        self.builder.fork("default", into=[line_qubit1, line_qubit2])
        self.builder.emit(
            Pulse(f"gate_{index}_cz_qubit1", self.drive.qubit1), timeline=line_qubit1
        )
        self.builder.emit(
            Pulse(f"gate_{index}_cz_qubit2", self.drive.qubit2), timeline=line_qubit2
        )
        self.builder.join(line_qubit1, line_qubit2, into="default")
        for timeline in _QUBIT_TIMELINES:
            self.builder.join("default", into=timeline)

    def _align_qubits(self) -> None:
        """Move default and both qubit timelines to the later qubit end."""
        self.builder.join(*_QUBIT_TIMELINES, into="default")
        for timeline in _QUBIT_TIMELINES:
            self.builder.join("default", into=timeline)
