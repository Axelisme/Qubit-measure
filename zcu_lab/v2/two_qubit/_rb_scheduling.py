"""Shared per-qubit timing and logical frames for both paired RB experiments."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from zcu_tools.program.v2 import (
    Pulse,
    TimelineBuilder,
    TimelineProgramV2,
    TProcMemoryError,
)

from ._cz_phase import CZPhasePair
from ._drive import DualLineCfg
from ._rb_sequence import RBSequence, TwoQubitGate

_RB_TIMELINES = ("rb_f1_time", "rb_f2_time")
_DEPTH_LOOP = "depth"


def emit_sequence(
    builder: TimelineBuilder,
    sequence: RBSequence,
    *,
    x90_f1: Pulse.Cfg,
    x90_f2: Pulse.Cfg,
    drive: DualLineCfg,
    phase_pair: CZPhasePair,
) -> None:
    """Append recovered RB with parallel X90s within each Clifford.

    Start both roles at builder's current default end with zero logical frames.
    Each role preserves its own gate order and delays. At each Clifford end,
    including recovery, align both timelines to the later end and leave that end
    on default for terminal readout. CZ also joins both ends, plays drive.f1/f2
    together, and restarts both at the later CZ end.
    Virtual Z and phase_pair corrections
    advance each role's frame; X90 phase is its configured phase minus that frame.
    Physical CZ phases remain fixed. Pulse frequencies are MHz, delays us and
    phases degrees. Neither sequence nor pulse configurations are modified.
    Reserve rb_f1_time/rb_f2_time and gate-indexed CZ timeline names in builder;
    compilation errors propagate, and unknown gates raise ValueError.
    """
    builder.fork("default", into=list(_RB_TIMELINES))
    playback = _CliffordPlayback(builder, (x90_f1, x90_f2), drive, phase_pair)
    frames = (0.0, 0.0)
    for block in sequence.clifford_blocks:
        frames = playback.emit_clifford(block, frames)


@dataclass
class _CliffordPlayback:
    """Shared pulse scheduling and unique gate names, independent of frame snapshots."""

    builder: TimelineBuilder
    pulses: tuple[Pulse.Cfg, Pulse.Cfg]
    drive: DualLineCfg
    phase_pair: CZPhasePair
    next_gate: int = 0

    def emit_clifford(
        self, block: tuple[TwoQubitGate, ...], initial_frames: tuple[float, float]
    ) -> tuple[float, float]:
        """Play one gate block, align both qubits, and return updated frame degrees."""
        builder, pulses, drive, phase_pair = (
            self.builder,
            self.pulses,
            self.drive,
            self.phase_pair,
        )
        frames = list(initial_frames)
        timelines = _RB_TIMELINES
        for index, gate in enumerate(block, start=self.next_gate):
            if gate in (TwoQubitGate.Z90_Q0, TwoQubitGate.Z90_Q1):
                role = 0 if gate == TwoQubitGate.Z90_Q0 else 1
                frames[role] = (frames[role] + 90.0) % 360.0
            elif gate in (TwoQubitGate.X90_Q0, TwoQubitGate.X90_Q1):
                role = 0 if gate == TwoQubitGate.X90_Q0 else 1
                pulse = pulses[role]
                builder.emit(
                    Pulse(
                        f"gate_{index}_x90_f{role + 1}",
                        pulse,
                        phase=(float(pulse.phase) - frames[role]) % 360.0,
                    ),
                    timeline=timelines[role],
                )
            elif gate == TwoQubitGate.CZ:
                builder.join(*timelines, into="default")
                builder.fork(
                    "default",
                    into=[f"gate_{index}_cz_f1_time", f"gate_{index}_cz_f2_time"],
                )
                builder.emit(
                    Pulse(f"gate_{index}_cz_f1", drive.f1),
                    timeline=f"gate_{index}_cz_f1_time",
                )
                builder.emit(
                    Pulse(f"gate_{index}_cz_f2", drive.f2),
                    timeline=f"gate_{index}_cz_f2_time",
                )
                builder.join(
                    f"gate_{index}_cz_f1_time",
                    f"gate_{index}_cz_f2_time",
                    into="default",
                )
                for timeline in timelines:
                    builder.join("default", into=timeline)
                frames[0] = (frames[0] + phase_pair.correction_f1_deg) % 360.0
                frames[1] = (frames[1] + phase_pair.correction_f2_deg) % 360.0
            else:
                raise ValueError(f"Unknown two-qubit gate: {gate!r}")
        self.next_gate += len(block)
        builder.join(*timelines, into="default")
        for timeline in timelines:
            builder.join("default", into=timeline)
        return frames[0], frames[1]


def emit_depth_sweep(
    builder: TimelineBuilder,
    sequences: Sequence[RBSequence],
    *,
    x90_f1: Pulse.Cfg,
    x90_f2: Pulse.Cfg,
    drive: DualLineCfg,
    phase_pair: CZPhasePair,
) -> None:
    """Append a board depth loop over a nonempty same-seed, same-arm chunk.

    Supply one sequence per depth, in increasing order. The longest contains
    every shorter random prefix. Each loop index plays its prefix and recovery
    and reaches the common exit at its own end. Emit reset and herald before
    this call and terminal readout after it. Pulse scheduling is as in
    emit_sequence. Reserve the depth loop, rb_recovery_* and rb_terminal labels,
    and emit_sequence's timeline names. Compilation errors propagate at build.
    """
    builder.add_loop(_DEPTH_LOOP, len(sequences))
    builder.fork("default", into=list(_RB_TIMELINES))
    playback = _CliffordPlayback(builder, (x90_f1, x90_f2), drive, phase_pair)
    checkpoints = _emit_prefix(playback, sequences)
    for point, (sequence, frames) in enumerate(
        zip(sequences, checkpoints, strict=True)
    ):
        builder.label(f"rb_recovery_{point}")
        playback.emit_clifford(sequence.recovery_gates, frames)
        # ADR-0083: sync each path before the label merges its timeline ends.
        builder.delay_by_timeline(timeline="default")
        if point < len(sequences) - 1:
            builder.jump("rb_terminal")
    builder.label("rb_terminal")


def _emit_prefix(
    playback: _CliffordPlayback, sequences: Sequence[RBSequence]
) -> list[tuple[float, float]]:
    frames = (0.0, 0.0)
    checkpoints = []
    start = 0
    last = len(sequences) - 1
    prefix = sequences[last].body_blocks
    for point, sequence in enumerate(sequences):
        end = len(sequence.body_blocks)
        for gates in prefix[start:end]:
            frames = playback.emit_clifford(gates, frames)
        start = end
        checkpoints.append(frames)
        if point < last:
            playback.builder.cond_jump(
                label=f"rb_recovery_{point}",
                arg1=_DEPTH_LOOP,
                test="Z",
                op="-",
                arg2=point,
            )
    if last > 0:
        playback.builder.jump(f"rb_recovery_{last}")
    return checkpoints


def build_depth_chunks(
    sequences: Sequence[RBSequence],
    build: Callable[[Sequence[RBSequence]], TimelineProgramV2],
) -> list[tuple[slice, TimelineProgramV2]]:
    """Compile the contiguous depths for one seed and arm, in ascending order.

    build records and compiles a fresh program for its nonempty sequence chunk.
    Each returned slice addresses that program's depths in the input sequence.
    Bisect only PMEM overflow, with the extra point in the first half. Other
    errors and singleton overflow propagate unchanged. All builds finish before
    the caller begins acquisition.
    """
    chunks: list[tuple[slice, TimelineProgramV2]] = []

    def compile_range(start: int, stop: int) -> None:
        try:
            program = build(sequences[start:stop])
        except TProcMemoryError as error:
            if error.memory != "pmem" or stop - start == 1:
                raise
            middle = start + (stop - start + 1) // 2
        else:
            chunks.append((slice(start, stop), program))
            return
        # Retry outside the handler so later failures do not chain to this overflow.
        compile_range(start, middle)
        compile_range(middle, stop)

    compile_range(0, len(sequences))
    return chunks
