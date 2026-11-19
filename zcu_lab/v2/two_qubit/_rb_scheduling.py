"""Shared per-qubit timing and logical frames for both paired RB experiments."""

from zcu_tools.program.v2 import Pulse, TimelineBuilder

from ._cz_phase import CZPhasePair
from ._drive import DualLineCfg
from ._rb_sequence import RBSequence, TwoQubitGate


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
    # ADR-0083: timeline ends are independent; join takes their max without moving reference.
    timelines = ("rb_f1_time", "rb_f2_time")
    builder.fork("default", into=list(timelines))
    frames = [0.0, 0.0]
    pulses = (x90_f1, x90_f2)
    offset = 0
    for block in sequence.clifford_blocks:
        for index, gate in enumerate(block, start=offset):
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
        offset += len(block)
        builder.join(*timelines, into="default")
        for timeline in timelines:
            builder.join("default", into=timeline)
