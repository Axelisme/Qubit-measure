"""Observe executed waveform words through the public linked timing analysis."""

from dataclasses import replace

from zcu_tools.program.v2 import TimelineProgramV2, ir


def analyze(prog: TimelineProgramV2) -> ir.TimingAnalysis:
    """Observe linked output and reference time for this compiled program."""
    return ir.analyze_timing(
        ir.CompiledTimingInput(
            prog.prog_list,
            prog.labels,
            prog.timing_layout,
            dmem=prog.compile_datamem(),
            wmem=ir.compiled_wmem(prog),
        )
    )


def relative_shot(shot: ir.ShotTiming) -> ir.ShotTiming:
    """Discard only linked addresses and the accumulated preceding-shot reference."""
    return replace(
        shot,
        start_cycle=0,
        body_end_cycle=shot.body_end_cycle - shot.start_cycle,
        end_cycle=shot.end_cycle - shot.start_cycle,
        events=tuple(
            replace(event, address=0, cycle=event.cycle - shot.start_cycle)
            for event in shot.events
        ),
    )


def assert_scalar_shots(
    observed: ir.TimingAnalysis, expected: list[ir.ShotTiming]
) -> None:
    """Compare scalar shots including complete ends and cumulative reference.

    Scalar loop coordinates may omit table/outer axes. Normalize only those
    coordinates and execution indexes, preserving every timing/output field.
    """
    reference = expected[0].start_cycle
    for shot, scalar in zip(observed.shots, expected, strict=True):
        assert relative_shot(shot) == replace(
            relative_shot(scalar), index=shot.index, loop_indices=shot.loop_indices
        )
        assert shot.start_cycle == reference
        reference += scalar.end_cycle - scalar.start_cycle
        assert shot.end_cycle == reference
    assert observed.end_cycle == reference


def executed_words(prog: TimelineProgramV2) -> list[tuple[ir.WaveWords, ...]]:
    """Return each shot\'s generator/ADC wave words in executed output order."""
    return [
        tuple(e.wave_words for e in shot.events if e.wave_words is not None)
        for shot in analyze(prog).shots
    ]
