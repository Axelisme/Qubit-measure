"""Observe executed waveform words through the public linked timing analysis."""

from zcu_tools.program.v2 import TimelineProgramV2, ir


def executed_words(prog: TimelineProgramV2) -> list[tuple[ir.WaveWords, ...]]:
    """Return each shot\'s generator/ADC wave words in executed output order."""
    analysis = ir.analyze_timing(
        ir.CompiledTimingInput(
            prog.prog_list,
            prog.labels,
            prog.timing_layout,
            dmem=prog.compile_datamem(),
            wmem=ir.compiled_wmem(prog),
        )
    )
    return [
        tuple(e.wave_words for e in shot.events if e.wave_words is not None)
        for shot in analysis.shots
    ]
