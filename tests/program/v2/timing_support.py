"""Shared public compiled-timing handoff for timeline and emission tests."""

from qick.qick_asm import AbsQickProgram
from zcu_tools.program.v2 import ir
from zcu_tools.program.v2.timeline import TimelineProgramV2


def recompile(prog: AbsQickProgram) -> None:
    """Clear the public binary cache and rebuild through the nullable QICK seam."""
    prog.binprog = None
    prog.compile()


def shots(prog: TimelineProgramV2) -> tuple[ir.ShotTiming, ...]:
    """Analyze a compiled program using its linked layout and public data memory."""
    return ir.analyze_timing(
        ir.CompiledTimingInput(
            prog.prog_list, prog.labels, prog.timing_layout, dmem=prog.compile_datamem()
        )
    ).shots


def waves(shot: ir.ShotTiming) -> list[int]:
    """Return wave event offsets in tProc cycles from this shot's start."""
    return [e.cycle - shot.start_cycle for e in shot.events if e.kind == "wave"]
