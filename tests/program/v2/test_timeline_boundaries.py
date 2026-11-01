"""Public compiler boundaries and independent scheduled-cycle expectations."""

from tests.program._profiles import load_soccfg
from zcu_tools.program.v2 import ir

from zcu_tools.program.v2.timeline import TimelineProgramV2


def test_body_layout_tracks_reference_and_native_shell_across_shots():
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("scan", 2)

        def _body(self, cfg):
            self.delay(10 / 430.08)

    prog = Program(load_soccfg(0, 0), reps=2, initial_delay=0, relax_delay=0)
    assert prog.timing_layout.loops == (
        ir.TimingLoop("reps", 2),
        ir.TimingLoop("scan", 2),
    )
    result = ir.analyze_timing(
        ir.CompiledTimingInput(prog.prog_list, prog.labels, prog.timing_layout)
    )
    assert [
        (s.loop_indices, s.start_cycle, s.body_end_cycle, s.end_cycle)
        for s in result.shots
    ] == [
        ((0, 0), 0, 10, 10),
        ((0, 1), 10, 20, 20),
        ((1, 0), 20, 30, 30),
        ((1, 1), 30, 40, 40),
    ]
