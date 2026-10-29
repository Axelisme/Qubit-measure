"""Runtime timeline ends observed through linked compiler timing."""

from collections.abc import Callable

from qick.asm_v2 import Macro, QickSweep1D

from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.timeline import TimelineProgramV2

from .timing_support import shots, waves


def program(body: Callable[[TimelineProgramV2], None], *, reps: int = 1):
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("scan", 3)
            for ch in range(2):
                self.declare_gen(ch, nqz=1)
                self.add_pulse(
                    ch, "p" + str(ch), style="const", freq=1000, phase=0, gain=0.5,
                    length=25 / 384,
                )  # 25 fabric cycles = 28 tProc cycles.
            self.append_macro(AllocateTime())

        def _body(self, cfg):
            body(self)

    return Program(make_mock_soccfg(2, 0), reps=reps, initial_delay=0, relax_delay=0)


def test_crossing_floor_places_at_pointwise_remaining_end_and_resets_each_shot():
    def body(prog):
        prog.delay_timeline(QickSweep1D("scan", 10 / 430.08, 30 / 430.08))
        prog.delay(20 / 430.08)
        prog.pulse(0, "p0", timeline="default")

    prog = program(body, reps=2)
    result = shots(prog)
    # End 10/20/30 minus reference advance 20 leaves 0/0/10.
    assert [waves(s) for s in result] == [[20], [20], [30]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [48, 48, 58] * 2
    prog.compile()
    assert shots(prog) == result


class AllocateTime(Macro):
    def preprocess(self, prog):
        prog.add_reg("duration")


def test_register_delay_adds_cycles_to_fork_without_advancing_reference():
    def body(prog):
        prog.write_reg("duration", 40)
        prog.fork("default", into="other")
        prog.delay_timeline("duration", timeline="other")
        prog.pulse(1, "p1", t=0)
        prog.pulse(0, "p0", timeline="other")

    result = shots(program(body, reps=2))
    assert [waves(s) for s in result] == [[0, 40]] * 6
    assert [s.body_end_cycle - s.start_cycle for s in result] == [68] * 6
    assert [s.end_cycle - s.start_cycle for s in result] == [68] * 6
