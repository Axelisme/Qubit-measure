"""Native timing uses MyProgramV2's shared compiler after the base correction."""

import numpy as np
import pytest
from qick.asm_v2 import AsmV2, AveragerProgramV2
from zcu_tools.program.v2.base import MyProgramV2
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.timeline import TimelineProgramV2


def _initialize(prog):
    for ch in range(2):
        prog.declare_gen(ch, nqz=1)
        prog.add_pulse(
            ch, f"p{ch}", style="const", freq=1000, phase=0, gain=0.5, length=0.1
        )
    prog.declare_readout(0, length=0.4)
    prog.add_readoutconfig(0, "ro", freq=100)
    before = AsmV2()
    before.nop()
    after = AsmV2()
    after.nop()
    prog.add_loop("scan", 3, exec_before=before, exec_after=after)
    prog.add_subroutine("sub", before)
    prog.send_readoutconfig(0, "ro")


def _body(prog, time=0):
    prog.pulse(0, "p0", t=time)
    prog.pulse(1, "p1", t="auto")
    prog.trigger(ros=[0], t=0.3)
    prog.wait_auto(0.1)
    prog.resync(0.05)
    prog.delay(0.2)
    prog.pulse(0, "dummypulse", t="auto")
    prog.trigger(ros=[0], t=None)
    prog.delay_auto(0.1)


def _compile(program_type, time=0, /, **kwargs):
    class Program(program_type):
        def _initialize(self, cfg):
            _initialize(self)

        def _body(self, cfg):
            _body(self, time)

        def _cleanup(self, cfg):
            self.call("sub")

    return Program(make_mock_soccfg(2, 1), reps=2, **kwargs)


def _assert_equal(actual: AveragerProgramV2, expected: AveragerProgramV2):
    assert actual.prog_list == expected.prog_list
    assert actual.labels == expected.labels
    assert [(name, count) for name, count, _, _ in actual.loops] == [
        (name, count) for name, count, _, _ in expected.loops
    ]
    assert actual.asm() == expected.asm()
    for memory in ("pmem", "wmem", "dmem"):
        np.testing.assert_array_equal(actual.binprog[memory], expected.binprog[memory])


@pytest.mark.parametrize("reps_innermost", [False, True])
@pytest.mark.parametrize(
    "shell",
    [
        {"initial_delay": 0.0, "relax_delay": 0.0},
        {"initial_delay": 0.2, "relax_delay": 0.4},
    ],
)
def test_unnamed_native_sequence_is_bitwise_equal(reps_innermost, shell):
    before = AsmV2()
    before.nop()
    after = AsmV2()
    after.nop()
    kwargs = dict(
        shell, reps_innermost=reps_innermost, before_reps=before, after_reps=after
    )
    native = _compile(MyProgramV2, **kwargs)
    timeline = _compile(TimelineProgramV2, **kwargs)
    _assert_equal(timeline, native)


def test_native_body_uses_the_shared_ir_pipeline():
    # Both native and module instructions must use the repo compiler, not bare QICK.
    expected = _compile(MyProgramV2)
    actual = _compile(TimelineProgramV2)
    _assert_equal(actual, expected)


def test_native_comparison_detects_a_changed_pulse_time():
    kwargs = {"initial_delay": 0.0, "relax_delay": 0.0}
    native = _compile(MyProgramV2, **kwargs)
    changed = _compile(TimelineProgramV2, 1, **kwargs)
    with pytest.raises(AssertionError):
        _assert_equal(changed, native)
