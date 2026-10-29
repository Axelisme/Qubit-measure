"""Control Module adapters measured through the public compiler/timing seam."""

from collections.abc import Callable, Sequence

import pytest
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modules import Branch, Module, Pulse, PulseCfg
from zcu_tools.program.v2.modules.delay import Join
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.timeline import TimelineProgramV2

from .timing_support import shots, waves


def _pulse(name: str, ch: int, fabric_cycles: int = 25) -> Pulse:
    return Pulse(
        name,
        PulseCfg(
            ch=ch,
            nqz=1,
            freq=1000,
            gain=0.5,
            waveform=ConstWaveformCfg(length=fabric_cycles / 384),
        ),
    )


def _program(
    modules: Sequence[Module],
    body: Callable[[TimelineProgramV2], None],
    *,
    selector: str | None = "scan",
) -> TimelineProgramV2:
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            if selector is not None:
                self.add_loop(selector, 3)
            for module in modules:
                self.declare(module)

        def _body(self, cfg):
            body(self)

    return Program(make_mock_soccfg(3, 1), reps=2, initial_delay=0, relax_delay=0)


@pytest.mark.parametrize("timeline", [None, "drive"])
@pytest.mark.parametrize("explicit_selector", [False, True])
def test_module_branch_selects_ordered_cases_without_padding_and_reuses_declaration(
    timeline, explicit_selector
):
    short, long, tail = _pulse("short", 0), _pulse("long", 1, 50), _pulse("tail", 2)
    branch = (
        Branch("choice", [short, None, long], None, short, compare_by="scan")
        if explicit_selector
        else Branch("scan", [short, None, long], None, short)
    )

    def body(prog):
        if timeline is not None:
            prog.fork("default", into=timeline)
        prog.emit(branch, timeline=timeline, tag="first")
        prog.emit(branch, timeline=timeline, tag="second")
        prog.emit(tail, timeline=timeline)

    prog = _program([branch, short, tail], body)
    result = shots(prog)
    # 25/50 generator clocks are exactly 28/56 tProc cycles. Each selected
    # case closes at 84/0/28; two emissions advance by 168/0/56, with no padding.
    assert [waves(s) for s in result] == [
        [0, 28, 84, 112, 168],
        [0],
        [0, 28, 56],
    ] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [196, 28, 84] * 2
    assert [s.end_cycle - s.start_cycle for s in result] == [196, 28, 84] * 2
    prog.compile()
    assert shots(prog) == result


def test_module_branch_rejects_measurement_selector_instead_of_sweep():
    branch = Branch("choice", None, None, None, compare_by="feedback")
    with pytest.raises(ValueError, match="feedback.*sweep selector.*measurement"):
        _program([branch], lambda prog: prog.emit(branch), selector=None)


@pytest.mark.parametrize("timeline", [None, "parallel"])
def test_module_join_emits_parallel_sequences_while_program_join_only_merges_ends(
    timeline,
):
    first = _pulse("first", 0, 38)
    second = _pulse("second", 0, 38)
    other = _pulse("other", 1, 77)
    tail = _pulse("tail", 2)
    joined = Join([first, second], None, other)
    parent = "default" if timeline is None else timeline

    def bookkeeping_body(prog):
        if timeline is not None:
            prog.fork("default", into=parent)
        prog.delay_timeline(10 / 430.08, timeline=parent)
        prog.fork(parent, into=["left", "right"])
        prog.emit(first, timeline="left")
        prog.emit(second, timeline="left")
        prog.emit(other, timeline="right")
        prog.join("left", "right", into=parent)
        prog.emit(tail, timeline=parent)

    def module_body(prog):
        if timeline is not None:
            prog.fork("default", into=parent)
        prog.delay_timeline(10 / 430.08, timeline=parent)
        prog.emit(joined, timeline=timeline)
        prog.emit(tail, timeline=timeline)

    # At origin 10: two sequential 38-clock pulses each ceil(42.56)=43,
    # so left ends at 96. The parallel 77-clock pulse ceil(86.24)=87 ends
    # at 97. The tail starts at max(96,97)=97 and lasts 28 cycles.
    # Bookkeeping join adds no events; Module Join must emit its children.
    bookkeeping = _program(
        [first, second, other, tail], bookkeeping_body, selector=None
    )
    expected_waves = [[10, 53, 10, 97]] * 2
    assert [waves(s) for s in shots(bookkeeping)] == expected_waves
    assert [s.body_end_cycle - s.start_cycle for s in shots(bookkeeping)] == [125] * 2
    assert [s.end_cycle - s.start_cycle for s in shots(bookkeeping)] == [125] * 2

    emitted = _program([joined, tail], module_body, selector=None)
    assert [waves(s) for s in shots(emitted)] == expected_waves
    assert [s.body_end_cycle - s.start_cycle for s in shots(emitted)] == [125] * 2
    assert [s.end_cycle - s.start_cycle for s in shots(emitted)] == [125] * 2
