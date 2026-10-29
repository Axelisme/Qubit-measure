"""Control Module adapters measured through the public compiler/timing seam."""

from collections.abc import Callable, Sequence

import pytest
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modules import (
    Branch,
    Module,
    Pulse,
    PulseCfg,
    TableLengthPulse,
)
from zcu_tools.program.v2.modules.delay import Join
from zcu_tools.program.v2.modules.dmem import LoadValue
from zcu_tools.program.v2.modules.waveform import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)
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
    data_registers: int = 16,
) -> TimelineProgramV2:
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            if selector is not None:
                self.add_loop(selector, 3)
            for module in modules:
                self.declare(module)

        def _body(self, cfg):
            body(self)

    soccfg = make_mock_soccfg(3, 1)
    soccfg["tprocs"][0]["dreg_qty"] = data_registers
    return Program(soccfg, reps=2, initial_delay=0, relax_delay=0)


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
    other = _pulse("other", 1, 77)
    tail = _pulse("tail", 2)
    joined = Join([first, first], None, other)
    parent = "default" if timeline is None else timeline

    def bookkeeping_body(prog):
        if timeline is not None:
            prog.fork("default", into=parent)
        prog.delay_timeline(10 / 430.08, timeline=parent)
        prog.fork(parent, into=["left", "right"])
        prog.emit(first, timeline="left")
        prog.emit(first, timeline="left")
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
    bookkeeping = _program([first, other, tail], bookkeeping_body, selector=None)
    expected_waves = [[10, 53, 10, 97]] * 2
    assert [waves(s) for s in shots(bookkeeping)] == expected_waves
    assert [s.body_end_cycle - s.start_cycle for s in shots(bookkeeping)] == [125] * 2
    assert [s.end_cycle - s.start_cycle for s in shots(bookkeeping)] == [125] * 2

    emitted = _program([joined, tail], module_body, selector=None)
    assert [waves(s) for s in shots(emitted)] == expected_waves
    assert [s.body_end_cycle - s.start_cycle for s in shots(emitted)] == [125] * 2
    assert [s.end_cycle - s.start_cycle for s in shots(emitted)] == [125] * 2
    emitted.compile()
    assert [waves(s) for s in shots(emitted)] == expected_waves


def _table_pulse(
    name: str,
    ch: int,
    *,
    flat: bool = False,
    idx_reg: str = "scan",
    count: int = 3,
) -> TableLengthPulse:
    lengths = (
        [55 / 384, 80 / 384, 105 / 384] if flat else [39 / 384, 64 / 384, 89 / 384]
    )
    waveform = (
        FlatTopWaveformCfg(
            length=lengths[0],
            raise_waveform=GaussWaveformCfg(length=16 / 384, sigma=4 / 384),
        )
        if flat
        else ConstWaveformCfg(length=lengths[0])
    )
    return TableLengthPulse(
        name,
        PulseCfg(
            ch=ch,
            nqz=1,
            freq=1000,
            gain=0.5,
            waveform=waveform,
            pre_delay=0.01,
            post_delay=0.0005,
        ),
        lengths=(lengths if ch == 0 else list(reversed(lengths)))[:count],
        idx_reg=idx_reg,
    )


@pytest.mark.parametrize("timeline", [None, "parallel"])
@pytest.mark.parametrize("flat", [False, True])
def test_module_join_selected_lengths_keep_pointwise_maximum_and_runtime_origin(
    timeline, flat
):
    left, right = (
        _table_pulse("left", 0, flat=flat),
        _table_pulse("right", 1, flat=flat),
    )
    joined = Join([None, left], None, [right, None])
    origin = LoadValue("origin", [7, 7, 7], idx_reg="scan", val_reg="origin_cycles")
    tail = _pulse("tail", 2)
    parent = "default" if timeline is None else timeline

    def body(prog):
        if timeline is not None:
            prog.fork("default", into=parent)
        prog.emit(origin)
        prog.delay_timeline("origin_cycles", timeline=parent)
        prog.emit(joined, timeline=timeline)
        prog.emit(joined, timeline=timeline)
        prog.emit(tail, timeline=timeline)

    prog = _program([origin, joined, tail], body, data_registers=32)
    result = shots(prog)
    # pre rounds4; post=.21504. Const raw ends43.68/71.68/99.68
    # give per-child ceil48/76/104. Flat adds ramps16 gen clocks,
    # giving ceil66/94/122. Crossing maxima:104/76/104 or122/94/122.
    # Both emissions start at origin7 plus the accumulated pointwise maximum.
    starts = [133, 105, 133] if flat else [115, 87, 115]
    tails = [251, 195, 251] if flat else [215, 159, 215]
    copies = 6 if flat else 2  # Each flat pulse writes three contiguous segments.
    assert [waves(s) for s in result] == [
        [11] * copies + [start] * copies + [tail_start]
        for start, tail_start in zip(starts, tails, strict=True)
    ] * 2
    ends = [279, 223, 279] if flat else [243, 187, 243]
    assert [s.body_end_cycle - s.start_cycle for s in result] == ends * 2
    assert [s.end_cycle - s.start_cycle for s in result] == ends * 2
    assert prog.compile_datamem() is not None
    prog.compile()
    assert shots(prog) == result


def test_module_join_minimal_selected_lengths_use_default_register_profile():
    left, right = _table_pulse("left", 0), _table_pulse("right", 1)
    joined = Join(left, right)
    tail = _pulse("tail", 2)

    def body(prog):
        prog.emit(joined)
        prog.emit(tail)

    result = shots(_program([joined, tail], body))
    assert [waves(s) for s in result] == [[4, 4, 104], [4, 4, 76], [4, 4, 104]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [132, 104, 132] * 2
    assert [s.end_cycle - s.start_cycle for s in result] == [132, 104, 132] * 2


@pytest.mark.parametrize(
    "invalid, message",
    [
        ("mixed", "one TableLengthPulse per branch"),
        ("sequence", "one TableLengthPulse per branch"),
        ("same_branch", "one TableLengthPulse per branch"),
        ("generator", "independent generators"),
        ("index", "shared index and table size"),
        ("count", "shared index and table size"),
    ],
)
def test_module_join_retains_table_parallel_input_restrictions(invalid, message):
    left = _table_pulse("left", 0)
    right = _table_pulse(
        "right",
        0 if invalid == "generator" else 1,
        idx_reg="other" if invalid == "index" else "scan",
        count=2 if invalid == "count" else 3,
    )
    groups: list[Module | list[Module]] = [left, right]
    if invalid == "mixed":
        groups = [left, _pulse("scalar", 1)]
    elif invalid == "sequence":
        groups = [[left, _pulse("scalar", 2)], right]
    elif invalid == "same_branch":
        groups = [[left, right]]
    with pytest.raises(ValueError, match=message):
        Join(*groups)


def test_module_join_reference_advancing_child_preserves_parallel_origin():
    choice = Branch("scan", _pulse("a", 0), None, _pulse("long_a", 0, 50))
    other = _pulse("b", 1)
    joined = Join(choice, other)
    result = shots(_program([joined], lambda prog: prog.emit(joined)))
    # A closes its selected case at28/0/56. B belongs to another parallel
    # branch and must still start at0; the Join ends at28/28/56, not56/28/84.
    assert [waves(s) for s in result] == [[0, 0], [0], [0, 0]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [28, 28, 56] * 2
    assert [s.end_cycle - s.start_cycle for s in result] == [28, 28, 56] * 2
