"""Control Module adapters measured through the public compiler/timing seam."""

from collections.abc import Callable, Sequence

import pytest
from qick.asm_v2 import QickParam, QickSweep1D
from zcu_tools.program.v2.macro.structure import SourcePulse
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modules import (
    Branch,
    Module,
    Pulse,
    PulseCfg,
    TableLengthPulse,
)
from zcu_tools.program.v2.modules.delay import Join, SoftDelay
from zcu_tools.program.v2.modules.dmem import LoadValue, ScanWith
from zcu_tools.program.v2.modules.readout import PulseReadout, PulseReadoutCfg
from zcu_tools.program.v2.modules.reset import BathResetCfg, TwoPulseResetCfg
from zcu_tools.program.v2.modules.waveform import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)
from zcu_tools.program.v2.timeline import TimelineProgramV2

from .timing_support import shots, waves


def _pulse_cfg(
    ch: int,
    fabric_cycles: int = 25,
    *,
    pre: float | QickParam = 0.0,
    post: float = 0.0,
) -> PulseCfg:
    return PulseCfg(
        ch=ch,
        nqz=1,
        freq=1000,
        gain=0.5,
        waveform=ConstWaveformCfg(length=fabric_cycles / 384),
        pre_delay=pre,
        post_delay=post,
    )


def _pulse(
    name: str,
    ch: int,
    fabric_cycles: int = 25,
    *,
    pre: float | QickParam = 0.0,
    post: float = 0.0,
) -> Pulse:
    return Pulse(name, _pulse_cfg(ch, fabric_cycles, pre=pre, post=post))


def _program(
    modules: Sequence[Module],
    body: Callable[[TimelineProgramV2], None],
    *,
    initialize: Callable[[TimelineProgramV2], None] | None = None,
    selector: str | None = "scan",
    data_registers: int = 16,
) -> TimelineProgramV2:
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            if initialize is not None:
                initialize(self)
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
    # case ends at 84/0/28. Each emission accounts the maximum 84 without
    # moving reference; the second starts at 84 and the tail at 168 on all points.
    assert [waves(s) for s in result] == [
        [0, 28, 84, 112, 168],
        [168],
        [0, 84, 168],
    ] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [196] * 6
    assert [s.end_cycle - s.start_cycle for s in result] == [196] * 6
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


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("timeline", [None, "parallel"])
def test_module_join_exposes_first_child_timing_for_each_occurrence(nested, timeline):
    first = _pulse("first", 0, pre=7 / 430.08)
    other = _pulse("other", 1, 50, pre=17 / 430.08)
    joined = Join(None, [first, first], other)
    if nested:
        joined = Join(None, joined, _pulse("third", 2))

    def body(prog):
        if timeline is not None:
            prog.fork("default", into=timeline)
        prog.emit(joined, timeline=timeline, tag="first")
        prog.emit(joined, timeline=timeline, tag="second")

    prog = _program([joined], body, selector=None)
    for _ in range(2):
        # First sequence ends at 2*(7+28)=70, sibling at 17+56=73.
        # The next occurrence starts at73, so its first event is at80.
        assert prog.list_time_params("first") == ["t"]
        assert prog.list_time_params("second") == ["t"]
        assert prog.get_time_param("first", "t") * 430.08 == pytest.approx(7)
        assert prog.get_time_param("second", "t") * 430.08 == pytest.approx(80)
        prog.compile()


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
    # Start ceils5; post=.21504. Const ends49/77/105.
    # Flat adds16 gen clocks, giving ends67/95/123.
    # Crossing maxima are105/77/105 or123/95/123.
    # Both emissions start at origin7 plus the accumulated pointwise maximum.
    starts = [135, 107, 135] if flat else [117, 89, 117]
    tails = [253, 197, 253] if flat else [217, 161, 217]
    copies = 6 if flat else 2  # Each flat pulse writes three contiguous segments.
    assert [waves(s) for s in result] == [
        [12] * copies + [start] * copies + [tail_start]
        for start, tail_start in zip(starts, tails, strict=True)
    ] * 2
    ends = [281, 225, 281] if flat else [245, 189, 245]
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
    assert [waves(s) for s in result] == [[5, 5, 105], [5, 5, 77], [5, 5, 105]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [133, 105, 133] * 2
    assert [s.end_cycle - s.start_cycle for s in result] == [133, 105, 133] * 2


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


def _declare_registers(*names: str) -> Callable[[TimelineProgramV2], None]:
    def declare(prog: TimelineProgramV2) -> None:
        for name in names:
            prog.add_reg(name)

    return declare


class _NativeSequence(SoftDelay):
    """Synthetic Module that lowers caller-supplied native operations."""

    def __init__(self, name, operations):
        super().__init__(name, 0)
        self.operations = operations

    def lower(self, prog, *, timeline, tag):
        self.operations(prog)


def _advancing_pulse(operation: str, scan: ScanWith) -> Pulse:
    class AdvancingPulse(Pulse):
        def lower(self, prog, *, timeline, tag):
            if operation == "delay":
                prog.delay(0.1)
            elif operation == "delay_auto":
                prog.delay_auto(0.1)
            elif operation == "sync":
                prog.delay_timeline(0.1, timeline=timeline)
                prog.sync(timeline=timeline)
            elif operation == "runtime_sync":
                prog.write_reg("duration", 10)
                prog.delay_timeline("duration", timeline=timeline)
                prog.sync(timeline=timeline)
            elif operation == "register":
                prog.write_reg("duration", 10)
                prog.delay_reg_auto("duration")
            elif operation == "repeat":
                with prog.repeat("inner", 2, advance_ref=True, timeline=timeline):
                    prog.delay_timeline(0.1, timeline=timeline)
            elif operation == "scan":
                prog.emit(scan, timeline=timeline)
            else:
                raise ValueError(f"unknown reference operation: {operation}")

    return AdvancingPulse("advancing", _pulse_cfg(0))


@pytest.mark.parametrize(
    "operation",
    ["delay", "delay_auto", "sync", "runtime_sync", "register", "repeat", "scan"],
)
@pytest.mark.parametrize("position", ["first", "last", "nested"])
def test_module_join_rejects_reference_advancing_child_at_build(operation, position):
    scan = ScanWith("scan_child", [7, 9], "value").add_content(_pulse("scan_pulse", 0))
    advancing = _advancing_pulse(operation, scan)
    other = _pulse("other", 1)
    joined = Join(other, advancing) if position == "last" else Join(advancing, other)
    if position == "nested":
        joined = Join(joined)
    with pytest.raises(
        ValueError, match="cannot advance reference; use delay_timeline instead"
    ):
        _program(
            [scan, joined],
            lambda prog: prog.emit(joined),
            initialize=_declare_registers("duration"),
            selector=None,
        )


@pytest.mark.parametrize("kind", ["two_pulse", "bath"])
@pytest.mark.parametrize(
    "operation",
    ["delay", "delay_auto", "sync", "runtime_sync", "register", "repeat", "scan"],
)
def test_reset_rejects_reference_advancing_pulse_lowering_at_build(kind, operation):
    scan = ScanWith("scan_child", [7, 9], "value").add_content(_pulse("scan_pulse", 0))
    first, second, pi2 = _pulse_cfg(0), _pulse_cfg(1), _pulse_cfg(0)
    child = _advancing_pulse(operation, scan)
    if kind == "two_pulse":
        reset = TwoPulseResetCfg(pulse1_cfg=first, pulse2_cfg=second).build("reset")
        reset.reset_pulse1 = child
    else:
        reset = BathResetCfg(
            cavity_tone_cfg=first, qubit_tone_cfg=second, pi2_cfg=pi2
        ).build("reset")
        reset.res_pulse = child
    with pytest.raises(
        ValueError, match="cannot advance reference; use delay_timeline instead"
    ):
        _program(
            [scan, reset],
            lambda prog: prog.emit(reset),
            initialize=_declare_registers("duration"),
            selector=None,
        )


@pytest.mark.parametrize("branch_first", [False, True])
def test_module_join_preserves_annotated_native_pulse_effects(branch_first):
    template = _pulse("native", 1, 50)
    native = _NativeSequence(
        "native_output",
        lambda prog: prog.append_macro(
            SourcePulse(ch=1, name=template.pulse_name(prog), t=0, source_name="native")
        ),
    )
    choice = Branch("scan", _pulse("a", 0), None, None)
    joined = Join(choice, native) if branch_first else Join(native, choice)
    prog = _program([template, joined], lambda prog: prog.emit(joined))
    # Native output stays at the parallel origin. It does not advance a Module
    # timeline; the branch accounts max(28, 0, 0)=28 on every selector point.
    assert [waves(s) for s in shots(prog)] == [[0, 0], [0], [0]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [28] * 6


@pytest.mark.parametrize("nested", [False, True])
def test_module_join_preserves_sequential_branch_children(nested):
    choice = Branch("scan", _pulse("a", 0), None, _pulse("long_a", 0, 50))
    first = Join(choice) if nested else choice
    joined = Join([first, choice], _pulse("b", 1))
    result = shots(_program([joined], lambda prog: prog.emit(joined)))
    # Each branch emission accounts 56; two sequential emissions end at 112.
    assert [waves(s) for s in result] == [[0, 56, 0], [0], [0, 56, 0]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [112] * 6
    assert [s.end_cycle - s.start_cycle for s in result] == [112] * 6


@pytest.mark.parametrize("empty", [False, True])
def test_module_join_cannot_bypass_wait_only_parent_via_internal_forks(empty):
    choice = Branch("scan", _pulse("a", 0), None, _pulse("long_a", 0, 50))
    joined = Join(None if empty else _pulse("b", 1))

    def body(prog):
        prog.fork("default", into="waiting")
        prog.delay_timeline(100 / 430.08, timeline="waiting")
        # A register Repeat leaves its unconsumed external end wait-only.
        prog.write_reg("duration", 1)
        with prog.repeat("runtime", "duration", advance_ref=True, range_hint=(1, 3)):
            prog.delay_timeline(10 / 430.08)
        prog.emit(joined, timeline="waiting")

    with pytest.raises(ValueError, match="timeline can only wait"):
        _program([choice, joined], body, initialize=_declare_registers("duration"))


@pytest.mark.parametrize("bad_case", [0, 1, 2])
def test_module_join_rejects_reverse_generator_fifo_in_every_branch_case(bad_case):
    cases: list[Module | None] = [None, None, None]
    cases[bad_case] = _pulse("a", 0)
    choice = Branch("scan", *cases)
    joined = Join(_pulse("late", 0, pre=100 / 430.08), choice)
    with pytest.raises(ValueError, match="channel 0 pulse overlap"):
        _program([joined], lambda prog: prog.emit(joined))


def test_module_join_rejects_reverse_generator_fifo_across_entire_sweep():
    choice = Branch("scan", *[_pulse("a", 0, pre=50 / 430.08)] * 3)
    late = _pulse("swept", 0, pre=QickSweep1D("scan", 0, 100 / 430.08))
    joined = Join(choice, late)
    # At the first point B ends28 before A50. Later points overlap or enqueue
    # B100 before A50. Compilation must prove all points, not sample the first.
    with pytest.raises(ValueError, match="channel 0 pulse overlap"):
        _program([joined], lambda prog: prog.emit(joined))


@pytest.mark.parametrize("register", [False, True])
@pytest.mark.parametrize("safe", [False, True])
def test_module_join_preserves_repeat_backedge_guards_without_padding(register, safe):
    a = _pulse("a", 0, 50, post=0 if safe else -28 / 430.08)
    choice = Branch("scan", a, a, a)
    joined = Join(choice, _pulse("b", 1))

    def body(prog):
        prog.write_reg("count", 2)
        with prog.repeat(
            "pair", "count" if register else 2, advance_ref=True, range_hint=(2, 3) if register else None
        ):
            prog.emit(joined)

    def compile_program():
        return _program([joined], body, initialize=_declare_registers("count"))

    if safe:
        result = shots(compile_program())
        assert [waves(s) for s in result] == [[0, 0, 56, 56]] * 6
        assert [s.body_end_cycle - s.start_cycle for s in result] == [112] * 6
        assert [s.end_cycle - s.start_cycle for s in result] == [112] * 6
    else:
        with pytest.raises(ValueError, match="channel 0 pulse overlap"):
            compile_program()


def test_module_join_preserves_known_readout_provenance_and_parallel_origin():
    readout = PulseReadout(
        "read",
        PulseReadoutCfg(
            gen_ch=1,
            nqz=1,
            freq=1000,
            gain=0.5,
            waveform=ConstWaveformCfg(length=0.3),
            pre_delay=0.05,
            post_delay=0.02,
            ro_ch=0,
            ro_length=0.4,
            trig_offset=0.1,
        ),
    )
    choice = Branch("scan", _pulse("a", 0), None, _pulse("long_a", 0, 50))
    joined = Join(readout, choice)
    result = shots(_program([joined], lambda prog: prog.emit(joined)))
    # Readout config0, playback22 and ADC44..216.2 plus post8.6016
    # finish at225, independent of the shorter Branch's28/0/56 advance.
    assert [waves(s) for s in result] == [[0, 22, 0], [0, 22], [0, 22, 0]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [225] * 6
    assert [s.end_cycle - s.start_cycle for s in result] == [225] * 6


def test_module_join_preserves_author_order_and_branch_parallel_origin():
    choice = Branch(
        "scan",
        _pulse("a", 0, pre=7 / 430.08),
        None,
        _pulse("long_a", 0, 50, pre=7 / 430.08),
    )
    other = _pulse("b", 1, pre=17 / 430.08)
    joined = Join(choice, other)
    result = shots(_program([joined], lambda prog: prog.emit(joined)))
    # A ends at max(35,0,63)=63 without moving reference. B starts at 17.
    # Event order is author order, even when the first child is a Branch.
    assert [waves(s) for s in result] == [[7, 17], [17], [7, 17]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [63] * 6
    assert [s.end_cycle - s.start_cycle for s in result] == [63] * 6
