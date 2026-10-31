"""Control Module adapters measured through the public compiler/timing seam."""

from collections.abc import Callable, Sequence

import pytest
from qick.asm_v2 import Delay as NativeDelay
from qick.asm_v2 import Macro, QickParam, QickSweep1D
from qick.asm_v2 import Pulse as NativePulse
from zcu_tools.program.v2.macro.meta import MetaMacro
from zcu_tools.program.v2.macro.structure import MarkedCloseLoop, SourcePulse
from zcu_tools.program.v2.macro.wmem import PatchWmemFromRegs
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
from zcu_tools.program.v2.modules.waveform import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)
from zcu_tools.program.v2.timeline import TimelineProgramV2

from .timing_support import shots, waves


def _pulse(
    name: str,
    ch: int,
    fabric_cycles: int = 25,
    *,
    pre: float | QickParam = 0.0,
    post: float = 0.0,
) -> Pulse:
    return Pulse(
        name,
        PulseCfg(
            ch=ch,
            nqz=1,
            freq=1000,
            gain=0.5,
            waveform=ConstWaveformCfg(length=fabric_cycles / 384),
            pre_delay=pre,
            post_delay=post,
        ),
    )


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


@pytest.mark.parametrize("nested", [False, True])
def test_module_join_tag_stays_with_author_first_child_after_exchange(nested):
    first = _pulse("first", 0, pre=7 / 430.08)
    scan = ScanWith("inner", [7, 9], "value").add_content(_pulse("scan_pulse", 0))
    other = _pulse("other", 1, 50, pre=17 / 430.08)
    advancing = Join([first, scan]) if nested else [first, scan]
    joined = Join(advancing, other)

    def body(prog):
        prog.emit(joined, tag="first")
        prog.delay_timeline(11 / 430.08)
        prog.emit(joined, tag="second")

    prog = _program([joined], body, selector=None)
    for _ in range(2):
        # Sibling emits first at17 but the tag belongs to the author-first
        # pulse at7. Scan entry syncs at35; two28-cycle iterations end at91.
        # The next emission starts11 cycles after that reference, at102.
        assert prog.list_time_params("first") == ["t"]
        assert prog.list_time_params("second") == ["t"]
        assert prog.get_time_param("first", "t") * 430.08 == pytest.approx(7)
        assert prog.get_time_param("second", "t") * 430.08 == pytest.approx(18)
        assert [waves(s) for s in shots(prog)] == [
            [17, 7, 35, 63, 119, 109, 137, 165]
        ] * 2
        assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [193] * 2
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


@pytest.mark.parametrize("position", ["first", "last", "nested"])
def test_module_join_scan_sequence_keeps_load_and_loop_order_at_parallel_origin(
    position,
):
    scan = ScanWith("inner", [7, 9], "value").add_content(_pulse("a", 0))
    other = _pulse("b", 1, 50)
    advancing = Join(None, scan) if position == "nested" else scan
    joined = Join(other, advancing) if position == "last" else Join(advancing, other)
    prog = _program([joined], lambda prog: prog.emit(joined))
    result = shots(prog)
    # The scan loads two points before their pulses at 0 and 28; B occupies
    # [0,56) concurrently. Its origin must not become the scan end, 56.
    assert [waves(s) for s in result] == [[0, 0, 28]] * 6
    assert [s.body_end_cycle - s.start_cycle for s in result] == [56] * 6
    assert [s.end_cycle - s.start_cycle for s in result] == [56] * 6
    prog.compile()
    assert shots(prog) == result


def test_module_join_rejects_unknown_native_pulse_reference_effect_before_compiling():
    class CustomPulse(NativePulse):
        pass

    template = _pulse("custom", 0)

    class CustomOutput(SoftDelay):
        def declare(self, prog):
            prog.declare(template)

        def lower(self, prog, *, timeline, tag):
            prog.append_macro(CustomPulse(ch=0, name=template.pulse_id, t=0))

    joined = Join(CustomOutput("custom_output", 0), _pulse("b", 1))
    with pytest.raises(
        ValueError, match="parallel admission.*unknown reference effect.*CustomPulse"
    ):
        _program([joined], lambda prog: prog.emit(joined))


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


@pytest.mark.parametrize("exchange", [False, True])
def test_module_join_preserves_annotated_native_pulse_effects(exchange):
    template = _pulse("native", 1, 50)
    native = _NativeSequence(
        "native_output",
        lambda prog: prog.append_macro(
            SourcePulse(ch=1, name=template.pulse_id, t=0, source_name="native")
        ),
    )
    choice = Branch("scan", _pulse("a", 0), None, None)
    joined = Join(choice, native) if exchange else Join(native, choice)
    prog = _program([template, joined], lambda prog: prog.emit(joined))
    # Native output stays at the parallel origin. It does not advance a Module
    # timeline; the branch accounts max(28, 0, 0)=28 on every selector point.
    assert [waves(s) for s in shots(prog)] == [[0, 0], [0], [0]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [28] * 6


@pytest.mark.parametrize("annotation", ["LOOP_START", "BRANCH_START", "CUSTOM"])
def test_module_join_rejects_metadata_that_is_not_a_source_annotation(annotation):
    unknown = _NativeSequence(
        "unknown",
        lambda prog: prog.append_macro(
            MetaMacro(type=annotation, name="unknown", info={"native": True})
        ),
    )
    joined = Join(unknown, _pulse("b", 1))
    with pytest.raises(ValueError, match="unknown reference effect of MetaMacro"):
        _program([joined], lambda prog: prog.emit(joined))


@pytest.mark.parametrize("wrapper", ["source", "pulse", "closing"])
def test_module_join_rejects_unknown_subclasses_of_annotation_wrappers(wrapper):
    class CustomSource(MetaMacro):
        pass

    class CustomPulse(SourcePulse):
        pass

    class CustomClosing(MarkedCloseLoop):
        pass

    macros = {
        "source": CustomSource(type="SOURCE_START", name="unknown"),
        "pulse": CustomPulse(ch=0, name="unknown", t=0, source_name="unknown"),
        "closing": CustomClosing(),
    }
    unknown = _NativeSequence(
        "unknown", lambda prog: prog.append_macro(macros[wrapper])
    )
    joined = Join(unknown, _pulse("b", 1))
    with pytest.raises(ValueError, match="unknown reference effect of Custom"):
        _program([joined], lambda prog: prog.emit(joined))


@pytest.mark.parametrize("nested", [False, True])
def test_module_join_counts_whole_advancing_sequences_without_splitting_them(nested):
    choice = Branch("scan", _pulse("a", 0), None, _pulse("long_a", 0, 50))
    advancing = Join(choice) if nested else choice
    joined = Join([advancing, choice], _pulse("b", 1))
    result = shots(_program([joined], lambda prog: prog.emit(joined)))
    # Each branch emission accounts 56; two sequential emissions end at 112.
    assert [waves(s) for s in result] == [[0, 0, 56], [0], [0, 0, 56]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [112] * 6
    assert [s.end_cycle - s.start_cycle for s in result] == [112] * 6


@pytest.mark.parametrize("nested", [False, True])
def test_module_join_rejects_two_advancing_parallel_sequences(nested):
    left = Branch("scan", _pulse("a", 0), None, None)
    right = ScanWith("inner", [7, 9], "value").add_content(_pulse("b", 1))
    joined = Join(Join(left) if nested else left, [right, right])
    with pytest.raises(
        ValueError, match="parallel admission.*multiple reference-advancing branches"
    ):
        _program([joined], lambda prog: prog.emit(joined))


@pytest.mark.parametrize("inside_case", [False, True])
def test_module_join_checks_unknown_effects_even_after_an_advancing_operation(
    inside_case,
):
    unknown = _NativeSequence("unknown", lambda prog: prog.append_macro(Macro()))
    choice = (
        Branch("scan", [_pulse("a", 0), unknown], None, None)
        if inside_case
        else Branch("scan", _pulse("a", 0), None, None)
    )
    joined = Join(choice if inside_case else [choice, unknown], _pulse("b", 1))
    with pytest.raises(
        ValueError, match="parallel admission.*unknown reference effect.*Macro"
    ):
        _program([joined], lambda prog: prog.emit(joined))


@pytest.mark.parametrize("access", ["RR", "RW", "WR", "WW"])
@pytest.mark.parametrize("alias", [False, True])
def test_module_join_exchange_checks_physical_register_dependencies(access, alias):
    def initialize(prog):
        for name in ("shared", "left_sink", "right_sink"):
            prog.add_reg(name)

    def operation(mode, sink, use_alias):
        def lower(prog):
            name = prog.reg_dict["shared"].full_addr() if use_alias else "shared"
            if mode == "R":
                prog.write_reg(sink, name)
            else:
                prog.write_reg(name, 7)

        return lower

    left = _NativeSequence("left", operation(access[0], "left_sink", False))
    right = _NativeSequence("right", operation(access[1], "right_sink", alias))
    choice = Branch("scan", _pulse("a", 0), None, _pulse("long_a", 0, 50))
    joined = Join([left, choice], [right, _pulse("b", 1)])

    def body(prog):
        prog.emit(joined)

    if access == "RR":
        result = shots(_program([joined], body, initialize=initialize))
        assert [waves(s) for s in result] == [[0, 0], [0], [0, 0]] * 2
        assert [s.body_end_cycle - s.start_cycle for s in result] == [56] * 6
    else:
        with pytest.raises(
            ValueError, match="parallel exchange.*register/word conflict.*r0"
        ):
            _program([joined], body, initialize=initialize)


@pytest.mark.parametrize("shared_wave", [False, True])
def test_module_join_exchange_checks_persistent_wave_words(shared_wave):
    a, b = _pulse("a", 0), _pulse("b", 1)

    def patch(template):
        return _NativeSequence(
            "patch",
            lambda prog: prog.append_macro(
                PatchWmemFromRegs(name=template.pulse_id, freq_reg="frequency")
            ),
        )

    choice = Branch("scan", a, None, a)
    joined = Join([patch(a), choice], [patch(a if shared_wave else b), b])
    initialize = _declare_registers("frequency")
    if shared_wave:
        with pytest.raises(
            ValueError, match="parallel exchange.*register/word conflict.*wmem"
        ):
            _program([joined], lambda prog: prog.emit(joined), initialize=initialize)
    else:
        result = shots(
            _program([joined], lambda prog: prog.emit(joined), initialize=initialize)
        )
        assert [waves(s) for s in result] == [[0, 0], [0], [0, 0]] * 2
        assert [s.body_end_cycle - s.start_cycle for s in result] == [28] * 6


@pytest.mark.parametrize("swapped", [False, True])
def test_module_join_unknown_word_effects_only_reject_when_exchange_is_needed(swapped):
    choice = Branch("scan", _pulse("a", 0), None, None)
    # Native trigger has known stable reference semantics. Its output/port
    # effects are not certified by the parallel register/word summary.
    output = _NativeSequence("output", lambda prog: prog.trigger(t=0))
    joined = Join(choice, output) if swapped else Join(output, choice)
    if swapped:
        with pytest.raises(
            ValueError, match="parallel exchange.*unknown register/word effects"
        ):
            _program([joined], lambda prog: prog.emit(joined))
    else:
        assert len(shots(_program([joined], lambda prog: prog.emit(joined)))) == 6


@pytest.mark.parametrize("empty", [False, True])
def test_module_join_cannot_bypass_wait_only_parent_via_internal_forks(empty):
    choice = Branch("scan", _pulse("a", 0), None, _pulse("long_a", 0, 50))
    joined = Join(None if empty else _pulse("b", 1))

    def body(prog):
        prog.fork("default", into="waiting")
        prog.delay_timeline(100 / 430.08, timeline="waiting")
        # A runtime branch still leaves a path-selected, wait-only external end.
        prog.write_reg("duration", 10)
        with prog.branch(by="scan", align_timeline=True) as cases, cases.case(1):
            prog.delay_timeline("duration")
        prog.emit(joined, timeline="waiting")

    with pytest.raises(ValueError, match="timeline can only wait"):
        _program([choice, joined], body, initialize=_declare_registers("duration"))


@pytest.mark.parametrize("bad_case", [0, 1, 2])
def test_module_join_rejects_reverse_generator_fifo_in_every_branch_case(bad_case):
    cases: list[Module | None] = [None, None, None]
    cases[bad_case] = _pulse("a", 0)
    choice = Branch("scan", *cases)
    joined = Join(choice, _pulse("late", 0, pre=100 / 430.08))
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
            "pair", "count" if register else 2, range_hint=(2, 3) if register else None
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


@pytest.mark.parametrize("swapped", [False, True])
def test_module_join_does_not_inspect_unknown_native_scratch_without_exchange(swapped):
    update = _NativeSequence("update", lambda prog: prog.inc_reg("value", 2**24))
    choice = Branch("scan", update, update, update)
    other = _pulse("b", 1)
    joined = Join(choice, other) if swapped else Join(other, choice)
    initialize = _declare_registers("value")
    if swapped:
        with pytest.raises(
            ValueError, match="parallel exchange.*unknown register/word effects"
        ):
            _program([joined], lambda prog: prog.emit(joined), initialize=initialize)
    else:
        result = shots(
            _program([joined], lambda prog: prog.emit(joined), initialize=initialize)
        )
        assert [waves(s) for s in result] == [[0]] * 6
        assert [s.body_end_cycle - s.start_cycle for s in result] == [28] * 6


@pytest.mark.parametrize("alias", ["w0", "w_freq"])
def test_module_join_does_not_exclude_native_wave_scratch_with_external_readers(alias):
    read = _NativeSequence("read", lambda prog: prog.write_reg("value", alias))
    choice = Branch("scan", _pulse("a", 0), None, None)
    joined = Join(choice, [read, _pulse("b", 1)])
    with pytest.raises(
        ValueError, match="parallel exchange.*register/word conflict.*w0"
    ):
        _program(
            [joined],
            lambda prog: prog.emit(joined),
            initialize=_declare_registers("value"),
        )


def test_module_join_rejects_unknown_reference_hidden_by_native_delay_adapter():
    class CustomDelay(NativeDelay):
        pass

    unknown = _NativeSequence(
        "unknown",
        lambda prog: prog.append_macro(
            CustomDelay(t=0, auto=True, gens=True, ros=True)
        ),
    )
    joined = Join(unknown, _pulse("b", 1))
    with pytest.raises(
        ValueError, match="parallel admission.*unknown reference effect.*CustomDelay"
    ):
        _program([joined], lambda prog: prog.emit(joined))


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
    # Readout config0, playback22 and ADC43..215.2 plus post8.6016
    # finish at224, independent of the shorter Branch's28/0/56 advance.
    assert [waves(s) for s in result] == [[0, 22, 0], [0, 22], [0, 22, 0]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [224] * 6
    assert [s.end_cycle - s.start_cycle for s in result] == [224] * 6


@pytest.mark.parametrize("register", ["s2", "s_cfg", "s_ctrl"])
def test_module_join_rejects_unknown_processor_control_register_effects(register):
    control = _NativeSequence("control", lambda prog: prog.write_reg(register, 0))
    joined = Join(control, _pulse("b", 1))
    with pytest.raises(
        ValueError, match="parallel admission.*unknown reference effect"
    ):
        _program([joined], lambda prog: prog.emit(joined))


@pytest.mark.parametrize("register", ["s1", "s_rand", "s11", "s_usr_time"])
def test_module_join_cannot_exchange_volatile_register_reads_as_pure_reads(register):
    read = _NativeSequence("read", lambda prog: prog.write_reg("value", register))
    choice = Branch("scan", _pulse("a", 0), None, None)
    joined = Join(choice, read)
    with pytest.raises(
        ValueError, match="parallel exchange.*unknown register/word effects"
    ):
        _program(
            [joined],
            lambda prog: prog.emit(joined),
            initialize=_declare_registers("value"),
        )


def test_module_join_reference_advancing_child_preserves_parallel_origin():
    choice = Branch("scan", _pulse("a", 0), None, _pulse("long_a", 0, 50))
    other = _pulse("b", 1)
    joined = Join(choice, other)
    result = shots(_program([joined], lambda prog: prog.emit(joined)))
    # A accounts max(28,0,56)=56 without moving reference. B belongs to another
    # parallel branch and still starts at 0. The Join ends at max(56,28)=56.
    assert [waves(s) for s in result] == [[0, 0], [0], [0, 0]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [56] * 6
    assert [s.end_cycle - s.start_cycle for s in result] == [56] * 6
