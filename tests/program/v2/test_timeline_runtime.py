"""Runtime timeline ends observed through linked compiler timing."""

from collections.abc import Callable

import pytest
from qick.asm_v2 import (
    CloseLoop,
    Macro,
    OpenLoop,
    Pulse,
    QickParam,
    QickSweep1D,
    WriteReg,
)
from zcu_tools.program.v2 import ir
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modules import LoadValue, LoadWord
from zcu_tools.program.v2.timeline import TimelineProgramV2

from .timing_support import shots, waves


def program(
    body: Callable[[TimelineProgramV2], None],
    *,
    reps: int = 1,
    points: int = 3,
    initialize: Callable[[TimelineProgramV2], None] | None = None,
):
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("scan", points)
            for ch in range(2):
                self.declare_gen(ch, nqz=1)
                self.add_pulse(
                    ch,
                    "p" + str(ch),
                    style="const",
                    freq=1000,
                    phase=0,
                    gain=0.5,
                    length=25 / 384,
                )  # 25 fabric cycles = 28 tProc cycles.
            self.append_macro(AllocateTime())
            if initialize is not None:
                initialize(self)

        def _body(self, cfg):
            body(self)

    return Program(make_mock_soccfg(2, 1), reps=reps, initial_delay=0, relax_delay=0)


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


def test_crossing_join_uses_three_instruction_flag_selection():
    def body(prog):
        prog.fork("default", into="other")
        prog.delay_timeline(QickSweep1D("scan", 20 / 430.08, 180 / 430.08))
        prog.delay_timeline(100 / 430.08, timeline="other")
        prog.join("other", into="default")

    prog = program(body, points=41)
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [100] * 21 + list(
        range(104, 181, 4)
    )
    selections = [
        (index, inst)
        for index, inst in enumerate(prog.prog_list)
        if inst.get("CMD") == "REG_WR" and inst.get("IF") == "S"
    ]
    assert len(selections) == 1
    index, selection = selections[0]
    assert [inst["CMD"] for inst in prog.prog_list[index - 2 : index + 1]] == [
        "TEST",
        "REG_WR",
        "REG_WR",
    ]
    changed = [dict(inst) for inst in prog.prog_list]
    changed[index] = {**selection, "IF": "NS"}
    inverted = ir.analyze_timing(
        ir.CompiledTimingInput(changed, prog.labels, prog.timing_layout)
    ).shots
    assert [s.body_end_cycle - s.start_cycle for s in inverted] == list(
        range(20, 100, 4)
    ) + [100] * 21


@pytest.mark.parametrize("candidate_count", [2, 3, 4])
def test_k_way_join_has_one_copy_and_two_instructions_per_comparison(candidate_count):
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            for axis in range(candidate_count - 1):
                self.add_loop(f"i{axis}", 2)

        def _body(self, cfg):
            names = [f"side{axis}" for axis in range(candidate_count - 1)]
            self.fork("default", into=names)
            self.delay_timeline(100 / 430.08)
            for axis, name in enumerate(names):
                self.delay_timeline(
                    QickSweep1D(f"i{axis}", 20 / 430.08, 180 / 430.08),
                    timeline=name,
                )
            self.join(*names, into="default")

    prog = Program(make_mock_soccfg(0, 0), initial_delay=0, relax_delay=0)
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [100] + [180] * (
        2 ** (candidate_count - 1) - 1
    )
    writes = [
        index for index, inst in enumerate(prog.prog_list) if inst.get("IF") == "S"
    ]
    assert len(writes) == candidate_count - 1
    selection = prog.prog_list[writes[0] - 2 : writes[-1] + 1]
    assert [inst["CMD"] for inst in selection] == ["TEST", "REG_WR", "REG_WR"] + [
        cmd for _ in range(candidate_count - 2) for cmd in ("TEST", "REG_WR")
    ]


def test_unhinted_author_time_uses_nonnegative_contract_without_runtime_floor():
    def body(prog):
        prog.write_reg_op("duration", "scan", "SL", 4)
        prog.write_reg_op("duration", "duration", "+", 10)
        prog.delay_timeline("duration")

    prog = program(body)
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [10, 26, 42]
    assert not any("^" in str(inst.get("OP", "")) for inst in prog.prog_list)
    assert not any(
        inst.get("CMD") == "REG_WR" and inst.get("IF") == "S" for inst in prog.prog_list
    )


def test_branch_join_extracts_unbounded_author_source_with_nonuniform_coefficients():
    def body(prog):
        prog.write_reg_op("duration", "scan", "SL", 4)
        prog.write_reg_op("duration", "duration", "+", 10)
        with prog.branch(by="scan", align_timeline=True) as branch:
            with branch.case(0):
                prog.delay_timeline("duration")
                prog.delay_timeline(10 / 430.08)
            with branch.case(1):
                prog.delay_timeline("duration")
                prog.delay_timeline(20 / 430.08)
            with branch.case(2):
                prog.delay_timeline("duration")
                prog.delay_timeline("duration")
                prog.delay_timeline(5 / 430.08)

    prog = program(body)
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [30, 57, 89]
    masks = [
        index
        for index, inst in enumerate(prog.prog_list)
        if "^" in str(inst.get("OP", ""))
    ]
    # Unknown signed differences use a mask. The shared author value is added
    # once after selection, not separately rebuilt into both candidates.
    assert masks
    assert "+" in str(prog.prog_list[masks[-1] + 1]["OP"])


def rebased_runtime_end_program():
    loader = LoadValue("c_times", [200, 80, 220], "scan", "c_duration")

    def body(prog):
        prog.emit(loader)
        prog.fork("default", into=["a", "b", "c"])
        prog.delay_timeline(
            QickSweep1D("scan", 20 / 430.08, 180 / 430.08), timeline="a"
        )
        prog.delay_timeline(
            QickSweep1D("scan", 180 / 430.08, 20 / 430.08), timeline="b"
        )
        prog.delay_timeline("c_duration", timeline="c")
        prog.join("a", "b", into="default")
        prog.sync(timeline="default")
        prog.pulse(0, "p0", timeline="c")

    return program(body, initialize=lambda p: p.declare(loader))


def test_sync_rebases_unjoined_runtime_end_after_crossing_join():
    result = shots(rebased_runtime_end_program())
    assert [waves(shot) for shot in result] == [[200], [100], [220]]
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        228,
        128,
        248,
    ]


class AllocateTime(Macro):
    def preprocess(self, prog):
        prog.add_reg("duration")

    def translate(self, prog):
        pass


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


@pytest.mark.parametrize("producer", ["alu", "dmem"])
def test_author_calculated_register_delay_joins_at_actual_runtime_maximum(producer):
    def body(prog):
        if producer == "alu":
            prog.write_reg_op("duration", "scan", "+", 10)
        else:
            prog.read_dmem("duration", 0)
        prog.fork("default", into="other")
        prog.delay_timeline("duration")
        prog.delay_timeline(11 / 430.08, timeline="other")
        prog.join("other", into="default")
        prog.pulse(0, "p0", timeline="default")

    def initialize(prog):
        prog.add_dmem([11])

    result = shots(program(body, reps=2, initialize=initialize))
    expected = [11, 11, 12] if producer == "alu" else [11, 11, 11]
    assert [waves(s) for s in result] == [[t] for t in expected] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [
        t + 28 for t in expected
    ] * 2


@pytest.mark.parametrize("producer", ["alu", "dmem"])
@pytest.mark.parametrize("hint", [None, (6, 6)])
def test_author_register_overwrite_after_constant_accepts_optional_hint(producer, hint):
    def body(prog):
        prog.write_reg("duration", 1)
        if producer == "alu":
            prog.write_reg_op("duration", "duration", "+", 5)
        else:
            prog.read_dmem("duration", 0)
        prog.delay_timeline("duration", range_hint=hint)
        prog.pulse(0, "p0", timeline="default")

    def initialize(prog):
        prog.add_dmem([6])

    result = shots(program(body, initialize=initialize))
    assert [waves(s) for s in result] == [[6]] * 3
    assert [s.body_end_cycle - s.start_cycle for s in result] == [34] * 3


@pytest.mark.parametrize("hint", [None, (10, 12)])
def test_author_register_hint_controls_same_channel_overlap_proof(hint):
    def body(prog):
        prog.write_reg_op("duration", "scan", "+", 10)
        prog.fork("default", into="other")
        prog.pulse(0, "p0", t=0)
        prog.delay_timeline("duration", timeline="other", range_hint=hint)
        prog.pulse(0, "p0", timeline="other")

    if hint is not None:
        with pytest.raises(ValueError, match="channel 0.*overlap"):
            program(body)
    else:
        assert [waves(s) for s in shots(program(body))] == [[0, 10], [0, 11], [0, 12]]


@pytest.mark.parametrize("as_array", [False, True])
@pytest.mark.parametrize("operation", ["delay", "pulse", "config", "trigger"])
def test_runtime_only_tag_time_getter_never_returns_preprocessing_metadata(
    operation, as_array
):
    def initialize(prog):
        prog.add_dmem([40])
        prog.declare_readout(0, length=4 / 307.2)
        prog.add_readoutconfig(0, "ro", freq=100)

    def body(prog):
        prog.read_dmem("duration", 0)
        prog.delay_timeline("duration", tag="runtime" if operation == "delay" else None)
        if operation == "pulse":
            prog.pulse(0, "p0", timeline="default", tag="runtime")
        elif operation == "config":
            prog.send_readoutconfig(0, "ro", timeline="default", tag="runtime")
        elif operation == "trigger":
            prog.trigger([0], timeline="default", tag="runtime")

    prog = program(body, initialize=initialize)
    with pytest.raises(RuntimeError, match="runtime"):
        prog.get_time_param("runtime", "t", as_array=as_array)


def test_known_affine_register_getter_returns_actual_time_without_rescheduling():
    loader = LoadValue("time", [5, 10, 15], "scan", "loaded")

    def body(prog):
        prog.emit(loader)
        prog.delay_timeline("loaded", tag="end")
        prog.pulse(0, "p0", timeline="default", tag="pulse")

    prog = program(body, initialize=lambda p: p.declare(loader))
    before = shots(prog)
    assert prog.get_time_param("end", "t", as_array=True) * 430.08 == pytest.approx(
        [5, 10, 15]
    )
    value = prog.get_time_param("end", "t")
    assert isinstance(value, QickParam)
    assert value.to_array({"scan": 3}) * 430.08 == pytest.approx([5, 10, 15])
    assert shots(prog) == before


@pytest.mark.parametrize("as_array", [False, True])
def test_nonaffine_author_table_time_getter_rejects_array_and_scalar(as_array):
    loader = LoadValue("time", [5, 10, 6], "scan", "loaded")

    def body(prog):
        prog.emit(loader)
        prog.delay_timeline("loaded", tag="end")
        prog.pulse(0, "p0", timeline="default", tag="pulse")

    prog = program(body, initialize=lambda p: p.declare(loader))
    for tag in ("end", "pulse"):
        with pytest.raises(RuntimeError, match="non-affine"):
            prog.get_time_param(tag, "t", as_array=as_array)


@pytest.mark.parametrize("synchronize", [False, True])
def test_unknown_branch_local_adc_tail_requires_coverage_for_constant_shell_wait(
    synchronize,
):
    def initialize(prog):
        prog.add_dmem([40])
        prog.declare_readout(0, length=4 / 307.2)

    def body(prog):
        prog.read_dmem("duration", 0)
        with (
            prog.branch(by="scan", align_timeline=False) as branch,
            branch.case(0) as case_timeline,
        ):
            prog.fork(case_timeline, into="local")
            prog.delay_timeline("duration", timeline="local")
            prog.trigger([0], width=1 / 430.08, timeline="local")
            if synchronize:
                prog.join("local", into=case_timeline)

    if not synchronize:
        with pytest.raises(ValueError, match="ADC.*0.*range_hint.*sync"):
            program(body, initialize=initialize)
    else:
        result = shots(program(body, initialize=initialize))
        assert [s.body_end_cycle - s.start_cycle for s in result] == [46, 0, 0]
        assert [s.end_cycle - s.start_cycle for s in result] == [46, 0, 0]


def test_hinted_branch_local_adc_tail_uses_constant_shell_wait_bound():
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("scan", 1)
            self.add_reg("duration")
            self.add_dmem([40])
            self.declare_readout(0, length=4 / 307.2)

        def _body(self, cfg):
            self.read_dmem("duration", 0)
            with (
                self.branch(by="scan", align_timeline=False) as branch,
                branch.case(0) as case_timeline,
            ):
                self.fork(case_timeline, into="local")
                self.delay_timeline("duration", timeline="local", range_hint=(30, 50))
                self.trigger([0], width=1 / 430.08, timeline="local")

    with pytest.warns(UserWarning, match="WAIT.*duration.*slack=20 cycles"):
        prog = Program(make_mock_soccfg(0, 1), initial_delay=0, relax_delay=20 / 430.08)
    (wait,) = [inst for inst in prog.prog_list if inst["CMD"] == "WAIT"]
    # The hinted latest origin is 50; four ADC samples take 5.6 tProc cycles.
    assert wait["TIME"] == "@56"
    (shot,) = shots(prog)
    assert shot.body_end_cycle - shot.start_cycle == 0


@pytest.mark.parametrize("count", [2, "duration"])
def test_author_register_duration_in_repeat_debits_actual_iteration_advances(count):
    def initialize(prog):
        prog.add_dmem([3])

    def body(prog):
        prog.read_dmem("duration", 0)
        prog.fork("default", into="external")
        prog.delay_timeline(100 / 430.08, timeline="external")
        prog.sync()
        with prog.repeat("r", count):
            prog.delay_timeline("duration")
            prog.pulse(0, "p0", timeline="default")
            prog.sync()
        prog.join("external", into="default")
        prog.pulse(1, "p1", timeline="default")

    result = shots(program(body, initialize=initialize))
    expected = [3, 34, 100] if count == 2 else [3, 34, 65, 100]
    assert [waves(s) for s in result] == [expected] * 3
    assert [s.body_end_cycle - s.start_cycle for s in result] == [128] * 3


@pytest.mark.parametrize("align", [False, True])
def test_branch_register_time_can_be_captured_from_author_entry_state(align):
    def initialize(prog):
        prog.add_dmem([40])

    def body(prog):
        prog.read_dmem("duration", 0)
        with prog.branch(by="scan", align_timeline=align) as branch:
            for index in range(3):
                with branch.case(index):
                    prog.delay_timeline("duration")
                    prog.delay_timeline(index / 430.08)
        prog.pulse(0, "p0", timeline="default")

    expected = [42] * 3 if align else [40, 41, 42]
    assert [waves(s) for s in shots(program(body, initialize=initialize))] == [
        [t] for t in expected
    ]


def test_hinted_register_time_name_does_not_conflict_with_repeat_counter_validation():
    def initialize(prog):
        prog.add_reg("time")

    def body(prog):
        prog.write_reg_op("time", "scan", "+", 10)
        prog.delay_timeline("time", range_hint=(10, 12))

    assert [
        s.body_end_cycle - s.start_cycle
        for s in shots(program(body, initialize=initialize))
    ] == [10, 11, 12]


def test_static_managed_pulse_after_unknown_segment_restores_native_auto_tail():
    def initialize(prog):
        prog.add_dmem([40])

    def body(prog):
        prog.read_dmem("duration", 0)
        prog.fork("default", into="fixed")
        prog.delay_timeline("duration")
        prog.pulse(0, "p0", timeline="default")
        prog.delay_timeline(100 / 430.08, timeline="fixed")
        prog.pulse(0, "p0", timeline="fixed")
        prog.pulse(0, "p0", t="auto")

    assert [waves(s) for s in shots(program(body, initialize=initialize))] == [
        [40, 100, 128]
    ] * 3


def test_register_repeat_debits_external_end_by_actual_count_and_resets():
    def body(prog):
        prog.write_reg_op("duration", "scan", "+", 1)
        prog.fork("default", into="external")
        prog.delay_timeline(100 / 430.08, timeline="external")
        prog.sync()
        with prog.repeat("r", "duration", range_hint=(1, 3)):
            prog.delay_timeline(15 / 430.08)
            prog.sync()
        prog.join("external", into="default")
        prog.pulse(0, "p0", timeline="default")

    result = shots(program(body, reps=2))
    # Repeats advance 15/30/45; remaining 85/70/55 still ends at cycle 100.
    assert [waves(s) for s in result] == [[100]] * 6
    assert [s.body_end_cycle - s.start_cycle for s in result] == [128] * 6


@pytest.mark.parametrize("loader_type", [LoadValue, LoadWord])
def test_known_table_time_producer_places_each_selected_value(loader_type):
    loader = loader_type("time", [10, 20, 30], "scan", "loaded")

    def body(prog):
        prog.emit(loader)
        prog.delay_timeline("loaded")
        prog.pulse(0, "p0", timeline="default")

    result = shots(program(body, initialize=lambda p: p.declare(loader), reps=2))
    assert [waves(s) for s in result] == [[10], [20], [30]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [38, 48, 58] * 2


def test_runtime_join_uses_pointwise_maximum_and_preserves_later_target():
    left = LoadValue("left", [10, 30, 20], "scan", "left_time")
    right = LoadValue("right", [30, 10, 20], "scan", "right_time")

    def initialize(prog):
        prog.declare(left)
        prog.declare(right)

    def body(prog):
        prog.fork("default", into=["left", "right"])
        prog.emit(left)
        prog.emit(right)
        prog.delay_timeline("left_time", timeline="left")
        prog.delay_timeline("right_time", timeline="right")
        prog.delay_timeline(25 / 430.08)
        prog.join("left", "right", into="default")
        prog.pulse(0, "p0", timeline="default")

    result = shots(program(body, initialize=initialize))
    assert [waves(s) for s in result] == [[30], [30], [25]]
    assert [s.body_end_cycle - s.start_cycle for s in result] == [58, 58, 53]


def test_runtime_config_and_trigger_use_register_origin_and_close_adc_window():
    def initialize(prog):
        prog.declare_readout(0, length=4 / 307.2)
        prog.add_readoutconfig(0, "ro", freq=100)

    def body(prog):
        prog.write_reg("duration", 40)
        prog.delay_timeline("duration")
        prog.send_readoutconfig(0, "ro", t=3 / 430.08, timeline="default")
        prog.trigger(ros=[0], t=2 / 430.08, width=3 / 430.08, timeline="default")
        prog.pulse(1, "p1", t=0)

    result = shots(program(body, initialize=initialize))
    assert [waves(s) for s in result] == [[43, 0]] * 3
    assert [
        [event.cycle - s.start_cycle for event in s.events if event.kind == "data"]
        for s in result
    ] == [[45, 48]] * 3
    assert [s.body_end_cycle - s.start_cycle for s in result] == [51] * 3
    assert [s.end_cycle - s.start_cycle for s in result] == [51] * 3


@pytest.mark.parametrize("managed", [False, True])
def test_auto_pulse_uses_actual_runtime_channel_end_not_its_upper_bound(managed):
    loader = LoadValue("time", [5, 10, 15], "scan", "loaded")

    def body(prog):
        prog.fork("default", into="other")
        prog.emit(loader)
        prog.delay_timeline("loaded")
        prog.pulse(0, "p0", timeline="default")
        prog.pulse(0, "p0", t="auto", timeline="other" if managed else None)

    result = shots(program(body, initialize=lambda p: p.declare(loader)))
    assert [waves(s) for s in result] == [[5, 33], [10, 38], [15, 43]]
    assert [s.body_end_cycle - s.start_cycle for s in result] == (
        [61, 66, 71] if managed else [33, 38, 43]
    )
    assert [s.end_cycle - s.start_cycle for s in result] == [61, 66, 71]


def test_static_trigger_promotes_crossing_width_and_adc_window_ends():
    def body(prog):
        prog.trigger(
            ros=[0],
            width=QickSweep1D("scan", 3 / 430.08, 9 / 430.08),
            timeline="default",
        )

    result = shots(
        program(body, initialize=lambda p: p.declare_readout(0, length=4 / 307.2))
    )
    assert [
        [e.cycle - s.start_cycle for e in s.events if e.kind == "data"] for s in result
    ] == [[0, 3], [0, 6], [0, 9]]
    assert [s.body_end_cycle - s.start_cycle for s in result] == [6, 6, 9]
    assert [s.end_cycle - s.start_cycle for s in result] == [6, 6, 9]


def test_native_register_reference_advance_preserves_unconsumed_end():
    def body(prog):
        prog.write_reg("duration", 15)
        prog.delay_timeline(40 / 430.08)
        prog.delay_reg_auto("duration")
        prog.pulse(1, "p1", t=0)
        prog.pulse(0, "p0", timeline="default")

    result = shots(program(body))
    assert [waves(s) for s in result] == [[15, 40]] * 3
    assert [s.body_end_cycle - s.start_cycle for s in result] == [68] * 3


@pytest.mark.parametrize("align_timeline", [True, False])
@pytest.mark.parametrize("inside_repeat", [False, True])
def test_static_branch_joins_case_maximum_and_allows_next_placement(
    align_timeline, inside_repeat
):
    def body(prog):
        prog.fork("default", into="untouched")
        prog.delay_timeline(7 / 430.08, timeline="untouched")

        def selected():
            with prog.branch(by="scan", align_timeline=align_timeline) as branch:
                with branch.case(0):
                    prog.delay_timeline(10 / 430.08)
                with branch.case(2):
                    prog.delay_timeline(30 / 430.08)
            prog.pulse(0, "p0", timeline="default")

        if inside_repeat:
            prog.sync()
            with prog.repeat("twice", 2):
                selected()
                prog.sync()
        else:
            selected()
        prog.pulse(1, "p1", timeline="untouched")

    result = shots(program(body))
    # Cases end at 10/0/30, so default resumes at 30 on every point.
    # No closing reference advance may shift untouched's independent cycle 7.
    starts = [30, 30, 30] if align_timeline else [10, 0, 30]
    expected = (
        [[start, start * 2 + 28, max(7, 2 * (start + 28))] for start in starts]
        if inside_repeat
        else [[start, 7] for start in starts]
    )
    assert [waves(s) for s in result] == expected
    assert [s.body_end_cycle - s.start_cycle for s in result] == [
        max(row[-1] + 28, (start + 28) * (2 if inside_repeat else 1))
        for row, start in zip(expected, starts, strict=True)
    ]


@pytest.mark.parametrize("explicit_join", [False, True])
@pytest.mark.parametrize("align_timeline", [True, False])
def test_branch_local_forks_are_discarded_unless_explicitly_joined(
    explicit_join, align_timeline
):
    def body(prog):
        prog.fork("default", into=["p", "q", "untouched"])
        prog.delay_timeline(9 / 430.08, timeline="untouched")
        with (
            prog.branch(by="scan", align_timeline=align_timeline) as branch,
            branch.case(1) as case,
        ):
            prog.fork(case, into=["p", "q"])
            prog.delay_timeline(20 / 430.08, timeline="p")
            prog.delay_timeline(40 / 430.08, timeline="q")
            prog.fork("p", into="local")
            prog.fork("local", into="leaf")
            prog.delay_timeline(80 / 430.08, timeline="leaf")
            if explicit_join:
                prog.join("leaf", into="q")
                prog.join("local", into="q")
            prog.join("p", "q", into=case)
        prog.pulse(0, "p0", timeline="default")
        prog.pulse(1, "p1", timeline="q")
        prog.sync(timeline="untouched")
        prog.pulse(0, "p0", t=200 / 430.08)

    result = shots(program(body))
    # Local p/q merge into the case; leaf affects it only with an explicit join.
    # Outer q stays at zero and untouched stays at 9.
    selected = [100 if explicit_join else 40, 0, 209]
    expected = (
        [selected] * 3 if align_timeline else [[0, 0, 209], selected, [0, 0, 209]]
    )
    assert [waves(s) for s in result] == expected


@pytest.mark.parametrize("align_timeline", [True, False])
@pytest.mark.parametrize("inside_repeat", [False, True])
def test_runtime_branch_joins_pointwise_without_advancing_reference(
    align_timeline, inside_repeat
):
    loader = LoadValue("time", [10, 20, 30], "scan", "loaded")

    def body(prog):
        def selected():
            with prog.branch(by="scan", align_timeline=align_timeline) as branch:
                with branch.case(0):
                    prog.emit(loader)
                    prog.delay_timeline("loaded")
                with branch.case(1):
                    prog.delay_timeline(25 / 430.08)
            prog.pulse(0, "p0", timeline="default")
            # Unmanaged output must still see the incoming reference.
            prog.append_macro(Pulse(ch=1, name="p1", t=0))

        if inside_repeat:
            prog.sync()
            with prog.repeat("twice", 2):
                selected()
                prog.sync()
        else:
            selected()

    result = shots(program(body, initialize=lambda p: p.declare(loader)))
    starts = [25, 25, 30] if align_timeline else [10, 25, 0]
    expected = (
        [[start, 0, 2 * start + 28, start + 28] for start in starts]
        if inside_repeat
        else [[start, 0] for start in starts]
    )
    assert [waves(s) for s in result] == expected


def test_aligned_branch_reconstructs_case_local_nonaffine_author_table():
    loader = LoadValue("time", [10, 30, 20], "scan", "loaded")

    def body(prog):
        with prog.branch(by="scan", align_timeline=True) as branch:
            with branch.case(0):
                prog.emit(loader)
                prog.delay_timeline("loaded")
            with branch.case(1):
                prog.delay_timeline(25 / 430.08)
        prog.pulse(0, "p0", timeline="default")

    prog = program(body, initialize=lambda p: p.declare(loader))
    assert [waves(s) for s in shots(prog)] == [[25], [30], [25]]
    memory = prog.compile_datamem()
    # Three declared data words plus three IR jump destinations, no time copy.
    assert memory is not None and len(memory) == 6
    assert memory[:3].tolist() == [10, 30, 20]
    assert set(int(word) for word in memory[3:]) <= {
        int(str(address).lstrip("&")) for address in prog.labels.values()
    }


def test_sweep_branch_preserves_path_advance_and_runtime_external_end():
    def body(prog):
        prog.fork("default", into="external")
        prog.delay_timeline(100 / 430.08, timeline="external")
        with prog.branch(by="scan", align_timeline=True) as branch:
            for i in range(3):
                with branch.case(i):
                    prog.delay_timeline((10 + 10 * i) / 430.08)
                    prog.pulse(0, "p0")
        prog.join("external", into="default")
        prog.pulse(1, "p1", timeline="default")

    result = shots(program(body, reps=2))
    # Each selected case closes at 38/48/58; external still ends at 100.
    assert [waves(s) for s in result] == [[10, 100], [20, 100], [30, 100]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [128] * 6


def test_aligned_branch_reconstructs_integer_max_without_timing_tables():
    def body(prog):
        prog.delay_timeline(QickSweep1D("scan", 10 / 430.08, 30 / 430.08))
        prog.delay(20 / 430.08)
        with prog.branch(by="scan", align_timeline=True) as branch:
            with branch.case(0):
                prog.delay_timeline(QickSweep1D("scan", 5 / 430.08, 35 / 430.08))
            with branch.case(1):
                prog.delay_timeline(25 / 430.08)
        prog.pulse(0, "p0", timeline="default")

    prog = program(body)
    assert [waves(s) for s in shots(prog)] == [[45], [45], [65]]
    memory = prog.compile_datamem()
    # Three IR dispatch destinations are allowed; no additional time words.
    assert memory is not None and len(memory) == 3
    assert set(int(word) for word in memory) <= {
        int(str(address).lstrip("&")) for address in prog.labels.values()
    }


def test_branch_fork_names_are_case_local_and_only_local_case_is_closed():
    def body(prog):
        with prog.branch(by="scan", align_timeline=True) as branch:
            for i in range(3):
                with branch.case(i) as case:
                    if i == 1:
                        prog.fork(case, into="local")
                        prog.delay_timeline(20 / 430.08, timeline="local")
                    else:
                        prog.delay_timeline((10 if i == 0 else 30) / 430.08)
        prog.pulse(0, "p0", timeline="default")

    result = shots(program(body))
    # Discard unjoined local end 20; default case ends 10/0/30 give True max=30.
    assert [waves(s) for s in result] == [[30]] * 3
    assert [s.body_end_cycle - s.start_cycle for s in result] == [58] * 3


def test_branch_local_names_expire_even_when_every_case_forks_same_name():
    def body(prog):
        with prog.branch(by="scan", align_timeline=True) as branch:
            for index in range(3):
                with branch.case(index) as case:
                    prog.fork(case, into="local")
        prog.delay_timeline(0, timeline="local")

    with pytest.raises(ValueError, match="local.*unknown timeline"):
        program(body)


@pytest.mark.parametrize("register_count", [False, True])
def test_repeat_branch_debits_actual_paths_and_preserves_long_pulse_tail(
    register_count,
):
    def body(prog):
        prog.write_reg_op("duration", "scan", "+", 1)
        prog.fork("default", into="external")
        prog.pulse(1, "p1", timeline="external")
        prog.sync()
        with prog.repeat(
            "r",
            "duration" if register_count else 2,
            range_hint=(1, 3) if register_count else None,
        ):
            with prog.branch(by="scan", align_timeline=True) as branch:
                for i in range(3):
                    with branch.case(i):
                        prog.delay_timeline((5 + 5 * i) / 430.08)
            prog.sync()

    result = shots(program(body, reps=2))
    # Branch duration is max(5,10,15)=15. Fixed count 2 ends at 30;
    # counts 1/2/3 end at 15/30/45, joined with the external pulse end 28.
    expected = [28, 30, 45] if register_count else [30, 30, 30]
    assert [waves(s) for s in result] == [[0]] * 6
    assert [s.body_end_cycle - s.start_cycle for s in result] == expected * 2
    assert [s.end_cycle - s.start_cycle for s in result] == expected * 2


def test_register_timeline_time_inside_repeat_preserves_legal_pulse_backedge():
    loader = LoadValue("time", [5, 10, 15], "scan", "loaded")

    def body(prog):
        prog.emit(loader)
        prog.sync()
        with prog.repeat("r", 2):
            prog.delay_timeline("loaded")
            prog.pulse(0, "p0", timeline="default")
            prog.sync()

    result = shots(program(body, initialize=lambda p: p.declare(loader)))
    assert [waves(s) for s in result] == [[5, 38], [10, 48], [15, 58]]
    assert [s.body_end_cycle - s.start_cycle for s in result] == [66, 76, 86]
    assert [s.end_cycle - s.start_cycle for s in result] == [66, 76, 86]


def test_repeat_identical_swept_branch_paths_keep_static_pointwise_advance():
    def body(prog):
        prog.fork("default", into="external")
        prog.delay_timeline(100 / 430.08, timeline="external")
        prog.sync()
        with prog.repeat("r", 2):
            with prog.branch(by="scan", align_timeline=True) as branch:
                for index in range(3):
                    with branch.case(index):
                        prog.delay_timeline(
                            QickSweep1D("scan", 10 / 430.08, 20 / 430.08)
                        )
            prog.sync()
        prog.delay_timeline(0, timeline="external", tag="remaining")
        prog.pulse(0, "p0", timeline="external")

    prog = program(body)
    # Every path has the same affine advance10/15/20. Two iterations leave
    # static remainders80/70/60; get_time_param remains available for this end.
    remaining = prog.get_time_param("remaining", "t", as_array=True)
    assert remaining * 430.08 == pytest.approx([80, 70, 60])
    assert [waves(s) for s in shots(prog)] == [[100]] * 3


@pytest.mark.parametrize("overlap", [False, True])
def test_repeat_branch_guard_checks_each_case_and_backedge_without_padding(overlap):
    def body(prog):
        prog.sync()
        with prog.repeat("r", 2):
            with prog.branch(by="scan", align_timeline=True) as branch:
                for i in range(3):
                    with branch.case(i) as case:
                        prog.fork(case, into="raw")
                        prog.pulse(0, "p0")
                        if i == 2:
                            prog.pulse(0, "p0", t=35 / 430.08, timeline="raw")
                            if not overlap:
                                prog.delay_timeline(35 / 430.08)
            prog.sync()

    if overlap:
        with pytest.raises(ValueError, match="overlap"):
            program(body)
    else:
        result = shots(program(body))
        # Safe case 2 ends at 28+35=63, which is every iteration's duration.
        assert [waves(s) for s in result] == [[0, 63], [0, 63], [0, 35, 63, 98]]
        assert [s.body_end_cycle - s.start_cycle for s in result] == [126] * 3
        assert [s.end_cycle - s.start_cycle for s in result] == [126] * 3


def test_long_pulse_crosses_register_repeat_without_conservative_shell_padding():
    def body(prog):
        prog.write_reg_op("duration", "scan", "+", 1)
        prog.fork("default", into="external")
        prog.pulse(1, "p1", timeline="external")
        prog.sync()
        with prog.repeat("r", "duration", range_hint=(1, 3)):
            prog.delay_timeline(5 / 430.08)
            prog.sync()

    result = shots(program(body))
    assert [waves(s) for s in result] == [[0]] * 3
    assert [s.body_end_cycle - s.start_cycle for s in result] == [28] * 3
    assert [s.end_cycle - s.start_cycle for s in result] == [28] * 3


def test_long_pulse_crosses_branch_without_conservative_shell_padding():
    def body(prog):
        prog.fork("default", into="external")
        prog.pulse(1, "p1", timeline="external")
        with prog.branch(by="scan", align_timeline=True) as branch:
            for i in range(3):
                with branch.case(i):
                    prog.delay_timeline((5 + 5 * i) / 430.08)

    result = shots(program(body))
    assert [waves(s) for s in result] == [[0]] * 3
    assert [s.body_end_cycle - s.start_cycle for s in result] == [28] * 3
    assert [s.end_cycle - s.start_cycle for s in result] == [28] * 3


class CustomWrite(WriteReg):
    value = 2**31 - 1

    def translate(self, prog):
        WriteReg(dst=self.dst, src=self.value).translate(prog)


def test_custom_write_subclass_cannot_certify_table_index_bounds():
    loader = LoadValue("time", [10, 20, 30], "duration", "loaded")

    def body(prog):
        prog.write_reg("duration", 1)
        prog.append_macro(CustomWrite(dst="duration", src=1))
        prog.emit(loader)
        prog.delay_timeline("loaded")

    with pytest.raises(
        ValueError, match="duration.*(bounds|unregistered.*instruction)"
    ):
        program(body, initialize=lambda p: p.declare(loader))


class CustomOpenLoop(OpenLoop):
    dst = "scan"

    def translate(self, prog):
        super().translate(prog)
        WriteReg(dst=self.dst, src=2**31 - 1).translate(prog)


class CustomCloseLoop(CloseLoop):
    dst = "scan"

    def translate(self, prog):
        super().translate(prog)
        WriteReg(dst=self.dst, src=2**31 - 1).translate(prog)


@pytest.mark.parametrize("custom_loop", ["open", "close"])
def test_custom_loop_subclass_cannot_bypass_time_write_audit(custom_loop):
    def body(prog):
        if custom_loop == "open":
            prog.append_macro(CustomOpenLoop(name="inner", n=1))
        else:
            prog.open_loop(1, name="inner")
        prog.delay_timeline("scan")
        if custom_loop == "close":
            prog.append_macro(CustomCloseLoop())
        else:
            prog.close_loop()

    with pytest.raises(ValueError, match="scan.*unregistered.*instruction"):
        program(body)


class CustomPreprocessWrite(Macro):
    def preprocess(self, prog):
        WriteReg(dst="scan", src=2**31 - 1).translate(prog)

    def translate(self, prog):
        pass


def test_custom_preprocess_writer_cannot_impersonate_sweep_initialization():
    def body(prog):
        prog.append_macro(CustomPreprocessWrite())
        prog.delay_timeline("scan")

    with pytest.raises(ValueError, match="scan.*unregistered.*instruction"):
        program(body)


@pytest.mark.parametrize("custom_writer", ["write", "open", "close"])
def test_custom_native_subclass_writes_do_not_restrict_non_time_words(custom_writer):
    def body(prog):
        if custom_writer == "open":
            prog.append_macro(CustomOpenLoop(name="inner", n=1, dst="duration"))
        else:
            prog.open_loop(1, name="inner")
        if custom_writer == "write":
            # QICK signed immediate -1 preserves the raw 0xffffffff word.
            prog.append_macro(CustomWrite(dst="duration", src=1, value=-1))
        prog.delay_timeline(10 / 430.08)
        if custom_writer == "close":
            prog.append_macro(CustomCloseLoop(dst="duration"))
        else:
            prog.close_loop()

    assert [s.body_end_cycle - s.start_cycle for s in shots(program(body))] == [10] * 3


def test_custom_native_subclass_repeat_count_keeps_author_range_hint():
    def body(prog):
        prog.append_macro(CustomWrite(dst="duration", src=1, value=3))
        prog.sync()
        with prog.repeat("r", "duration", range_hint=(1, 3)):
            prog.delay_timeline(10 / 430.08)
            prog.sync()
        prog.pulse(0, "p0", timeline="default")

    result = shots(program(body))
    assert [waves(s) for s in result] == [[30]] * 3
    assert [s.body_end_cycle - s.start_cycle for s in result] == [58] * 3


@pytest.mark.parametrize("align_timeline", [True, False])
@pytest.mark.parametrize("overwrite_before_branch", [False, True])
@pytest.mark.parametrize("external_cycles", [100, 2])
def test_branch_uses_captured_runtime_entry_after_register_repeat(
    align_timeline, overwrite_before_branch, external_cycles
):
    def body(prog):
        prog.append_macro(CustomWrite(dst="duration", src=1, value=3))
        prog.fork("default", into="external")
        prog.delay_timeline(external_cycles / 430.08, timeline="external")
        prog.sync()
        with prog.repeat("r", "duration", range_hint=(1, 3)):
            prog.delay_timeline(1 / 430.08)
            prog.sync()
        prog.fork("external", into="working")
        if overwrite_before_branch:
            prog.append_macro(CustomWrite(dst="duration", src=1, value=1))
        with prog.branch(
            by="scan", align_timeline=align_timeline, timeline="working"
        ) as branch:
            with branch.case(0):
                prog.delay_timeline(5 / 430.08)
                prog.append_macro(CustomWrite(dst="duration", src=1, value=1))
            with branch.case(2):
                prog.delay_timeline(15 / 430.08)
        prog.pulse(0, "p0", timeline="working")

    result = shots(program(body))
    # Repeat advances 3. External100 leaves97; external2 floors to0.
    if external_cycles == 100:
        expected = [[115]] * 3 if align_timeline else [[105], [100], [115]]
    else:
        expected = [[18]] * 3 if align_timeline else [[8], [3], [18]]
    assert [waves(s) for s in result] == expected


class OverwriteTime(Macro):
    def translate(self, prog):
        WriteReg(dst="duration", src=1).translate(prog)


@pytest.mark.parametrize("location", ["before", "after", "case", "backedge", "reset"])
def test_unregistered_translate_writer_invalidates_constant_time_proof(location):
    def initialize(prog):
        if location == "reset":
            prog.append_macro(OverwriteTime())

    def body(prog):
        prog.write_reg("duration", 40)
        if location == "before":
            prog.append_macro(OverwriteTime())
        prog.delay_timeline("duration")
        if location == "after":
            prog.append_macro(OverwriteTime())
        elif location == "case":
            with prog.branch(by="scan", align_timeline=True) as branch:
                for i in range(3):
                    with branch.case(i):
                        if i == 1:
                            prog.append_macro(OverwriteTime())
        elif location == "backedge":
            prog.sync()
            with prog.repeat("r", 2):
                prog.append_macro(OverwriteTime())
                prog.sync()

    with pytest.raises(ValueError, match="duration.*unregistered.*instruction"):
        program(body, initialize=initialize)


@pytest.mark.parametrize("value", [-1, 2**31])
def test_register_time_rejects_out_of_range_producer(value):
    def body(prog):
        prog.write_reg("duration", value)
        prog.delay_timeline("duration")

    with pytest.raises(ValueError, match="signed 32-bit"):
        program(body)


def test_wait_only_source_rejects_register_delay_but_derived_target_works():
    def body(prog):
        prog.write_reg("duration", 1)
        prog.fork("default", into="external")
        prog.delay_timeline(100 / 430.08, timeline="external")
        prog.sync()
        with prog.repeat("r", "duration", range_hint=(1, 3)):
            prog.delay_timeline(10 / 430.08)
            prog.sync()
        prog.fork("external", into="derived")
        prog.delay_timeline("duration", timeline="derived")
        prog.delay_timeline("duration", timeline="external")

    with pytest.raises(ValueError, match="external.*wait"):
        program(body)


def test_unknown_register_delay_cannot_be_treated_as_zero():
    with pytest.raises((ValueError, RuntimeError), match="missing"):
        program(lambda p: p.delay_timeline("missing"))


def test_register_source_is_snapshotted_before_subsequent_registered_writes():
    def body(prog):
        prog.write_reg("duration", 10)
        prog.delay_timeline("duration")
        prog.write_reg("duration", 20)
        prog.pulse(0, "p0", timeline="default")

    result = shots(program(body))
    assert [waves(s) for s in result] == [[10]] * 3
    assert [s.body_end_cycle - s.start_cycle for s in result] == [38] * 3


def test_time_producer_must_be_initialized_on_every_branch_path():
    def body(prog):
        with prog.branch(by="scan", align_timeline=True) as branch:
            for i in range(3):
                with branch.case(i):
                    if i != 1:
                        prog.write_reg("duration", 10)
        prog.delay_timeline("duration")

    with pytest.raises(ValueError, match="duration.*uninitialized"):
        program(body)


def test_sweep_time_rejects_overflow_on_native_backedge_before_reset():
    def body(prog):
        prog.delay_timeline(
            QickParam(start=(2**31 - 11) / 430.08, spans={"scan": 10 / 430.08})
        )

    # All consumed endpoints fit, but QICK increments the final sweep value
    # once before the loop test/reset. That intermediate would wrap to INT_MIN.
    with pytest.raises(ValueError, match="signed 32-bit.*(backedge|reset)"):
        program(body)


@pytest.mark.parametrize("pulse_end", [False, True])
def test_static_timeline_end_rejects_signed_cycle_overflow(pulse_end):
    def body(prog):
        prog.delay_timeline((2**31 - 1 if pulse_end else 2**31) / 430.08)
        if pulse_end:
            prog.pulse(0, "p0", timeline="default")

    with pytest.raises(ValueError, match="signed 32-bit"):
        program(body)


@pytest.mark.parametrize("loader_type", [LoadValue, LoadWord])
@pytest.mark.parametrize("index", [0, 2])
def test_time_lookup_accepts_inclusive_index_bounds(loader_type, index):
    loader = loader_type("time", [5, 10, 15], "duration", "loaded")

    def body(prog):
        prog.write_reg("duration", index)
        prog.emit(loader)
        prog.delay_timeline("loaded")
        prog.pulse(0, "p0", timeline="default")

    expected = 5 if index == 0 else 15
    result = shots(program(body, initialize=lambda p: p.declare(loader)))
    assert [waves(s) for s in result] == [[expected]] * 3
    assert [s.end_cycle - s.start_cycle for s in result] == [expected + 28] * 3


@pytest.mark.parametrize("loader_type", [LoadValue, LoadWord])
@pytest.mark.parametrize("hint", [None, (5, 10)])
def test_time_lookup_rejects_sweep_larger_than_table(loader_type, hint):
    loader = loader_type("time", [5, 10], "scan", "loaded")

    def body(prog):
        prog.emit(loader)
        prog.delay_timeline("loaded", range_hint=hint)

    with pytest.raises(ValueError, match=r"loaded.*scan.*allowed.*0.*1.*actual.*0.*2"):
        program(body, initialize=lambda p: p.declare(loader))


@pytest.mark.parametrize("loader_type", [LoadValue, LoadWord])
@pytest.mark.parametrize(
    "where", ["before_lookup", "after_lookup_reset", "backedge", "branch"]
)
def test_time_lookup_index_proof_covers_all_writes_and_lookup_execution(
    loader_type, where
):
    loader = loader_type("time", [5, 10, 15], "duration", "loaded")

    def lookup(prog):
        prog.emit(loader)
        if where in ("after_lookup_reset", "branch"):
            prog.write_reg("duration", 0)
        prog.delay_timeline("loaded")

    def body(prog):
        prog.write_reg("duration", 0)
        if where == "backedge":
            prog.sync()
            with prog.repeat("r", 2):
                lookup(prog)
                prog.write_reg("duration", 3)
                prog.sync()
        elif where == "branch":
            with prog.branch(by="scan", align_timeline=True) as branch:
                for i in range(3):
                    with branch.case(i):
                        prog.write_reg("duration", 3 if i == 2 else 0)
                        lookup(prog)
        else:
            prog.write_reg("duration", 3)
            lookup(prog)

    with pytest.raises(
        ValueError, match=r"loaded.*duration.*allowed.*0.*2.*actual.*0.*3"
    ):
        program(body, initialize=lambda p: p.declare(loader))


def test_runtime_delay_rejects_possible_sum_overflow():
    def body(prog):
        prog.write_reg("duration", 2**31 - 1)
        prog.delay_timeline(1 / 430.08)
        prog.delay_timeline("duration")

    with pytest.raises(ValueError, match="overflow"):
        program(body)


def test_runtime_maximum_accepts_signed_limit_without_doubling_overflow():
    left = LoadValue("left", [0, 2**31 - 1, 17], "scan", "left_time")
    right = LoadValue("right", [2**31 - 1, 0, 19], "scan", "right_time")

    def body(prog):
        prog.fork("default", into="other")
        prog.emit(left)
        prog.delay_timeline("left_time")
        prog.emit(right)
        prog.delay_timeline("right_time", timeline="other")
        prog.join("other", into="default")

    def initialize(prog):
        prog.declare(left)
        prog.declare(right)

    result = shots(program(body, initialize=initialize))
    assert [s.body_end_cycle - s.start_cycle for s in result] == [
        2147483647,
        2147483647,
        19,
    ]


def test_unrelated_translate_writer_does_not_restrict_non_time_registers():
    def body(prog):
        prog.append_macro(OverwriteTime())
        prog.delay_timeline(10 / 430.08)

    assert [s.body_end_cycle - s.start_cycle for s in shots(program(body))] == [10] * 3


def test_runtime_timeline_reuses_dead_cycle_storage_for_sequential_events():
    def body(prog):
        prog.write_reg("duration", 40)
        prog.delay_timeline("duration")
        for _ in range(12):
            prog.pulse(0, "p0", timeline="default")

    result = shots(program(body))
    assert [waves(s) for s in result] == [
        [40, 68, 96, 124, 152, 180, 208, 236, 264, 292, 320, 348]
    ] * 3
    assert [s.body_end_cycle - s.start_cycle for s in result] == [376] * 3


def test_runtime_placement_rejects_same_channel_overlap_from_other_timeline():
    def body(prog):
        prog.write_reg("duration", 40)
        prog.fork("default", into="other")
        prog.delay_timeline("duration")
        prog.pulse(0, "p0", timeline="default")
        prog.delay_timeline("duration", timeline="other")
        prog.pulse(0, "p0", timeline="other")

    with pytest.raises(ValueError, match="overlap"):
        program(body)
