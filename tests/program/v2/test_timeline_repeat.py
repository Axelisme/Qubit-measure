"""Repeat boundaries through native compilation and linked timing analysis."""

from collections.abc import Callable
from contextlib import nullcontext

import pytest
from qick.asm_v2 import Macro, Pulse, QickSweep1D, WriteReg
from zcu_tools.program.v2.macro.wmem import PulseFromRegs
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.timeline import CycleBounds, TimelineProgramV2

from tests.program.v2.timing_support import shots as _shots
from tests.program.v2.timing_support import waves as _waves


def _program(
    body: Callable[[TimelineProgramV2], None],
    initialize: Callable[[TimelineProgramV2], None] | None = None,
    *,
    reps: int = 1,
) -> TimelineProgramV2:
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            for ch in range(2):
                self.declare_gen(ch, nqz=1)
                self.add_pulse(
                    ch,
                    f"p{ch}",
                    style="const",
                    freq=1000,
                    phase=0,
                    gain=0.5,
                    length=25 / 384,
                )  # 25 generator ticks = 28 tProc ticks.
            if initialize is not None:
                initialize(self)

        def _body(self, cfg):
            body(self)

    return Program(make_mock_soccfg(2, 1), reps=reps, initial_delay=0, relax_delay=0)


class _CountRegister(Macro):
    def __init__(self, value: int):
        self.value = value

    def preprocess(self, prog):
        prog.add_reg("count")

    def translate(self, prog):
        WriteReg(dst="count", src=self.value).translate(prog)


def _register_initialize(prog: TimelineProgramV2, value: int = 3):
    prog.append_macro(_CountRegister(value))
    prog.add_pulse(
        0,
        "short",
        style="const",
        freq=1000,
        phase=0,
        gain=0.5,
        length=5 / 384,
    )  # Five generator ticks take 5.6 tProc cycles, ceiled to six.


@pytest.mark.parametrize("tail_sync", [False, True])
@pytest.mark.parametrize("position", ["before_write", "after_zero_path", "branch_path"])
def test_repeat_table_definition_must_dominate_its_time_consumer(position, tail_sync):
    from zcu_tools.program.v2.modules import LoadValue

    loader = LoadValue("time", [5, 10], "r", "loaded")

    def initialize(prog):
        prog.add_reg("count")
        prog.add_loop("scan", 2)
        prog.declare(loader)

    def body(prog):
        prog.write_reg("count", 0)
        with prog.repeat("r", "count", range_hint=(0, 2)):
            if position == "before_write":
                prog.timeline_inc("loaded")
            if position == "branch_path":
                with prog.branch(by="scan", align_timeline=False) as branch:
                    with branch.case(0):
                        prog.emit(loader)
                prog.timeline_inc("loaded")
            else:
                prog.emit(loader)
            if tail_sync:
                prog.delay_by_timeline()
        if position == "after_zero_path":
            prog.timeline_inc("loaded")

    with pytest.raises(ValueError, match="loaded.*unknown or uninitialized"):
        _program(body, initialize)


def test_repeat_table_index_proof_does_not_cover_the_final_counter_value():
    from zcu_tools.program.v2.modules import LoadValue

    loader = LoadValue("time", [5, 10], "r", "loaded")

    def body(prog):
        with prog.repeat("r", "count", range_hint=(2, 2)):
            prog.emit(loader)
            prog.timeline_inc("loaded")
            prog.delay_by_timeline()
        # The closing shell increments r to 2. Its body index proof cannot
        # justify another lookup into a two-element table after the loop.
        prog.emit(loader)
        prog.timeline_inc("loaded")

    def initialize(prog):
        _register_initialize(prog, value=2)
        prog.declare(loader)

    with pytest.raises(ValueError, match="r.*(uninitialized|index dependency)"):
        _program(body, initialize)


@pytest.mark.parametrize("invalid", ["index_range", "counter_writer"])
def test_repeat_table_index_keeps_range_and_writer_proof(invalid):
    from zcu_tools.program.v2.modules import LoadValue

    loader = LoadValue("time", [5, 10], "r", "loaded")

    class CounterWrite(Macro):
        def translate(self, prog):
            WriteReg(dst="r", src=0).translate(prog)

    def body(prog):
        with prog.repeat(
            "r", "count", range_hint=(0, 3 if invalid == "index_range" else 2)
        ):
            prog.emit(loader)
            prog.timeline_inc("loaded")
            if invalid == "counter_writer":
                prog.append_macro(CounterWrite())
            prog.delay_by_timeline()

    def initialize(prog):
        _register_initialize(prog, value=2)
        prog.declare(loader)

    error = (
        "loaded.*r.*allowed.*0.*1.*actual.*0.*2"
        if invalid == "index_range"
        else "r.*unregistered.*write at instruction"
    )
    with pytest.raises(ValueError, match=error):
        _program(body, initialize)


def test_register_repeat_uses_output_pointer_and_real_scheduling_window():
    def body(prog):
        with prog.repeat("r", "count", range_hint=(1, 10)):
            prog.pulse(0, "p0")
            prog.timeline_inc(12 / 430.08)
        prog.pulse(1, "p1", timeline="default")

    prog = _program(body, _register_initialize, reps=2)
    assert [_waves(s) for s in _shots(prog)] == [[0, 40, 80, 120]] * 2
    increments = [
        i
        for i in prog.prog_list
        if i.get("DST") == "s14" and i.get("OP") == "s14 + #40"
    ]
    assert 1 <= len(increments) < 32
    for index, inst in enumerate(prog.prog_list):
        if inst in increments:
            assert prog.prog_list[index - 1]["CMD"] == "WPORT_WR"
    assert sum(i["CMD"] == "WPORT_WR" for i in prog.prog_list) < 33


def test_repeat_fixed_point_matches_explicit_native_reference_loop():
    def body(prog):
        with prog.repeat("r", 2):
            prog.pulse(0, "p0")
            prog.timeline_inc(7 / 430.08)
            prog.delay_by_timeline()
        prog.pulse(1, "p1", timeline="default")

    def native_body(prog):
        prog.open_loop(2, name="r")
        prog.pulse(0, "p0", timeline="default")
        prog.timeline_inc(7 / 430.08)
        prog.delay_by_timeline()
        prog.close_loop()
        prog.pulse(1, "p1", timeline="default")

    actual = _program(body)
    expected = _program(native_body)
    (shot,) = _shots(actual)
    assert _waves(shot) == [0, 35, 70]
    assert (shot.body_end_cycle, shot.end_cycle) == (98, 98)
    # Only the native reps/r counters need data registers, not an end carrier.
    assert len(actual.reg_dict) == len(expected.reg_dict) == 2
    assert [
        (inst["C_OP"], inst.get("LIT"))
        for inst in actual.prog_list
        if inst["CMD"] == "TIME"
    ] == [
        (inst["C_OP"], inst.get("LIT"))
        for inst in expected.prog_list
        if inst["CMD"] == "TIME"
    ]
    assert any(
        inst["CMD"] == "TIME" and inst["C_OP"] == "inc_ref" and inst.get("LIT") == "#35"
        for inst in actual.prog_list
    )


@pytest.mark.parametrize("sync_in_middle", [False, True])
def test_repeat_nonfixed_tail_preserves_explicit_reference_position(sync_in_middle):
    def body(prog):
        with prog.repeat("r", 2):
            prog.pulse(0, "p0")
            if sync_in_middle:
                prog.delay_by_timeline()
            prog.timeline_inc(7 / 430.08)
        # A native output exposes reference, not the accumulated timeline end.
        prog.pulse(1, "p1")

    prog = _program(body)
    (shot,) = _shots(prog)
    assert _waves(shot) == [0, 35, 63 if sync_in_middle else 0]
    assert (shot.body_end_cycle, shot.end_cycle) == (
        (70, 91) if sync_in_middle else (70, 70)
    )
    assert len(prog.reg_dict) > 2


@pytest.mark.parametrize("shape", ["nested-scope", "other-timeline"])
def test_repeat_nested_or_other_timeline_sync_keeps_variable_tail(shape):
    def body(prog):
        with prog.repeat("r", 2):
            prog.fork("default", into="local")
            prog.pulse(0, "p0")
            if shape == "nested-scope":
                with prog.timeline("default"):
                    prog.delay_by_timeline()
            else:
                prog.join("default", into="local")
                prog.delay_by_timeline(timeline="local")
        prog.pulse(1, "p1")

    prog = _program(body)
    (shot,) = _shots(prog)
    assert _waves(shot) == [0, 28, 56]
    assert (shot.body_end_cycle, shot.end_cycle) == (56, 84)
    assert len(prog.reg_dict) > 2


def test_repeat_nonzero_entry_warns_without_inserting_reference_advance():
    def body(prog):
        prog.timeline_inc(7 / 430.08)
        with prog.repeat("r", 2):
            # Direct native macro exposes the first entry reference.
            prog.append_macro(Pulse(ch=1, name="p1", t=0))
            prog.pulse(0, "p0")
            prog.delay_by_timeline()

    with pytest.warns(UserWarning, match=r"delay_by_timeline\(\).*before.*loop"):
        prog = _program(body)
    (shot,) = _shots(prog)
    assert _waves(shot) == [0, 7, 35, 35]
    assert (shot.body_end_cycle, shot.end_cycle) == (63, 63)
    assert len(prog.reg_dict) > 2


@pytest.mark.parametrize("actual", [0, 3])
@pytest.mark.parametrize("hint", [None, (0, 3)])
def test_variable_reference_repeat_resets_count_and_external_debit_each_shot(
    actual, hint
):
    def initialize(prog):
        prog.add_reg("count")

    def body(prog):
        prog.write_reg("count", actual)
        prog.fork("default", into="external")
        prog.timeline_inc(140 / 430.08, timeline="external")
        with prog.repeat("r", "count", range_hint=hint):
            prog.pulse(0, "p0")
            prog.delay_by_timeline()
            prog.timeline_inc(7 / 430.08)
        prog.join("external", into="default")
        prog.pulse(1, "p1", timeline="default")

    result = _shots(_program(body, initialize, reps=2))
    expected = [0, 35, 70, 140] if actual else [140]
    assert [_waves(shot) for shot in result] == [expected] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [168] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [168] * 2


@pytest.mark.parametrize("operation", ["delay_by_timeline", "delay", "delay_auto"])
def test_repeat_allows_explicit_reference_owner_advances_outside_case(operation):
    def body(prog):
        with prog.repeat("r", 2):
            prog.pulse(0, "p0")
            if operation == "delay_by_timeline":
                prog.delay_by_timeline()
            elif operation == "delay":
                prog.delay(28 / 430.08)
            else:
                prog.delay_auto()
        prog.pulse(1, "p1")

    (shot,) = _shots(_program(body))
    assert _waves(shot) == [0, 28, 56]
    assert (shot.body_end_cycle, shot.end_cycle) == (56, 84)


@pytest.mark.parametrize("operation", ["delay_by_timeline", "delay", "delay_auto"])
@pytest.mark.parametrize("nested", [False, True])
def test_case_rejects_reference_owner_advances_through_repeat(operation, nested):
    def initialize(prog):
        prog.add_loop("scan", 2)

    def body(prog):
        with (
            prog.branch(by="scan", align_timeline=False) as cases,
            cases.case(0),
            prog.repeat("r", 2) if nested else nullcontext(),
        ):
            prog.timeline_inc(28 / 430.08)
            if operation == "delay":
                prog.delay(28 / 430.08)
            else:
                getattr(prog, operation)()

    with pytest.raises(ValueError, match="cannot advance reference"):
        _program(body, initialize)


def test_variable_repeat_shadows_default_without_advancing_reference():
    def body(prog):
        with prog.repeat("r", 2):
            prog.pulse(0, "p0")
        # Native, unscoped output still uses the original reference.
        prog.pulse(1, "p1")

    (shot,) = _shots(_program(body))
    assert _waves(shot) == [0, 28, 0]
    assert (shot.body_end_cycle, shot.end_cycle) == (56, 56)


@pytest.mark.parametrize(
    "align,count,hint,outer",
    [
        (False, 2, None, 2),
        (True, 2, None, 2),
        (False, "count", None, 1),
        (True, "count", None, 1),
        (False, "count", (0, 3), 1),
        (True, "count", (0, 3), 1),
        (False, "count", None, 2),
        (False, "count", (0, 3), 2),
    ],
)
def test_variable_repeat_nested_in_case_preserves_path_times_and_shot_reset(
    align, count, hint, outer
):
    def initialize(prog):
        _register_initialize(prog)
        prog.add_loop("scan", 2)

    def body(prog):
        prog.timeline_inc(14 / 430.08)
        with prog.branch(by="scan", align_timeline=align) as cases:
            with (
                cases.case(0),
                prog.repeat("outer", outer) if outer > 1 else nullcontext(),
                prog.repeat("inner", count, range_hint=hint),
            ):
                prog.pulse(0, "p0")
            with cases.case(1):
                prog.timeline_inc(28 / 430.08)
        prog.pulse(1, "p1", timeline="default")

    shots = _shots(_program(body, initialize, reps=2))
    n = 2 if count == 2 else 3
    duration = 28 * outer * n
    case0 = [14 + 28 * i for i in range(outer * n)] + [14 + duration]
    case1 = [14 + (duration if align else 28)]
    assert [_waves(s) for s in shots] == [case0, case1] * 2
    assert [s.end_cycle - s.start_cycle for s in shots] == [
        42 + duration,
        42 + (duration if align else 28),
    ] * 2


@pytest.mark.parametrize("overlap", ["parallel", "backedge", "nested"])
def test_variable_repeat_proves_same_channel_and_backedge_overlap(overlap):
    def body(prog):
        with (
            prog.repeat("outer", 2),
            prog.repeat("inner", 2) if overlap == "nested" else nullcontext(),
        ):
            prog.fork("default", into="local")
            prog.pulse(0, "p0")
            if overlap == "parallel":
                prog.pulse(0, "p0", timeline="local")
            else:
                # Raw native timing can leave a tail beyond the managed end.
                prog.append_macro(Pulse(ch=0, name="p0", t=28 / 430.08))

    with pytest.raises(ValueError, match="channel 0.*overlap"):
        _program(body)


def test_aligned_case_reconstructs_variable_repeat_with_pulse_and_delay():
    def initialize(prog):
        prog.add_loop("scan", 2)

    def body(prog):
        with prog.branch(by="scan", align_timeline=True) as cases:
            with cases.case(0), prog.repeat("r", 2):
                prog.pulse(0, "p0")
                prog.timeline_inc(5 / 430.08)
            with cases.case(1):
                prog.timeline_inc(10 / 430.08)
        prog.pulse(1, "p1", timeline="default")

    shots = _shots(_program(body, initialize))
    assert [_waves(s) for s in shots] == [[0, 33, 66], [66]]
    assert [s.end_cycle - s.start_cycle for s in shots] == [94, 94]


@pytest.mark.parametrize("align", [False, True])
def test_branch_inside_variable_repeat_retains_iteration_origin(align):
    def initialize(prog):
        prog.add_loop("scan", 2)

    def body(prog):
        with (
            prog.repeat("r", 2),
            prog.branch(by="scan", align_timeline=align) as cases,
        ):
            for index in range(2):
                with cases.case(index):
                    prog.pulse(0, "p0")
                    prog.timeline_inc((5 + 5 * index) / 430.08)
        prog.pulse(1, "p1", timeline="default")

    shots = _shots(_program(body, initialize))
    assert [_waves(s) for s in shots] == (
        [[0, 38, 76]] * 2 if align else [[0, 33, 66], [0, 38, 76]]
    )


@pytest.mark.parametrize("safe", [False, True])
def test_outer_reference_repeat_protects_inner_variable_repeat_backedge(safe):
    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("outer", 2):
            with prog.repeat("inner", 2):
                prog.pulse(0, "p0")
            if not safe:
                prog.append_macro(Pulse(ch=0, name="p0", t=112 / 430.08))
            prog.delay_by_timeline()

    if not safe:
        with pytest.raises(ValueError, match="channel 0.*overlap"):
            _program(body)
    else:
        (shot,) = _shots(_program(body))
        assert _waves(shot) == [0, 28, 56, 84]
        assert shot.end_cycle == 112


@pytest.mark.parametrize("sync_at_end", [False, True])
def test_repeat_preserves_swept_parent_and_cross_loop_long_pulse(sync_at_end):
    def initialize(prog):
        prog.add_loop("scan", 3)
        prog.add_pulse(
            1, "flux", style="const", freq=1000, phase=0, gain=0.5, length=125 / 384
        )

    def body(prog):
        prog.fork("default", into=["drive", "external"])
        prog.pulse(1, "flux", timeline="external")
        prog.timeline_inc(
            QickSweep1D("scan", 20 / 430.08, 40 / 430.08), timeline="drive"
        )
        with prog.timeline("drive"):
            if sync_at_end:
                prog.delay_by_timeline()
            with prog.repeat("r", 2):
                prog.pulse(0, "p0", timeline="drive")
                prog.timeline_inc(7 / 430.08)
                if sync_at_end:
                    prog.delay_by_timeline()
            prog.pulse(0, "p0", timeline="drive")
        prog.join("external", into="default")
        prog.pulse(1, "p1", timeline="default")

    shots = _shots(_program(body, initialize, reps=2))
    assert [_waves(s) for s in shots] == [
        [0, t, t + 35, t + 70, 140] for t in [20, 30, 40]
    ] * 2
    assert [s.end_cycle - s.start_cycle for s in shots] == [168] * 6


def test_variable_repeat_selected_cross_grid_lengths_and_runtime_getters():
    from zcu_tools.program.v2.modules import PulseCfg, TableLengthPulse
    from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

    drive = TableLengthPulse(
        "selected",
        PulseCfg(
            ch=0, nqz=1, freq=1000, gain=0.5, waveform=ConstWaveformCfg(length=7 / 384)
        ),
        lengths=[7 / 384, 11 / 384],
        idx_reg="scan",
    )

    def initialize(prog):
        prog.add_loop("scan", 2)
        prog.declare(drive)

    def body(prog):
        with prog.repeat("r", 2):
            prog.emit(drive, tag="inside")
        prog.timeline_inc(0, tag="after")
        prog.pulse(1, "p1", timeline="default")

    prog = _program(body, initialize)
    assert [_waves(s) for s in _shots(prog)] == [[0, 8, 16], [0, 13, 26]]
    with pytest.raises(RuntimeError, match="runtime-only"):
        prog.get_time_param("inside", "t", as_array=True)
    assert prog.get_time_param("after", "t", as_array=True) * 430.08 == pytest.approx(
        [16, 26]
    )


@pytest.mark.parametrize("parent", ["default", "drive"])
def test_repeat_synchronizes_parent_before_first_iteration_and_closes_each_iteration(
    parent,
):
    def body(prog):
        if parent != "default":
            prog.fork("default", into=parent)
        prog.pulse(0, "p0", timeline=parent)
        prog.delay_by_timeline(timeline=parent)
        with prog.repeat("repeat", 2, timeline=parent):
            prog.pulse(0, "p0", timeline=parent)
            prog.timeline_inc(7 / 430.08, timeline=parent)
            prog.delay_by_timeline(timeline=parent)
        prog.pulse(0, "p0", timeline=parent)

    (shot,) = _shots(_program(body))
    assert _waves(shot) == [0, 28, 63, 98]
    assert (shot.body_end_cycle, shot.end_cycle) == (126, 126)


@pytest.mark.parametrize("advance", [0, 20])
def test_fixed_repeat_rebases_long_external_pulse_and_reuses_parent_and_local_names(
    advance,
):
    def initialize(prog):
        prog.add_loop("scan", 2)
        prog.add_pulse(
            1,
            "flux",
            style="const",
            freq=1000,
            phase=0,
            gain=0.5,
            length=125 / 384,
        )  # 125 generator ticks = 140 tProc ticks.

    def body(prog):
        prog.fork("default", into="flux")
        prog.pulse(1, "flux", timeline="flux")
        prog.timeline_inc(advance / 430.08)
        prog.delay_by_timeline()
        with prog.repeat("r", 2):
            prog.fork("default", into=["a", "b"])
            prog.pulse(0, "p0", timeline="a")
            prog.timeline_inc(7 / 430.08, timeline="a")
            prog.join("a", "b", into="default")
            prog.join("a", "b", into="default")
            prog.delay_by_timeline()
        prog.pulse(1, "p1", timeline="flux")
        prog.fork("default", into="a")
        prog.pulse(0, "p0", timeline="a")

    prog = _program(body, initialize, reps=2)
    shots = _shots(prog)
    assert [_waves(s) for s in shots] == [
        [0, advance, advance + 35, 140, advance + 70]
    ] * 4
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [168] * 4
    assert [s.end_cycle - s.start_cycle for s in shots] == [168] * 4
    prog.compile()
    assert _shots(prog) == shots


@pytest.mark.parametrize("inner_parent", ["default", "local"])
def test_nested_repeat_syncs_chosen_parent_and_expires_inner_locals(inner_parent):
    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("outer", 2):
            prog.fork("default", into="local")
            prog.timeline_inc(10 / 430.08, timeline=inner_parent)
            prog.delay_by_timeline(timeline=inner_parent)
            with prog.repeat("inner", 2, timeline=inner_parent):
                prog.fork(inner_parent, into="inner_local")
                prog.pulse(0, "p0", timeline="inner_local")
                prog.join("inner_local", into=inner_parent)
                prog.delay_by_timeline(timeline=inner_parent)
            prog.fork(inner_parent, into="inner_local")
            prog.pulse(0, "p0", timeline=inner_parent)
            prog.join("local", "inner_local", into="default")
            prog.delay_by_timeline()

    (shot,) = _shots(_program(body))
    assert _waves(shot) == [10, 38, 66, 104, 132, 160]
    assert (shot.body_end_cycle, shot.end_cycle) == (188, 188)


@pytest.mark.parametrize(
    "operation", ["pulse", "trigger", "delay", "delay_by_timeline", "join", "fork"]
)
def test_repeat_rejects_using_nonparent_external_timeline(operation):
    def body(prog):
        prog.fork("default", into="external")
        prog.delay_by_timeline()
        with prog.repeat("r", 2):
            if operation == "pulse":
                prog.pulse(0, "p0", timeline="external")
            elif operation == "trigger":
                prog.trigger(timeline="external")
            elif operation == "delay":
                prog.timeline_inc(0, timeline="external")
            elif operation == "delay_by_timeline":
                prog.delay_by_timeline(timeline="external")
            elif operation == "join":
                prog.join("external", into="default")
            else:
                prog.fork("external", into="local")
            prog.delay_by_timeline()

    with pytest.raises(ValueError, match="external.*timeline"):
        _program(body)


def test_repeat_local_timeline_is_undeclared_after_exit():
    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("r", 2):
            prog.fork("default", into="local")
            prog.pulse(0, "p0", timeline="local")
            prog.join("local", into="default")
            prog.delay_by_timeline()
        prog.delay_by_timeline(timeline="local")

    with pytest.raises(ValueError, match="local.*fork"):
        _program(body)


def test_fixed_repeat_swept_duration_debits_outer_end_at_each_scan_point():
    def initialize(prog):
        prog.add_loop("scan", 3)

    def body(prog):
        prog.fork("default", into="external")
        prog.timeline_inc(100 / 430.08, timeline="external")
        prog.delay_by_timeline()
        with prog.repeat("r", 2):
            prog.timeline_inc(QickSweep1D("scan", 10 / 430.08, 20 / 430.08))
            prog.pulse(0, "p0", timeline="default")
            prog.delay_by_timeline()
        prog.pulse(1, "p1", timeline="external")

    shots = _shots(_program(body, initialize, reps=2))
    assert [_waves(s) for s in shots] == [
        [10, 48, 100],
        [15, 58, 100],
        [20, 68, 100],
    ] * 2
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [128] * 6
    assert [s.end_cycle - s.start_cycle for s in shots] == [128] * 6


def test_fixed_repeat_preserves_crossing_external_floor():
    def initialize(prog):
        prog.add_loop("scan", 3)

    def body(prog):
        prog.fork("default", into="external")
        prog.timeline_inc(30 / 430.08, timeline="external")
        prog.delay_by_timeline()
        with prog.repeat("r", 2):
            prog.timeline_inc(QickSweep1D("scan", 10 / 430.08, 20 / 430.08))
            prog.delay_by_timeline()

    result = _shots(_program(body, initialize))
    assert [s.body_end_cycle - s.start_cycle for s in result] == [30, 30, 40]


def test_zero_repeat_syncs_parent_but_does_not_execute_or_declare_body():
    def body(prog):
        prog.pulse(0, "p0", timeline="default")
        prog.delay_by_timeline()
        with prog.repeat("zero", 0):
            prog.fork("default", into="local")
            prog.pulse(1, "p1", timeline="local")
            prog.join("local", into="default")
            prog.delay_by_timeline()
        prog.fork("default", into="local")
        prog.pulse(0, "p0", timeline="local")

    (shot,) = _shots(_program(body))
    assert _waves(shot) == [0, 28]
    assert (shot.body_end_cycle, shot.end_cycle) == (56, 56)


@pytest.mark.parametrize("count", [-1, True, False])
def test_repeat_rejects_invalid_integer_count(count):
    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("r", count):
            pass
            prog.delay_by_timeline()

    with pytest.raises(ValueError, match="nonnegative integer"):
        _program(body)


@pytest.mark.parametrize(
    "count,hint",
    [
        (2, (1, 3)),
        ("n", (-1, 3)),
        ("n", (3, 2)),
        ("n", (True, 3)),
    ],
)
def test_repeat_rejects_invalid_range_hint(count, hint):
    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("r", count, range_hint=hint):
            pass
            prog.delay_by_timeline()

    with pytest.raises(ValueError, match="range_hint"):
        _program(body)


def test_repeat_register_count_cannot_alias_its_counter():
    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("same", "same", range_hint=(0, 3)):
            pass
            prog.delay_by_timeline()

    with pytest.raises(ValueError, match="differ.*counter"):
        _program(body)


def test_repeat_preserves_trigger_cycles_and_adc_window_closure():
    def initialize(prog):
        prog.declare_readout(0, length=25 / 307.2)

    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("r", 2):
            prog.trigger([0], t=5 / 430.08, width=10 / 430.08, timeline="default")
            prog.pulse(0, "p0", timeline="default")
            prog.delay_by_timeline()

    (shot,) = _shots(_program(body, initialize))
    # 25 ADC ticks = 35 tProc ticks, so trigger end=40 and pulse end=68.
    assert [(e.cycle, e.value) for e in shot.events if e.kind == "data"] == [
        (5, 1),
        (15, 0),
        (73, 1),
        (83, 0),
    ]
    assert _waves(shot) == [40, 108]
    assert (shot.body_end_cycle, shot.end_cycle) == (136, 136)


@pytest.mark.parametrize("inner_parent", ["default", "local"])
def test_nested_repeat_cannot_use_outer_nonparent_timeline(inner_parent):
    other = "local" if inner_parent == "default" else "default"

    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("outer", 2):
            prog.fork("default", into="local")
            prog.delay_by_timeline(timeline=inner_parent)
            with prog.repeat("inner", 2, timeline=inner_parent):
                prog.fork(other, into="forbidden")
                prog.join("forbidden", into=inner_parent)
                prog.delay_by_timeline(timeline=inner_parent)
            prog.join("local", into="default")
            prog.delay_by_timeline()

    with pytest.raises(ValueError, match=f"{other}.*timeline"):
        _program(body)


@pytest.mark.parametrize("actual", [0, 3])
def test_unhinted_register_repeat_preserves_actual_external_end(actual):
    def initialize(prog):
        _register_initialize(prog, actual)

    def body(prog):
        prog.fork("default", into="external")
        prog.timeline_inc(100 / 430.08, timeline="external")
        prog.delay_by_timeline()
        with prog.repeat("r", "count"):
            prog.pulse(0, "short", timeline="default")
            prog.timeline_inc(9 / 430.08)
            prog.delay_by_timeline()
        prog.join("external", into="default")
        prog.pulse(1, "p1", timeline="default")

    result = _shots(_program(body, initialize, reps=2))
    expected = [100] if actual == 0 else [0, 15, 30, 100]
    assert [_waves(s) for s in result] == [expected] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [128] * 2


@pytest.mark.parametrize("consumer", ["body", "join", "fork", "delay_by_timeline"])
def test_register_repeat_wait_consumers_use_actual_remaining_end(consumer):
    def body(prog):
        prog.fork("default", into="external")
        prog.timeline_inc(100 / 430.08, timeline="external")
        prog.delay_by_timeline()
        with prog.repeat("r", "count", range_hint=(2, 4)):
            prog.pulse(0, "short", timeline="default")
            prog.timeline_inc(9 / 430.08)
            prog.delay_by_timeline()
        if consumer == "join":
            prog.join("external", into="default")
            prog.pulse(1, "p1", timeline="default")
        elif consumer == "fork":
            prog.fork("external", into="derived")
            prog.pulse(1, "p1", timeline="derived")
        elif consumer == "delay_by_timeline":
            prog.delay_by_timeline(timeline="external")
            prog.pulse(1, "p1", timeline="external")

    shots = _shots(_program(body, _register_initialize, reps=2))
    # Actual reference is 3*15=45; the runtime remainder is 55.
    expected = [0, 15, 30] if consumer == "body" else [0, 15, 30, 100]
    end = 100 if consumer == "body" else 128
    assert [_waves(s) for s in shots] == [expected] * 2
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [end] * 2
    assert [s.end_cycle - s.start_cycle for s in shots] == [end] * 2


@pytest.mark.parametrize("operation", ["pulse", "trigger", "emit", "delay"])
@pytest.mark.parametrize("derive", ["none", "join", "fork"])
def test_register_repeat_wait_source_stays_restricted_after_deriving_schedule_point(
    operation, derive
):
    from zcu_tools.program.v2.modules.delay import SoftDelay

    delay = SoftDelay("delay", 0.01)

    def initialize(prog):
        _register_initialize(prog)
        prog.declare(delay)

    def body(prog):
        prog.fork("default", into="external")
        prog.timeline_inc(100 / 430.08, timeline="external")
        prog.delay_by_timeline()
        with prog.repeat("r", "count", range_hint=(2, 4)):
            prog.timeline_inc(15 / 430.08)
            prog.delay_by_timeline()
        if derive == "join":
            prog.join("external", into="default")
        elif derive == "fork":
            prog.fork("external", into="derived")
        if operation == "pulse":
            prog.pulse(1, "p1", timeline="external")
        elif operation == "trigger":
            prog.trigger(timeline="external")
        elif operation == "emit":
            prog.emit(delay, timeline="external")
        else:
            prog.timeline_inc(0, timeline="external")

    with pytest.raises(ValueError, match="external.*wait"):
        _program(body, initialize)


def _readout_register_initialize(prog: TimelineProgramV2):
    _register_initialize(prog)
    prog.declare_readout(0, length=0.4)
    prog.add_readoutconfig(0, "ro", freq=100)


def _repeat_with_external_wait(prog: TimelineProgramV2):
    prog.fork("default", into="external")
    prog.timeline_inc(100 / 430.08, timeline="external")
    prog.delay_by_timeline()
    with prog.repeat("r", "count", range_hint=(2, 4)):
        prog.timeline_inc(15 / 430.08)
        prog.delay_by_timeline()


def _send_config(
    prog: TimelineProgramV2, runtime_words: bool, offset: int, timeline: str | None
):
    if runtime_words:
        prog.send_readoutconfig_from_regs(
            0, "ro", freq_reg="count", t=offset / 430.08, timeline=timeline
        )
    else:
        prog.send_readoutconfig(0, "ro", t=offset / 430.08, timeline=timeline)


@pytest.mark.parametrize("runtime_words", [False, True])
@pytest.mark.parametrize("offset", [0, 5])
@pytest.mark.parametrize("transition", ["none", "join", "fork", "partial-native"])
def test_register_repeat_wait_source_rejects_config_placement(
    runtime_words, offset, transition
):
    def body(prog):
        _repeat_with_external_wait(prog)
        if transition == "join":
            prog.join("external", into="default")
        elif transition == "fork":
            prog.fork("external", into="derived")
        elif transition == "partial-native":
            prog.delay(20 / 430.08)
        _send_config(prog, runtime_words, offset, "external")

    with pytest.raises(ValueError, match="external.*wait"):
        _program(body, _readout_register_initialize)


@pytest.mark.parametrize("runtime_words", [False, True])
@pytest.mark.parametrize("offset", [0, 5])
@pytest.mark.parametrize("transition", ["join", "fork", "delay_by_timeline", "native"])
def test_register_repeat_ordinary_schedule_point_allows_config_placement(
    runtime_words, offset, transition
):
    def body(prog):
        _repeat_with_external_wait(prog)
        target = "external"
        if transition == "join":
            prog.join("external", into="default")
            target = "default"
        elif transition == "fork":
            prog.fork("external", into="derived")
            target = "derived"
        elif transition == "delay_by_timeline":
            prog.delay_by_timeline(timeline="external")
        else:
            prog.delay(70 / 430.08)
        _send_config(prog, runtime_words, offset, target)
        prog.pulse(1, "p1", timeline=target)

    (shot,) = _shots(_program(body, _readout_register_initialize))
    # Sync/join/fork consume the actual remainder55. Native delay70 still
    # advances reference from45 to115. Config is instantaneous.
    origin = 115 if transition == "native" else 100
    assert _waves(shot) == [origin + offset, origin + offset]
    assert (shot.body_end_cycle, shot.end_cycle) == (origin + 28 + offset,) * 2


@pytest.mark.parametrize("runtime_words", [False, True])
@pytest.mark.parametrize("offset", [0, 5])
def test_register_repeat_native_config_does_not_place_on_wait_source(
    runtime_words, offset
):
    def body(prog):
        _repeat_with_external_wait(prog)
        _send_config(prog, runtime_words, offset, None)
        prog.pulse(1, "p1", timeline="default")

    (shot,) = _shots(_program(body, _readout_register_initialize))
    assert _waves(shot) == [45 + offset, 45]
    assert (shot.body_end_cycle, shot.end_cycle) == (100, 100)


@pytest.mark.parametrize("actual", [0, 3])
def test_register_repeat_zero_minimum_preserves_parent_scheduling(actual):
    def initialize(prog):
        _register_initialize(prog, actual)

    def body(prog):
        prog.fork("default", into="external")
        prog.timeline_inc(100 / 430.08, timeline="external")
        prog.delay_by_timeline()
        with prog.repeat("r", "count", range_hint=(0, 4)):
            prog.pulse(0, "short", timeline="default")
            prog.timeline_inc(9 / 430.08)
            prog.delay_by_timeline()
        prog.pulse(0, "p0", timeline="default")

    (shot,) = _shots(_program(body, initialize))
    expected = [0] if actual == 0 else [0, 15, 30, 45]
    assert _waves(shot) == expected
    assert (shot.body_end_cycle, shot.end_cycle) == (100, 100)


@pytest.mark.parametrize("advance", [20, 69])
def test_partial_native_advance_keeps_wait_only_source_restricted(advance):
    def body(prog):
        prog.fork("default", into="external")
        prog.timeline_inc(100 / 430.08, timeline="external")
        prog.delay_by_timeline()
        with prog.repeat("r", "count", range_hint=(2, 4)):
            prog.timeline_inc(15 / 430.08)
            prog.delay_by_timeline()
        prog.delay(advance / 430.08)
        prog.timeline_inc(0, timeline="external")

    with pytest.raises(ValueError, match="external.*wait"):
        _program(body, _register_initialize)


@pytest.mark.parametrize("consumer", ["delay_by_timeline", "native", "repeat"])
def test_consuming_full_bound_restores_original_timeline_for_placement(consumer):
    def body(prog):
        prog.fork("default", into="external")
        prog.timeline_inc(100 / 430.08, timeline="external")
        prog.delay_by_timeline()
        with prog.repeat("r", "count", range_hint=(2, 4)):
            prog.timeline_inc(15 / 430.08)
            prog.delay_by_timeline()
        if consumer == "delay_by_timeline":
            prog.delay_by_timeline(timeline="external")
        elif consumer == "native":
            prog.delay(70 / 430.08)
        else:
            prog.delay_by_timeline(timeline="external")
            with prog.repeat("after", 2, timeline="external"):
                prog.pulse(1, "p1", timeline="external")
                prog.delay_by_timeline(timeline="external")
        prog.pulse(1, "p1", timeline="external")

    (shot,) = _shots(_program(body, _register_initialize))
    expected = (
        [100, 128, 156]
        if consumer == "repeat"
        else [115 if consumer == "native" else 100]
    )
    end = 184 if consumer == "repeat" else (143 if consumer == "native" else 128)
    assert _waves(shot) == expected
    assert (shot.body_end_cycle, shot.end_cycle) == (end, end)


def test_native_register_advance_consumes_wait_bound_before_restoring_placement():
    def initialize(prog):
        _register_initialize(prog)
        prog.add_pulse(
            1,
            "flux",
            style="const",
            freq=1000,
            phase=0,
            gain=0.5,
            length=125 / 384,
        )  # 140 tProc cycles.

    def body(prog):
        prog.fork("default", into="external")
        prog.pulse(1, "flux", timeline="external")
        prog.delay_by_timeline()
        with prog.repeat("r", "count", range_hint=(2, 4)):
            prog.pulse(0, "short", timeline="default")
            prog.timeline_inc(9 / 430.08)
            prog.delay_by_timeline()
        prog.delay_reg_auto("count")
        prog.pulse(1, "p1", timeline="external")

    (shot,) = _shots(_program(body, initialize))
    # Actual ref45 plus runtime native tail95 reaches140, then count adds3.
    assert _waves(shot) == [0, 0, 15, 30, 143]
    assert (shot.body_end_cycle, shot.end_cycle) == (171, 171)


def test_register_repeat_swept_duration_proves_latest_wait_at_each_point():
    def initialize(prog):
        _register_initialize(prog)
        prog.add_loop("scan", 3)

    def body(prog):
        prog.fork("default", into="external")
        prog.timeline_inc(100 / 430.08, timeline="external")
        prog.delay_by_timeline()
        with prog.repeat("r", "count", range_hint=(2, 4)):
            prog.timeline_inc(QickSweep1D("scan", 10 / 430.08, 20 / 430.08))
            prog.delay_by_timeline()
        prog.join("external", into="default")
        prog.pulse(1, "p1", timeline="default")

    shots = _shots(_program(body, initialize, reps=2))
    assert [_waves(s) for s in shots] == [[100], [100], [100]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [128, 128, 128] * 2
    assert [s.end_cycle - s.start_cycle for s in shots] == [128, 128, 128] * 2


@pytest.mark.parametrize("consume_after", [False, True])
def test_register_repeat_clamps_crossing_wait_bound_or_partial_consumption(
    consume_after,
):
    def initialize(prog):
        _register_initialize(prog)
        prog.add_loop("scan", 3)

    def body(prog):
        prog.fork("default", into="external")
        prog.timeline_inc((100 if consume_after else 30) / 430.08, timeline="external")
        prog.delay_by_timeline()
        with prog.repeat("r", "count", range_hint=(2, 4)):
            prog.timeline_inc(QickSweep1D("scan", 10 / 430.08, 20 / 430.08))
            prog.delay_by_timeline()
        if consume_after:
            prog.delay(75 / 430.08)

    result = _shots(_program(body, initialize))
    expected = [105, 120, 135] if consume_after else [30, 45, 60]
    assert [s.body_end_cycle - s.start_cycle for s in result] == expected


def test_register_count_updates_and_resets_across_scan_and_shots():
    def initialize(prog):
        _register_initialize(prog)
        prog.add_loop("scan", 3)

    def body(prog):
        prog.write_reg_op("count", "scan", "+", 1)
        prog.fork("default", into="external")
        prog.timeline_inc(100 / 430.08, timeline="external")
        prog.delay_by_timeline()
        with prog.repeat("r", "count", range_hint=(1, 3)):
            prog.pulse(0, "short", timeline="default")
            prog.timeline_inc(9 / 430.08)
            prog.delay_by_timeline()
        prog.join("external", into="default")
        prog.pulse(1, "p1", timeline="default")

    prog = _program(body, initialize, reps=2)
    shots = _shots(prog)
    assert [_waves(s) for s in shots] == [[0, 100], [0, 15, 100], [0, 15, 30, 100]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [128, 128, 128] * 2
    assert [s.end_cycle - s.start_cycle for s in shots] == [128, 128, 128] * 2
    prog.compile()
    assert _shots(prog) == shots


@pytest.mark.parametrize("register_outer", [False, True])
def test_nested_register_repeat_preserves_minimum_advance_for_outer_wait_bound(
    register_outer,
):
    def body(prog):
        prog.fork("default", into="external")
        prog.timeline_inc(100 / 430.08, timeline="external")
        prog.delay_by_timeline()
        with prog.repeat(
            "outer",
            "count" if register_outer else 2,
            range_hint=(2, 4) if register_outer else None,
        ):
            prog.fork("default", into="local")
            prog.delay_by_timeline(timeline="local")
            with prog.repeat(
                "inner",
                2 if register_outer else "count",
                timeline="local",
                range_hint=None if register_outer else (2, 4),
            ):
                prog.timeline_inc(15 / 430.08, timeline="local")
                prog.delay_by_timeline(timeline="local")
            prog.join("local", into="default")
            prog.delay_by_timeline()
        prog.join("external", into="default")
        prog.pulse(1, "p1", timeline="default")

    (shot,) = _shots(_program(body, _register_initialize))
    # Six actual iterations advance90; the runtime remainder is10.
    assert _waves(shot) == [100]
    assert (shot.body_end_cycle, shot.end_cycle) == (128, 128)


def test_native_pulse_tail_after_last_repeat_iteration_is_not_debited_again():
    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("r", 2):
            prog.append_macro(
                Pulse(ch=0, name="p0", t=0)
            )  # Native events do not advance default.
            prog.timeline_inc(10 / 430.08)
            prog.delay_by_timeline()

    (shot,) = _shots(_program(body))
    assert _waves(shot) == [0, 10]
    assert (shot.body_end_cycle, shot.end_cycle) == (20, 38)


@pytest.mark.parametrize("tail_sync", [False, True])
@pytest.mark.parametrize("external_end", [0, 100])
def test_repeat_allows_native_register_reference_advance(tail_sync, external_end):
    def initialize(prog):
        prog.add_reg("count")

    def body(prog):
        with prog.time_producer("count", CycleBounds(3, 3)):
            prog.write_reg("count", 3)
        prog.fork("default", into="external")
        prog.timeline_inc(external_end / 430.08, timeline="external")
        with prog.repeat("r", 2):
            prog.pulse(0, "p0")
            prog.delay_reg_auto("count")
            if tail_sync:
                prog.delay_by_timeline()
        prog.join("external", into="default")
        prog.pulse(1, "p1", timeline="default")

    prog = _program(body, initialize)
    (shot,) = _shots(prog)
    # Each iteration consumes the 28-cycle native end plus count=3 cycles.
    final_start = 100 if external_end else 62
    assert _waves(shot) == [0, 31, final_start]
    assert (shot.body_end_cycle, shot.end_cycle) == (final_start + 28,) * 2


def test_wrong_repeat_pulse_offset_is_detected_by_independent_timing_oracle():
    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("r", 2):
            prog.pulse(0, "p0", t=1 / 430.08, timeline="default")
            prog.delay_by_timeline()

    (shot,) = _shots(_program(body))
    assert _waves(shot) == [1, 30]
    with pytest.raises(AssertionError):
        assert _waves(shot) == [0, 28]


def test_nested_native_pulse_tails_survive_both_repeat_boundaries():
    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("outer", 2):
            prog.delay_by_timeline()
            with prog.repeat("inner", 2):
                prog.append_macro(Pulse(ch=0, name="p0", t=0))
                prog.timeline_inc(10 / 430.08)
                prog.delay_by_timeline()
            prog.delay_by_timeline()

    (shot,) = _shots(_program(body))
    assert _waves(shot) == [0, 10, 20, 30]
    assert (shot.body_end_cycle, shot.end_cycle) == (40, 58)


@pytest.mark.parametrize(
    "count,hint", [(2, None), ("count", (0, 2)), ("count", (2, 2))]
)
@pytest.mark.parametrize("native_prefix", [False, True])
def test_repeat_rejects_managed_pulse_overlap_across_backedge(
    count, hint, native_prefix
):
    def initialize(prog):
        _register_initialize(prog, 2)

    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("r", count, range_hint=hint):
            if native_prefix:
                prog.append_macro(Pulse(ch=0, name="p0", t=0))
                prog.timeline_inc(28 / 430.08)
            prog.pulse(0, "p0", timeline="default")
            prog.append_macro(
                Pulse(ch=0, name="p0", t=(56 if native_prefix else 28) / 430.08)
            )
            prog.delay_by_timeline()

    with pytest.raises(ValueError, match="channel 0.*overlap"):
        _program(body, initialize)


@pytest.mark.parametrize("outer,inner", [(2, 1), (1, 2), (2, 2)])
def test_nested_repeat_rejects_managed_pulse_overlap_at_either_backedge(outer, inner):
    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("outer", outer):
            prog.delay_by_timeline()
            with prog.repeat("inner", inner):
                prog.pulse(0, "p0", timeline="default")
                prog.append_macro(Pulse(ch=0, name="p0", t=28 / 430.08))
                prog.delay_by_timeline()
            prog.delay_by_timeline()

    with pytest.raises(ValueError, match="channel 0.*overlap"):
        _program(body)


@pytest.mark.parametrize("count,hint", [(0, None), (1, None), ("count", (0, 1))])
def test_repeat_without_backedge_allows_native_tail_after_managed_pulse(count, hint):
    def initialize(prog):
        _register_initialize(prog, 1)

    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("r", count, range_hint=hint):
            prog.pulse(0, "p0", timeline="default")
            prog.append_macro(Pulse(ch=0, name="p0", t=28 / 430.08))
            prog.delay_by_timeline()

    (shot,) = _shots(_program(body, initialize))
    assert _waves(shot) == ([] if count == 0 else [0, 28])
    assert (shot.body_end_cycle, shot.end_cycle) == ((0, 0) if count == 0 else (28, 56))


@pytest.mark.parametrize("register", [False, True])
def test_repeat_allows_managed_pulse_touching_previous_native_tail(register):
    def initialize(prog):
        _register_initialize(prog, 2)

    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat(
            "r", "count" if register else 2, range_hint=(0, 2) if register else None
        ):
            prog.pulse(0, "p0", timeline="default")
            prog.append_macro(Pulse(ch=0, name="p0", t=28 / 430.08))
            prog.timeline_inc(28 / 430.08)
            prog.delay_by_timeline()

    (shot,) = _shots(_program(body, initialize))
    assert _waves(shot) == [0, 28, 56, 84]
    assert (shot.body_end_cycle, shot.end_cycle) == (112, 112)


@pytest.mark.parametrize("actual", [0, 1, 2, 3, 4])
def test_nested_register_range_preserves_legal_interior_counts(actual):
    def initialize(prog):
        _register_initialize(prog, actual)

    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("outer", 2):
            prog.delay_by_timeline()
            with prog.repeat("inner", "count", range_hint=(0, 4)):
                prog.append_macro(Pulse(ch=0, name="p0", t=0))
                prog.timeline_inc(28 / 430.08)
                prog.pulse(0, "p0", timeline="default")
                prog.delay_by_timeline()
            prog.delay_by_timeline()

    (shot,) = _shots(_program(body, initialize))
    expected = [t for i in range(2 * actual) for t in (56 * i, 56 * i + 28)]
    assert _waves(shot) == expected
    assert (shot.body_end_cycle, shot.end_cycle) == (112 * actual,) * 2


@pytest.mark.parametrize("register", [False, True])
@pytest.mark.parametrize("safe", [False, True])
def test_repeat_guard_proves_swept_native_prefix_at_every_scan_point(register, safe):
    def initialize(prog):
        _register_initialize(prog, 2)
        prog.add_loop("scan", 3)

    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("outer", 2):
            prog.delay_by_timeline()
            with prog.repeat(
                "inner",
                "count" if register else 2,
                range_hint=(1, 3) if register else None,
            ):
                prog.append_macro(Pulse(ch=0, name="p0", t=0))
                prog.timeline_inc(QickSweep1D("scan", 28 / 430.08, 38 / 430.08))
                prog.pulse(0, "p0", timeline="default")
                prog.append_macro(
                    Pulse(
                        ch=0, name="p0", t=QickSweep1D("scan", 56 / 430.08, 66 / 430.08)
                    )
                )
                if safe:
                    prog.timeline_inc(28 / 430.08)
                prog.delay_by_timeline()
            prog.delay_by_timeline()

    if not safe:
        with pytest.raises(ValueError, match="channel 0.*overlap"):
            _program(body, initialize)
        return
    shots = _shots(_program(body, initialize))
    assert [_waves(s) for s in shots] == [
        [0, 28, 56, 84, 112, 140, 168, 196, 224, 252, 280, 308],
        [0, 33, 61, 89, 122, 150, 178, 211, 239, 267, 300, 328],
        [0, 38, 66, 94, 132, 160, 188, 226, 254, 282, 320, 348],
    ]
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [336, 356, 376]
    assert [s.end_cycle - s.start_cycle for s in shots] == [336, 356, 376]


@pytest.mark.parametrize("register", [False, True])
def test_nested_native_sweep_can_cross_timestamp_floor_without_losing_managed_guard(
    register,
):
    def initialize(prog):
        _register_initialize(prog, 2)
        prog.add_loop("scan", 3)

    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("outer", 2):
            prog.delay_by_timeline()
            with prog.repeat(
                "inner",
                "count" if register else 2,
                range_hint=(1, 3) if register else None,
            ):
                prog.append_macro(Pulse(ch=0, name="p0", t=0))
                prog.timeline_inc(QickSweep1D("scan", 20 / 430.08, 40 / 430.08))
                prog.delay_by_timeline()
            prog.pulse(0, "p0", t=30 / 430.08, timeline="default")
            prog.delay_by_timeline()

    shots = _shots(_program(body, initialize))
    # Inner iterations advance20/30/40; managed start is ref +30, length28.
    assert [_waves(s) for s in shots] == [
        [0, 20, 70, 98, 118, 168],
        [0, 30, 90, 118, 148, 208],
        [0, 40, 110, 138, 178, 248],
    ]
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [196, 236, 276]
    assert [s.end_cycle - s.start_cycle for s in shots] == [196, 236, 276]


@pytest.mark.parametrize("native", ["macro", "runtime-words", "auto"])
def test_repeat_guard_observes_native_pulse_entrypoints(native):
    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("r", 2):
            prog.pulse(0, "p0", timeline="default")
            if native == "macro":
                prog.append_macro(Pulse(ch=0, name="p0", t=28 / 430.08))
            elif native == "runtime-words":
                prog.append_macro(
                    PulseFromRegs(ch=0, name="p0", gain_reg="count", t=28 / 430.08)
                )
            else:
                prog.append_macro(Pulse(ch=0, name="p0", t="auto"))
            prog.delay_by_timeline()

    with pytest.raises(ValueError, match="channel 0.*overlap"):
        _program(body, _register_initialize)


def test_repeat_guard_protects_module_emission_after_native_tail():
    from zcu_tools.program.v2.modules import Pulse as ModulePulse
    from zcu_tools.program.v2.modules import PulseCfg
    from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

    pulse = ModulePulse(
        "managed",
        PulseCfg(
            ch=0,
            nqz=1,
            freq=1000,
            gain=0.5,
            waveform=ConstWaveformCfg(length=25 / 384),
        ),
    )

    def initialize(prog):
        prog.declare(pulse)

    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("r", 2):
            prog.emit(pulse)
            prog.append_macro(Pulse(ch=0, name="p0", t=28 / 430.08))
            prog.delay_by_timeline()

    with pytest.raises(ValueError, match="channel 0.*overlap"):
        _program(body, initialize)


def test_repeat_cannot_shadow_an_existing_external_name():
    def body(prog):
        prog.fork("default", into="external")
        with prog.repeat("r", 2):
            prog.fork("default", into="external")

    with pytest.raises(ValueError, match="new.*timeline"):
        _program(body)
