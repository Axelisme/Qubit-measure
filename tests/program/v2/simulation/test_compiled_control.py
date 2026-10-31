"""Structured compiled control through public compilation and numeric requests."""

from dataclasses import replace

import numpy as np
import pytest
from qick.asm_v2 import QickRawParam
from zcu_tools.program.v2.ir import CompiledRecord, CompiledStream
from zcu_tools.program.v2.ir.instructions import MetaInst, TestInst, WaitInst
from zcu_tools.program.v2.ir.labels import Label, LabelRef
from zcu_tools.program.v2.ir.operands import AluExpr, AluOp, Immediate, Register
from zcu_tools.program.v2.modules import LoadValue

from tests.program.v2.simulation._compiled_support import program, request
from tests.program.v2.simulation.support import request_of


@pytest.mark.parametrize("align_timeline", [True, False])
def test_public_acquire_skips_undeclared_branch_points_without_padding(align_timeline):
    def initialize(p):
        p.add_loop("scan", 3)
        p.declare_gen(1, nqz=1)
        p.add_pulse(1, "drive", style="const", freq=80, phase=0, gain=0.2, length=0.1)

    def body(p):
        with (
            p.branch(by="scan", align_timeline=align_timeline) as branch,
            branch.case(1),
        ):
            p.pulse(1, "drive", tag="selected-drive")
        # The routed ADC always needs a carrier, including undeclared points.
        p.pulse(0, "p", t=0, tag="terminal")
        p.trigger([0], t=0)

    prog, env = program(
        body, initialize, trigger_time=None, n_readouts=2, routes=((0, 0), (1, 1))
    )
    try:
        result = prog.acquire(env.soc, progress=False)
        np.testing.assert_array_equal(result[0], np.zeros((1, 3, 2)))
        observed = request_of(env)
        points = [observed.materialize_point(index) for index in range(3)]
        assert [[event.source for event in point.events] for point in points] == [
            ["terminal"],
            ["selected-drive", "terminal"],
            ["terminal"],
        ]
        assert points[1].events[0].gain == pytest.approx(0.2, abs=1e-4)
    finally:
        env.devices.close_all_devices()


def test_nested_native_repeats_keep_each_event_and_iteration_coordinate():
    def body(p):
        p.trigger([0], t=0)
        with p.repeat("outer", 2), p.repeat("inner", 2):
            p.pulse(0, "p", timeline="default", tag="repeated")

    prog, env = program(body, trigger_time=None)
    result = request(prog, env)
    events = result.materialize_point(0).events
    assert len(events) == 4
    assert [e.source for e in events] == ["repeated"] * 4
    assert [e.start_us for e in events] == pytest.approx(
        [0, 43 / 430.08, 86 / 430.08, 129 / 430.08]
    )
    assert [e.iterations for e in events] == [
        (("outer", 0), ("inner", 0)),
        (("outer", 0), ("inner", 1)),
        (("outer", 1), ("inner", 0)),
        (("outer", 1), ("inner", 1)),
    ]
    assert result.readouts[0].trigger_us == (0,)


def test_register_repeat_zero_guard_and_counts_are_local_to_each_scan_point():
    loader = LoadValue("counts", [0, 1, 2], "scan", "count")

    def initialize(p):
        p.add_loop("scan", 3)
        p.declare(loader)

    def body(p):
        p.trigger([0], t=0)
        p.emit(loader)
        with p.repeat("inner", "count", range_hint=(0, 2)):
            p.pulse(0, "p", timeline="default", tag="repeated")
        p.pulse(0, "p", tag="final")

    # Counts 0..2 leave 86 cycles of shell WAIT slack; allow that exact budget.
    prog, env = program(body, initialize, trigger_time=None, relax_delay=0.2)
    result = request(prog, env)
    assert result.point_names == ("scan",)
    for index in range(3):
        events = result.materialize_point(index).events
        assert [e.source for e in events] == ["repeated"] * index + ["final"]
        assert [e.start_us for e in events] == pytest.approx(
            [i * 43 / 430.08 for i in range(index + 1)]
        )
        assert [e.iterations for e in events[:-1]] == [
            (("inner", i),) for i in range(index)
        ]


def test_native_branch_selects_each_lane_and_debits_runtime_external_timeline():
    def initialize(p):
        p.add_loop("scan", 3)
        p.declare_gen(1, nqz=1)
        p.add_pulse(1, "tail", style="const", freq=80, phase=0, gain=0.2, length=0.1)

    def body(p):
        p.trigger([0], t=0)
        p.fork("default", into="external")
        p.delay_timeline(100 / 430.08, timeline="external")
        with p.branch(by="scan", align_timeline=True) as branch:
            for index in range(3):
                with branch.case(index):
                    p.pulse(0, "p", tag=f"case-{index}")
                    p.delay_timeline((10 + index * 10) / 430.08)
        p.join("external", into="default")
        p.pulse(1, "tail", timeline="default", tag="tail")

    prog, env = program(body, initialize, trigger_time=None)
    result = request(prog, env)
    for index in range(3):
        events = result.materialize_point(index).events
        assert [e.source for e in events] == [f"case-{index}", "tail"]
        assert [e.start_us for e in events] == pytest.approx([0, 100 / 430.08])


@pytest.mark.parametrize("register_count", [False, True])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_repeat_checks_actual_adc_reads_against_unchanged_static_layout(
    register_count, count
):
    loader = LoadValue("counts", [count, count], "shot", "count")

    def initialize(p):
        if register_count:
            p.add_loop("shot", 2)
            p.declare(loader)

    def body(p):
        if register_count:
            p.emit(loader)
        with p.repeat(
            "inner",
            "count" if register_count else count,
            **({"range_hint": (0, 3)} if register_count else {}),
        ):
            p.trigger([0], t=0)
            p.pulse(0, "p", timeline="default")

    prog, env = program(body, initialize, trigger_time=None, reps=1)
    static = prog.ro_chs[0]["trigs"]
    if count == 1:
        result = request(prog, env)
        assert len(result.readouts[0].trigger_us) == static == 1
        if register_count:
            assert result.point_dims == (2,)
            assert [
                len(result.materialize_point(i).readouts[0].trigger_us)
                for i in range(2)
            ] == [1, 1]
    elif count == 0 and not register_count:
        with pytest.raises(ValueError, match="ADC 0.*zero-read"):
            request(prog, env)
    else:
        with pytest.raises(
            ValueError, match=rf"ADC 0: static={static}, actual={count} reads"
        ):
            request(prog, env)
    assert prog.ro_chs[0]["trigs"] == static
    assert env.soc.binding.last_request is None


def test_register_repeat_rejects_point_dependent_adc_counts_against_static_layout():
    loader = LoadValue("counts", [1, 3], "shot", "count")

    def initialize(p):
        p.add_loop("shot", 2)
        p.declare(loader)

    def body(p):
        p.emit(loader)
        with p.repeat("inner", "count", range_hint=(1, 3)):
            p.trigger([0], t=0)
            p.pulse(0, "p", timeline="default")

    prog, env = program(body, initialize, trigger_time=None, reps=1)
    static = prog.ro_chs[0]["trigs"]
    with pytest.raises(ValueError, match=rf"ADC 0: static={static}, actual=3 reads"):
        request(prog, env)
    assert prog.ro_chs[0]["trigs"] == static
    assert env.soc.binding.last_request is None
    assert env.soc.binding.rounds == ()


def test_runtime_timeline_max_floor_fork_join_and_sync_use_selected_table_values():
    left = LoadValue("left", [10, 30, 20], "scan", "left_time")
    right = LoadValue("right", [30, 10, 20], "scan", "right_time")

    def initialize(p):
        p.add_loop("scan", 3)
        p.declare(left)
        p.declare(right)

    def body(p):
        p.trigger([0], t=0)
        p.fork("default", into=["left", "right"])
        p.emit(left)
        p.emit(right)
        p.delay_timeline("left_time", timeline="left")
        p.delay_timeline("right_time", timeline="right")
        p.delay_timeline(25 / 430.08)
        p.join("left", "right", into="default")
        p.delay(20 / 430.08)
        p.sync()
        p.pulse(0, "p", timeline="default")

    prog, env = program(body, initialize, trigger_time=None)
    result = request(prog, env)
    assert [
        result.materialize_point(i).events[0].start_us for i in range(3)
    ] == pytest.approx([30 / 430.08, 30 / 430.08, 25 / 430.08])


@pytest.mark.parametrize("counts", [(2, 0), (0, 2)])
def test_scan_total_cannot_mask_per_shot_adc_read_count_mismatch(counts):
    loader = LoadValue("counts", counts, "scan", "count")

    def initialize(p):
        p.add_loop("scan", 2)
        p.declare(loader)

    def body(p):
        p.emit(loader)
        with p.repeat("inner", "count", range_hint=(0, 2)):
            p.trigger([0], t=0)
            p.pulse(0, "p", timeline="default")

    prog, env = program(body, initialize, trigger_time=None)
    assert prog.ro_chs[0]["trigs"] == 1
    with pytest.raises(ValueError, match=rf"ADC 0: static=1, actual={counts[0]} reads"):
        request(prog, env)
    assert env.soc.binding.last_request is None


def test_branch_validates_each_executed_lane_not_total_static_case_triggers():
    def body(p):
        p.trigger([0], t=0)
        p.pulse(0, "p", timeline="default")
        with p.branch(by="scan", align_timeline=True) as branch, branch.case(0):
            p.trigger([0], t=43 / 430.08)
            p.pulse(0, "p")

    prog, env = program(body, lambda p: p.add_loop("scan", 2), trigger_time=None)
    # Lane zero has two reads, matching the two statically emitted triggers.
    # Lane one has only one; admission must not return a partially valid request.
    assert prog.ro_chs[0]["trigs"] == 2
    with pytest.raises(ValueError, match="ADC 0: static=2, actual=1 reads"):
        request(prog, env)
    assert env.soc.binding.last_request is None


def test_balanced_branch_repeats_match_each_adc_static_count_in_every_case():
    def initialize(p):
        p.add_loop("scan", 3)
        p.declare_readout(1, length=0.1)
        p.add_readoutconfig(1, "ro1", freq=100, gen_ch=0)

    def body(p):
        p.send_readoutconfig(1, "ro1", t=0)
        with p.repeat("reads", 3), p.branch(by="scan", align_timeline=True) as branch:
            for index in range(3):
                with branch.case(index):
                    p.trigger([0, 1], t=0)
                    p.pulse(0, "p", tag=f"case-{index}")

    prog, env = program(
        body,
        initialize,
        n_readouts=2,
        routes=((0, 0), (0, 1)),
        trigger_time=None,
    )
    static = {ch: cfg["trigs"] for ch, cfg in prog.ro_chs.items()}
    assert static == {0: 3, 1: 3}
    result = request(prog, env)
    for index in range(3):
        point = result.materialize_point(index)
        assert [e.source for e in point.events] == [f"case-{index}"] * 3
        for readout in point.readouts:
            assert readout.trigger_us == pytest.approx((0, 43 / 430.08, 86 / 430.08))
    assert {ch: cfg["trigs"] for ch, cfg in prog.ro_chs.items()} == static


@pytest.mark.parametrize("defect", ["missing-end", "case-order", "missing-selector"])
def test_malformed_branch_metadata_fails_before_numeric_request(defect):
    def transform(snapshot):
        records = []
        changed = False
        for record in snapshot.records:
            inst = record.instruction
            if isinstance(inst, MetaInst):
                if defect == "missing-end" and inst.type == "BRANCH_END":
                    continue
                if (
                    defect == "case-order"
                    and inst.type == "BRANCH_CASE_START"
                    and not changed
                ):
                    inst = replace(inst, name="1")
                    changed = True
                if defect == "missing-selector" and inst.type == "BRANCH_START":
                    inst = replace(inst, info={})
            records.append(CompiledRecord(inst, record.p_addr))
        return CompiledStream(tuple(records))

    def body(p):
        with p.branch(by="scan", align_timeline=True) as branch:
            for index in range(2):
                with branch.case(index):
                    p.pulse(0, "p")

    prog, env = program(
        body, lambda p: p.add_loop("scan", 2), stream_transform=transform
    )
    reason = {
        "missing-end": "missing/unbalanced",
        "case-order": "unordered branch case",
        "missing-selector": "missing branch selector",
    }[defect]
    with pytest.raises(ValueError, match=rf"MetaInst.*preopt.*{reason}"):
        request(prog, env)
    assert env.soc.binding.last_request is None


@pytest.mark.parametrize("defect", ["selector", "target", "effect", "tail-effect"])
def test_branch_dispatch_instructions_must_match_marked_cases(defect):
    def transform(snapshot):
        records = list(snapshot.records)
        start = next(
            i
            for i, r in enumerate(records)
            if isinstance(r.instruction, MetaInst)
            and r.instruction.type == "BRANCH_START"
        )
        end = next(
            i
            for i, r in enumerate(records)
            if isinstance(r.instruction, MetaInst)
            and r.instruction.type == "BRANCH_END"
        )
        if defect == "selector":
            index = next(
                i
                for i in range(start, end)
                if isinstance(records[i].instruction, TestInst)
            )
            inst = TestInst(AluExpr(Register("s0"), AluOp.SUB, Immediate(0)), uf=True)
        elif defect == "target":
            index = start + 1
            inst = replace(
                records[index].instruction,
                label=LabelRef(Label("_timeline_branch_0_end")),
            )
        else:
            index = start + 1 if defect == "effect" else end - 1
            inst = WaitInst(c_op="time")
        records[index] = CompiledRecord(inst, records[index].p_addr)
        return CompiledStream(tuple(records))

    def body(p):
        with p.branch(by="scan", align_timeline=True) as branch:
            for index in range(2):
                with branch.case(index):
                    p.pulse(0, "p")

    def triggered_body(p):
        p.trigger([0], t=0)
        body(p)

    prog, env = program(
        triggered_body,
        lambda p: p.add_loop("scan", 2),
        trigger_time=None,
        stream_transform=transform,
    )
    instruction = {
        "selector": "TestInst",
        "target": "RegWriteInst",
        "effect": "WaitInst",
        "tail-effect": "WaitInst",
    }[defect]
    with pytest.raises(
        ValueError, match=rf"{instruction}.*preopt.*loops=.*branch dispatch"
    ):
        request(prog, env)
    assert env.soc.binding.last_request is None
    assert env.soc.binding.rounds == ()


@pytest.mark.parametrize("reps", [1, 2])
def test_structured_branch_memory_live_in_is_rejected(reps):
    def initialize(p):
        p.add_loop("scan", 2)
        p.add_reg("carry", init=QickRawParam("carry", 0, {}))

    def body(p):
        with p.branch(by="scan", align_timeline=True) as branch:
            for index, amount in enumerate([1, 2]):
                with branch.case(index):
                    p.inc_reg("carry", amount)
        wave = p.pulses["p"].get_wavenames()[0]
        p.read_wmem(wave)
        p.write_reg("w_gain", "carry")
        p.write_wmem(wave)
        p.pulse(0, "p")

    prog, env = program(body, initialize, reps=reps)
    with pytest.raises(ValueError, match="preopt.*cross-shot.*live-in"):
        request(prog, env)
    assert env.soc.binding.last_request is None


@pytest.mark.parametrize(
    "effect,instruction",
    [
        ("measurement", "DportReadInst"),
        ("wait", "WaitInst"),
        ("clock", "RegWriteInst"),
        ("jump", "JumpInst"),
    ],
)
def test_unsupported_effect_inside_marked_case_retains_case_diagnostics(
    effect, instruction
):
    def body(p):
        with p.branch(by="scan", align_timeline=True) as branch:
            with branch.case(0):
                p.pulse(0, "p", tag="drive")
                if effect == "measurement":
                    p.read_and_jump(0, "I", 0, ">=", "done")
                    p.label("done")
                elif effect == "wait":
                    p.wait(0.1)
                elif effect == "clock":
                    p.write_reg("s1", "s11")
                else:
                    p.jump("HERE")
            with branch.case(1):
                p.pulse(0, "p")

    prog, env = program(body, lambda p: p.add_loop("scan", 2))
    with pytest.raises(ValueError, match=rf"{instruction}.*preopt.*case-0"):
        request(prog, env)
    assert env.soc.binding.last_request is None
