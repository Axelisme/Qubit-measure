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

from tests.program._profiles import load_soccfg
from tests.program._timing_expectations import generator_duration, tproc_ceil
from tests.program.v2.simulation._compiled_support import program, request
from tests.program.v2.simulation.support import request_of
from tests.program.v2.simulation.time_observations import time_us, times_us
from tests.program.v2.wave_observations import analyze


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


@pytest.mark.parametrize("pmem_size", [2048, 4096])
def test_forward_cond_jump_uses_each_sweep_lanes_reference_time(pmem_size):
    def body(p):
        p.cond_jump("scan", "scan", "Z", "-", 0)
        p.pulse(0, "p", t=0, tag="fall-through")
        p.delay(0.4)
        p.label("scan")  # Also the name of the native loop.
        p.pulse(0, "p", t=0, tag="joined")
        p.trigger([0], t=0)

    profile = load_soccfg(n_gens=2, n_readouts=1)
    profile["tprocs"][0]["pmem_size"] = pmem_size
    prog, env = program(
        body,
        lambda p: p.add_loop("scan", 2),
        trigger_time=None,
        soccfg=profile,
    )
    result = request(prog, env)
    step = result.time_base.period_ticks(prog.tproccfg["f_time"])
    for i, (sources, shot) in enumerate(
        zip(
            [["joined"], ["fall-through", "joined"]],
            analyze(prog).shots[:2],
            strict=True,
        )
    ):
        point = result.materialize_point(i)
        assert [event.source for event in point.events] == sources
        assert [event.start_ticks for event in point.events] == [
            (event.cycle - shot.start_cycle) * step
            for event in shot.events
            if event.kind == "wave" and event.port == prog.soccfg["gens"][0]["tproc_ch"]
        ]
        adc = prog.soccfg["readouts"][0]
        adc_bit = 1 << adc["trigger_bit"] if adc["trigger_type"] == "dport" else 1
        assert tuple(point.readouts[0].trigger_ticks) == tuple(
            (event.cycle - shot.start_cycle) * step
            for event in shot.events
            if event.port == adc["trigger_port"]
            and event.kind == ("data" if adc["trigger_type"] == "dport" else "trigger")
            and (event.value or 0) & adc_bit
        )


def test_lanes_wait_for_their_own_label_including_an_unconditional_jump():
    def body(p):
        p.cond_jump("first", "scan", "Z")
        p.cond_jump("second", "scan", "Z", "-", 1)
        p.jump("join")
        p.label("first")
        p.pulse(0, "p", t=0, tag="first")
        p.delay(0.4)
        p.jump("join")
        p.label("second")
        p.pulse(0, "p", t=0, tag="second")
        p.delay(0.8)
        p.label("join")
        p.pulse(0, "p", t=0, tag="joined")

    prog, env = program(body, lambda p: p.add_loop("scan", 3))
    result = request(prog, env)
    assert [
        [event.source for event in result.materialize_point(i).events] for i in range(3)
    ] == [["first", "joined"], ["second", "joined"], ["joined"]]


def test_jump_lanes_each_deliver_the_compiled_trigger_count():
    def body(p):
        p.cond_jump("short", "scan", "Z")
        p.pulse(0, "p", t=0)
        p.trigger([0], t=0, timeline="default")
        p.jump("join")
        p.label("short")
        p.pulse(0, "p", t=0.1)
        p.trigger([0], t=0.1, timeline="default")
        p.label("join")

    prog, env = program(body, lambda p: p.add_loop("scan", 2), trigger_time=None)
    assert prog.ro_chs[0]["trigs"] == 1
    result = request(prog, env)
    step = result.time_base.period_ticks(prog.tproccfg["f_time"])
    assert [
        tuple(result.materialize_point(i).readouts[0].trigger_ticks) for i in range(2)
    ] == [(tproc_ceil(prog.soccfg, 0.1) * step,), (0,)]


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("target", "target label"),
        ("condition", "condition"),
        ("jump-position", "position"),
        ("label-position", "position"),
        ("F", "external flag"),
        ("NF", "external flag"),
    ],
)
def test_forward_jump_snapshot_mismatch_is_rejected(defect, reason):
    def transform(snapshot):
        records = list(snapshot.records)
        for i, record in enumerate(records):
            inst = record.instruction
            if not isinstance(inst, MetaInst):
                continue
            if inst.type == "CONTROL_JUMP":
                if defect == "target":
                    inst = replace(inst, info=inst.info | {"target": "other"})
                elif defect in ("condition", "F", "NF"):
                    test = "NZ" if defect == "condition" else defect
                    inst = replace(inst, info=inst.info | {"test": test})
                elif defect == "jump-position":
                    records[i] = CompiledRecord(inst, record.p_addr + 1)
                    break
                else:
                    continue
            elif inst.type != "CONTROL_LABEL" or defect != "label-position":
                continue
            records[i] = CompiledRecord(
                inst, record.p_addr + (defect == "label-position")
            )
            break
        return CompiledStream(tuple(records))

    def body(p):
        p.cond_jump("join", "scan", "Z")
        p.pulse(0, "p", t=0)
        p.label("join")
        p.pulse(0, "p", t=0.4)
        p.label("other")

    prog, env = program(
        body,
        lambda p: p.add_loop("scan", 2),
        stream_transform=transform,
        trigger_time=0.4,
    )
    with pytest.raises(
        ValueError, match=rf"MetaInst.*preopt.*loops=.*CONTROL_.*{reason}"
    ):
        request(prog, env)


def test_unmarked_body_jump_keeps_its_rejection_even_when_all_lanes_skip_it():
    def body(p):
        p.cond_jump("join", "s0", "Z")
        p.asm_inst({"CMD": "JUMP", "LABEL": "HERE"})
        p.label("join")
        p.pulse(0, "p")

    prog, env = program(body)
    with pytest.raises(
        ValueError,
        match=r"JumpInst at preopt \d+, loops=.*cases=\(\), source='': unsupported instruction effect",
    ):
        request(prog, env)


@pytest.mark.parametrize("skip", [False, True])
def test_condition_on_measurement_register_is_rejected_without_executing_feedback(skip):
    def body(p):
        if skip:
            p.cond_jump("join", "s0", "Z")
        p.cond_jump("join", "s_port_l", "Z")
        p.label("join")
        p.pulse(0, "p")

    prog, env = program(body)
    with pytest.raises(ValueError, match=r"TestInst at preopt.*undefined register s8"):
        request(prog, env)


def test_skipping_a_definition_does_not_hide_a_cross_shot_live_in():
    def initialize(p):
        p.add_loop("scan", 2)
        p.add_reg("carry", init=QickRawParam("carry", 0, {}))

    def body(p):
        p.cond_jump("join", "scan", "Z")
        p.write_reg("carry", 12)
        p.label("join")
        p.write_reg("s1", "carry")
        p.pulse(0, "p")

    prog, env = program(body, initialize)
    with pytest.raises(
        ValueError, match=r"RegWriteInst at preopt.*cross-shot.*live-in"
    ):
        request(prog, env)


@pytest.mark.parametrize("sync_at_end", [False, True])
def test_nested_native_repeats_keep_each_event_and_iteration_coordinate(sync_at_end):
    def body(p):
        p.trigger([0], t=0)
        with p.repeat("outer", 2):
            with p.repeat("inner", 2):
                p.pulse(0, "p", timeline="default", tag="repeated")
                if sync_at_end:
                    p.delay_by_timeline()
            if sync_at_end:
                p.delay_by_timeline()

    prog, env = program(body, trigger_time=None)
    result = request(prog, env)
    events = result.materialize_point(0).events
    step = env.soccfg.cycles2us(
        tproc_ceil(env.soccfg, generator_duration(env.soccfg, 0.1))
    )
    assert len(events) == 4
    assert [e.source for e in events] == ["repeated"] * 4
    assert [time_us(result.time_base, e.start_ticks) for e in events] == pytest.approx(
        [i * step for i in range(4)]
    )
    assert [e.iterations for e in events] == [
        (("outer", 0), ("inner", 0)),
        (("outer", 0), ("inner", 1)),
        (("outer", 1), ("inner", 0)),
        (("outer", 1), ("inner", 1)),
    ]
    assert times_us(result.time_base, result.readouts[0].trigger_ticks) == (0,)


@pytest.mark.parametrize("sync_at_end", [False, True])
def test_register_repeat_zero_guard_and_counts_are_local_to_each_scan_point(
    sync_at_end,
):
    loader = LoadValue("counts", [0, 1, 2], "scan", "count")

    def initialize(p):
        p.add_loop("scan", 3)
        p.declare(loader)

    def body(p):
        p.trigger([0], t=0)
        p.emit(loader)
        with p.repeat("inner", "count", range_hint=(0, 2)):
            p.pulse(0, "p", timeline="default", tag="repeated")
            if sync_at_end:
                p.delay_by_timeline()
        p.pulse(0, "p", timeline="default", tag="final")

    # Counts 0..2 leave two pulse advances of shell WAIT slack.
    prog, env = program(body, initialize, trigger_time=None, relax_delay=0.2)
    result = request(prog, env)
    step = env.soccfg.cycles2us(
        tproc_ceil(env.soccfg, generator_duration(env.soccfg, 0.1))
    )
    assert result.point_names == ("scan",)
    for index in range(3):
        events = result.materialize_point(index).events
        assert [e.source for e in events] == ["repeated"] * index + ["final"]
        assert [
            time_us(result.time_base, e.start_ticks) for e in events
        ] == pytest.approx([i * step for i in range(index + 1)])
        assert [e.iterations for e in events[:-1]] == [
            (("inner", i),) for i in range(index)
        ]


@pytest.mark.parametrize("align", [False, True])
def test_variable_repeats_in_cases_preserve_numeric_slot_times(align):
    def initialize(p):
        p.add_loop("scan", 2)
        p.declare_gen(1, nqz=1)
        p.add_pulse(1, "drive", style="const", freq=80, phase=0, gain=0.2, length=0.1)

    def body(p):
        p.pulse(0, "p", t=0, tag="carrier")
        p.trigger([0], t=0)
        with p.branch(by="scan", align_timeline=align) as cases:
            with (
                cases.case(0),
                p.repeat("outer", 2),
                p.repeat("inner", 2),
            ):
                p.pulse(1, "drive", tag="repeated")
            with cases.case(1):
                step_cycles = tproc_ceil(p.soccfg, generator_duration(p.soccfg, 0.1, 1))
                p.timeline_inc(p.cycles2us(step_cycles))
        p.pulse(1, "drive", timeline="default", tag="final")

    prog, env = program(body, initialize, trigger_time=None)
    result = request(prog, env)
    step = env.soccfg.cycles2us(
        tproc_ceil(env.soccfg, generator_duration(env.soccfg, 0.1, 1))
    )
    for index in range(2):
        events = [
            e for e in result.materialize_point(index).events if e.source != "carrier"
        ]
        expected = (
            [i * step for i in range(5)] if index == 0 else [(4 if align else 1) * step]
        )
        assert [
            time_us(result.time_base, e.start_ticks) for e in events
        ] == pytest.approx(expected)
        assert [e.iterations for e in events[:-1]] == (
            [(("outer", o), ("inner", i)) for o in range(2) for i in range(2)]
            if index == 0
            else []
        )
    assert result.reps == 2


def test_native_branch_selects_each_lane_and_debits_runtime_external_timeline():
    def initialize(p):
        p.add_loop("scan", 3)
        p.declare_gen(1, nqz=1)
        p.add_pulse(1, "tail", style="const", freq=80, phase=0, gain=0.2, length=0.1)

    def body(p):
        p.trigger([0], t=0)
        p.fork("default", into="external")
        p.timeline_inc(p.cycles2us(100), timeline="external")
        with p.branch(by="scan", align_timeline=True) as branch:
            for index in range(3):
                with branch.case(index):
                    p.pulse(0, "p", tag=f"case-{index}")
                    p.timeline_inc(p.cycles2us(10 + index * 10))
        p.join("external", into="default")
        p.pulse(1, "tail", timeline="default", tag="tail")

    prog, env = program(body, initialize, trigger_time=None)
    result = request(prog, env)
    for index in range(3):
        events = result.materialize_point(index).events
        assert [e.source for e in events] == [f"case-{index}", "tail"]
        assert [
            time_us(result.time_base, e.start_ticks) for e in events
        ] == pytest.approx([0, env.soccfg.cycles2us(100)])


@pytest.mark.parametrize("count", [0, 1, 3])
def test_fixed_repeat_acquires_without_a_manual_read_count(count):
    def body(p):
        with p.repeat("inner", count):
            p.trigger([0], timeline="default")
            p.pulse(0, "p", timeline="default")
            p.timeline_inc(0.2)
            p.delay_by_timeline()

    prog, env = program(body, trigger_time=None, reps=1)
    if count == 0:
        with pytest.raises(ValueError, match="ADC 0.*zero-read"):
            request(prog, env)
    else:
        result = request(prog, env)
        assert len(result.readouts[0].trigger_ticks) == count


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
        p.timeline_inc("left_time", timeline="left")
        p.timeline_inc("right_time", timeline="right")
        p.timeline_inc(p.cycles2us(25))
        p.join("left", "right", into="default")
        p.delay(p.cycles2us(20))
        p.delay_by_timeline()
        p.pulse(0, "p", timeline="default")

    prog, env = program(body, initialize, trigger_time=None)
    result = request(prog, env)
    assert [
        time_us(result.time_base, result.materialize_point(i).events[0].start_ticks)
        for i in range(3)
    ] == pytest.approx([env.soccfg.cycles2us(cycles) for cycles in (30, 30, 25)])


def test_simulator_checks_each_shot_for_triggers_emitted_by_raw_assembly():
    def body(p):
        p.trigger([0], t=0)
        p.pulse(0, "p", timeline="default")
        with p.branch(by="scan", align_timeline=True) as branch, branch.case(1):
            adc = p.soccfg["readouts"][0]
            tick = p.us2cycles(0.25)
            p.asm_inst(
                {
                    "CMD": "DPORT_WR",
                    "DST": str(adc["trigger_port"]),
                    "SRC": "imm",
                    "DATA": str(1 << adc["trigger_bit"]),
                    "TIME": f"@{tick}",
                }
            )

    prog, env = program(body, lambda p: p.add_loop("scan", 2), trigger_time=None)
    # Case zero matches the layout. Case one sends an uncounted ADC trigger.
    assert prog.ro_chs[0]["trigs"] == 1
    with pytest.raises(ValueError, match="ADC 0: static=1, actual=2 reads"):
        request(prog, env)
    assert env.soc.binding.last_request is None


def test_balanced_branch_repeats_match_each_adc_static_count_in_every_case():
    def initialize(p):
        p.add_loop("scan", 3)
        p.declare_readout(1, length=0.1)
        p.add_readoutconfig(1, "ro1", freq=100, gen_ch=0)

    def body(p):
        p.send_readoutconfig(1, "ro1", t=0)
        p.delay_by_timeline()
        with p.repeat("reads", 3):
            with p.branch(by="scan", align_timeline=True) as branch:
                for index in range(3):
                    with branch.case(index) as case:
                        p.fork(case, into="capture")
                        p.trigger([0, 1], t=0, timeline="capture")
                        p.pulse(0, "p", tag=f"case-{index}")
                        # Add 44 tProc cycles after the pulse so both ADC
                        # windows finish before the next trigger.
                        p.timeline_inc(p.cycles2us(44))
            p.delay_by_timeline()

    prog, env = program(
        body,
        initialize,
        n_readouts=2,
        routes=((0, 0), (0, 1)),
        trigger_time=None,
    )
    spacing = env.soccfg.cycles2us(
        tproc_ceil(env.soccfg, generator_duration(env.soccfg, 0.1)) + 44
    )
    static = {ch: cfg["trigs"] for ch, cfg in prog.ro_chs.items()}
    assert static == {0: 3, 1: 3}
    result = request(prog, env)
    for index in range(3):
        point = result.materialize_point(index)
        assert [e.source for e in point.events] == [f"case-{index}"] * 3
        for readout in point.readouts:
            assert times_us(result.time_base, readout.trigger_ticks) == pytest.approx(
                tuple(i * spacing for i in range(3))
            )
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
                    p.read_input(0)
                elif effect == "wait":
                    p.wait(0.1)
                elif effect == "clock":
                    p.write_reg("s1", "s11")
                else:
                    p.asm_inst({"CMD": "JUMP", "LABEL": "HERE"})
            with branch.case(1):
                p.pulse(0, "p")

    prog, env = program(body, lambda p: p.add_loop("scan", 2))
    with pytest.raises(ValueError, match=rf"{instruction}.*preopt.*case-0"):
        request(prog, env)
    assert env.soc.binding.last_request is None
