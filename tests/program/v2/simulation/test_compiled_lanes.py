"""Independent lane admission through public compile/snapshot/request seams."""

from dataclasses import replace

import pytest
from qick.asm_v2 import IncReg, QickRawParam, QickSweep1D
from zcu_tools.program.v2.ir import CompiledRecord, CompiledStream
from zcu_tools.program.v2.ir.instructions import MetaInst

from tests.program.v2.simulation._compiled_support import program, request


@pytest.mark.parametrize("damage", ["start", "step"])
@pytest.mark.parametrize("target", ["registers", "waves"])
def test_sweep_definition_must_match_compiled_initial_and_update_words(damage, target):
    def initialize(p):
        p.add_loop("scan", 3)

    def body(p):
        p.pulse(0, "swept", t=QickSweep1D("scan", 0, 0.02))

    def transform(snapshot):
        records = []
        for record in snapshot.records:
            inst = record.instruction
            if (
                isinstance(inst, MetaInst)
                and inst.type == "LOOP_START"
                and inst.name == "scan"
            ):
                if target == "registers":
                    definition = next(iter(inst.info["sweeps"][target].values()))
                else:
                    # The fixture sweeps the native pulse time. Also request a
                    # gain sweep so waveform definitions have an observable use.
                    definition = next(
                        iter(next(iter(inst.info["sweeps"][target].values())).values())
                    )
                definition[damage] += 1
                inst = replace(inst, info=inst.info)
            records.append(CompiledRecord(inst, record.p_addr))
        return CompiledStream(tuple(records))

    def setup(p):
        initialize(p)
        p.add_pulse(
            0,
            "swept",
            style="const",
            freq=100,
            phase=0,
            gain=QickSweep1D("scan", 0.1, 0.5),
            length=0.1,
        )

    prog, env = program(body, setup, stream_transform=transform)
    with pytest.raises(ValueError, match="preopt.*(sweep|shell).*disagree"):
        request(prog, env)
    assert env.soc.binding.last_request is None


@pytest.mark.parametrize(
    "key", ["sweeps", "registers", "waves", "tables", "n", "start", "step"]
)
def test_missing_loop_definition_keys_fail_before_request_with_location(key):
    def initialize(p):
        p.add_loop("scan", 2)

    def transform(snapshot):
        records = []
        for record in snapshot.records:
            inst = record.instruction
            if (
                isinstance(inst, MetaInst)
                and inst.type == "LOOP_START"
                and inst.name == "scan"
            ):
                if key in ("registers", "waves"):
                    del inst.info["sweeps"][key]
                elif key in ("start", "step"):
                    del next(iter(inst.info["sweeps"]["registers"].values()))[key]
                else:
                    del inst.info[key]
                inst = replace(inst, info=inst.info)
            records.append(CompiledRecord(inst, record.p_addr))
        return CompiledStream(tuple(records))

    prog, env = program(
        lambda p: p.pulse(0, "p", t=QickSweep1D("scan", 0, 0.01)),
        initialize,
        stream_transform=transform,
    )
    with pytest.raises(ValueError, match="MetaInst.*preopt"):
        request(prog, env)
    assert env.soc.binding.last_request is None


def test_dmem_live_in_rejects_two_register_aliases_of_one_word():
    def initialize(p):
        p.add_loop("scan", 2)
        p.add_reg("read_address")
        p.add_reg("write_address")
        p.add_reg("value")
        p.write_dmem(17, 10)

    def body(p):
        p.write_reg("read_address", 17)
        p.read_dmem("value", "read_address")
        p.inc_reg("value", 1)
        p.write_reg("write_address", 17)
        p.write_dmem("write_address", "value")
        p.pulse(0, "p")

    prog, env = program(body, initialize, reps=1)
    with pytest.raises(ValueError, match="preopt.*cross-shot.*DMEM"):
        request(prog, env)
    assert env.soc.binding.last_request is None


def test_branch_selected_address_does_not_hide_a_carried_memory_word():
    def initialize(p):
        p.add_loop("scan", 2)
        p.add_reg("address", init=QickRawParam("address", 18, {}))
        p.add_reg("value")
        p.write_dmem(17, 10)
        p.write_dmem(18, 99)

    def body(p):
        with p.branch(by="scan") as branch:
            for _ in range(2):
                with branch.case():
                    p.write_reg("address", 17)
        p.read_dmem("value", "address")
        p.inc_reg("value", 1)
        p.write_dmem(17, "value")
        p.pulse(0, "p")

    prog, env = program(body, initialize, reps=1)
    with pytest.raises(ValueError, match="preopt.*cross-shot.*DMEM"):
        request(prog, env)
    assert env.soc.binding.last_request is None


def test_reps_exec_after_cannot_update_an_unused_register():
    def initialize(p):
        p.add_reg("unused", init=QickRawParam("unused", 0, {}))
        p.append_loop_after("reps", IncReg(dst="unused", src=1))

    prog, env = program(lambda p: p.pulse(0, "p"), initialize, reps=3)
    with pytest.raises(ValueError, match="preopt.*reps.*shell"):
        request(prog, env)
    assert env.soc.binding.last_request is None


def test_inner_loop_initialization_is_local_to_the_shot_and_visible_after_exit():
    def initialize(p):
        p.add_loop("scan", 2)
        p.add_reg("value")

    def body(p):
        with p.repeat("inner", 2):
            p.write_reg("value", 123)
        p.read_wmem(p.pulses["p"].get_wavenames()[0])
        p.write_reg("w_gain", "value")
        p.asm_inst({"CMD": "WPORT_WR", "DST": "0", "SRC": "r_wave", "TIME": "@0"})

    prog, env = program(body, initialize)
    result = request(prog, env)
    assert [
        result.materialize_point(i).events[0].gain for i in range(2)
    ] == pytest.approx([123 / 32766] * 2)


@pytest.mark.parametrize("key", ["n", "counter_reg"])
def test_inner_loop_definition_must_match_compiled_counter_and_count(key):
    def initialize(p):
        p.add_reg("count", init=QickRawParam("count", 2, {}))

    def transform(snapshot):
        records = []
        for record in snapshot.records:
            inst = record.instruction
            if (
                isinstance(inst, MetaInst)
                and inst.type == "LOOP_START"
                and inst.name == "inner"
            ):
                inst.info[key] = 3 if key == "n" else "r31"
                inst = replace(inst, info=inst.info)
            records.append(CompiledRecord(inst, record.p_addr))
        return CompiledStream(tuple(records))

    def body(p):
        p.trigger([0], t=0)
        with p.repeat("inner", "count", range_hint=(0, 3)):
            p.pulse(0, "p", timeline="default")

    prog, env = program(
        body, initialize, stream_transform=transform, reps=1, trigger_time=None
    )
    with pytest.raises(ValueError, match="preopt.*loop.*disagree"):
        request(prog, env)
    assert env.soc.binding.last_request is None


def test_conditional_wave_port_only_emits_on_lanes_selected_by_flags():
    def initialize(p):
        p.add_loop("scan", 3)

    def body(p):
        p.write_reg("w_gain", "scan")
        p.asm_inst({"CMD": "TEST", "OP": "w3 - #1", "UF": "1"})
        wave = p.pulses["p"].get_wavenames()[0]
        p.asm_inst(
            {
                "CMD": "WPORT_WR",
                "DST": "0",
                "SRC": "wmem",
                "ADDR": f"&{p.wave2idx[wave]}",
                "TIME": "@0",
                "IF": "Z",
            }
        )
        p.pulse(0, "p", t=0.1)

    prog, env = program(body, initialize)
    result = request(prog, env)
    assert [len(result.materialize_point(i).events) for i in range(3)] == [1, 2, 1]


def test_branch_trigger_masks_do_not_extend_other_lanes_body_duration():
    def body(p):
        with p.branch(by="scan") as branch:
            for case in range(2):
                with branch.case(), p.repeat(f"reads_{case}", 2):
                    p.trigger([0], t=0)
                    p.pulse(0, "p", timeline="default")
                    p.delay_timeline(case * 43 / 430.08)

    prog, env = program(body, lambda p: p.add_loop("scan", 2), trigger_time=None)
    result = request(prog, env)
    assert result.materialize_point(0).duration_us == pytest.approx(
        43 / 430.08 + 61 / 307.2
    )
    assert result.materialize_point(1).duration_us == pytest.approx(172 / 430.08)


def test_progress_counter_read_in_body_is_not_a_sweep_lane_input():
    def body(p):
        p.read_wmem(p.pulses["p"].get_wavenames()[0])
        p.write_reg("w_gain", "s12")
        p.asm_inst({"CMD": "WPORT_WR", "DST": "0", "SRC": "r_wave", "TIME": "@0"})

    prog, env = program(body, reps=3)
    with pytest.raises(ValueError, match="preopt.*cross-shot.*s12"):
        request(prog, env)
    assert env.soc.binding.last_request is None
