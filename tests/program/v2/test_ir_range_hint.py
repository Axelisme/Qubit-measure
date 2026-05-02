from __future__ import annotations
from zcu_tools.program.v2.ir.node import IRLoop
from zcu_tools.program.v2.ir.factory import InstructionStream, parse_root
from zcu_tools.program.v2.ir.instructions import MetaInst, RegWriteInst


def test_loop_range_hint_preservation():
    """Verify that range_hint is preserved through IRLoop construction and emission."""
    # 1. Manually create a loop with range_hint
    loop = IRLoop(name="test_loop", counter_reg="r0", n="r_count", range_hint=(10, 10))
    assert loop.range_hint == (10, 10)


def test_parse_loop_restores_range_hint():
    """Verify that the parser restores range_hint from MetaInst ARGS."""
    insts = [
        MetaInst(
            type="LOOP_START",
            name="loop",
            args={"counter_reg": "r0", "n": "r_count", "range_hint": (5, 5)},
        ),
        MetaInst(type="LOOP_BODY_START", name="loop"),
        RegWriteInst(dst="r1", src="imm", lit="#1"),
        MetaInst(type="LOOP_BODY_END", name="loop"),
        MetaInst(type="LOOP_END", name="loop"),
    ]

    stream = InstructionStream(insts)
    root = parse_root(stream)

    assert isinstance(root.insts[0], IRLoop)
    loop = root.insts[0]
    # QICK might store hint as list, but we want tuple in IRLoop
    assert loop.range_hint == (5, 5)


def test_emit_loop_preserves_range_hint_in_meta():
    """Verify that emitting IRLoop puts range_hint back into MetaInst."""
    loop = IRLoop(name="test", counter_reg="r0", n=100, range_hint=(100, 100))
    inst_list = []
    loop.emit(inst_list)

    # Find LOOP_START meta
    start_meta = next(
        inst
        for inst in inst_list
        if isinstance(inst, MetaInst) and inst.type == "LOOP_START"
    )
    assert start_meta.args["range_hint"] == (100, 100)
