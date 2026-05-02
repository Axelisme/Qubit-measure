from typing import Any

import pytest
from zcu_tools.program.v2.ir.builder import IRBuilder
from zcu_tools.program.v2.ir.instructions import (
    GenericInst,
    JumpInst,
    RegWriteInst,
    TestInst,
)
from zcu_tools.program.v2.ir.node import BlockNode, IRLoop, RootNode
from zcu_tools.program.v2.ir.pipeline import PipeLineConfig, make_default_pipeline


def test_structural_loop_roundtrip():
    # This test verifies that we can build a structural IR from instructions,
    # and unbuild it back to instructions.

    prog_list: list[dict[str, Any]] = [
        {"CMD": "REG_WR", "DST": "r1", "SRC": "imm", "LIT": "#0"},
        {"CMD": "TEST", "OP": "r1 - #5", "UF": "0"},
        {"CMD": "JUMP", "LABEL": "loop1_end", "IF": "NS"},
        {"CMD": "NOP"},
        {"CMD": "REG_WR", "DST": "r1", "SRC": "op", "OP": "r1+#1"},
        {"CMD": "JUMP", "LABEL": "loop1_start"},
    ]
    for i, p in enumerate(prog_list):
        p["P_ADDR"] = i

    meta_infos = [
        {
            "kind": "meta",
            "type": "LOOP_START",
            "name": "loop1",
            "info": {"counter_reg": "r1", "n": 5},
            "p_addr": 0,
        },
        {"kind": "label", "name": "loop1_start", "p_addr": 0},
        {
            "kind": "meta",
            "type": "LOOP_BODY_START",
            "name": "loop1",
            "info": {},
            "p_addr": 3,
        },
        {
            "kind": "meta",
            "type": "LOOP_BODY_END",
            "name": "loop1",
            "info": {},
            "p_addr": 5,
        },
        {"kind": "label", "name": "loop1_end", "p_addr": 6},
        {
            "kind": "meta",
            "type": "LOOP_END",
            "name": "loop1",
            "info": {},
            "p_addr": 6,
        },
    ]

    builder = IRBuilder()
    root = builder.build(prog_list, {}, meta_infos)

    # Verify IR structure
    assert len(root.insts) == 1
    loop = root.insts[0]
    assert isinstance(loop, IRLoop)
    assert loop.name == "loop1"
    assert loop.counter_reg == "r1"
    assert loop.n == 5
    assert loop.start_label == "loop1_start"
    assert loop.end_label == "loop1_end"

    # Unbuild (emits instructions)
    opt_insts, opt_labels, opt_meta_infos, cursor = builder.unbuild(root)

    # Verify emitted instructions (no markers)
    # The new IRLoop.emit() outputs:
    # REG_WR(init), Label(start), TEST, JUMP(end), [BODY...], JUMP(start), Label(end)

    expected_cmds = ["REG_WR"]  # Init counter
    expected_cmds.append("TEST")  # Stop check
    expected_cmds.append("JUMP")  # CondJump out
    expected_cmds.append("NOP")  # Body
    expected_cmds.append("REG_WR")  # Body
    expected_cmds.append("JUMP")  # Jump back

    # Note: Labels are extracted into a dictionary when creating binprog, but `emit()` outputs them as LabelInst dicts.
    cmds = [inst.get("CMD") for inst in opt_insts if "CMD" in inst]
    assert cmds == expected_cmds
    assert cursor.final_p_addr == 6
    assert cursor.final_line == 8


def test_pipeline_roundtrip_with_normalization():
    # Test that the pipeline preserves well-formed loops
    # Use a large trip count to avoid automatic unrolling (default max is 16)
    loop = IRLoop(name="r", counter_reg="c", n=100)

    root = RootNode(
        insts=[loop],
    )

    pipeline = make_default_pipeline(pmem_capacity=8192)

    out_ir, _ctx = pipeline(root)

    # Check that it's still well-formed
    assert isinstance(out_ir.insts[0], IRLoop)
    out_loop = out_ir.insts[0]
    assert out_loop.counter_reg == "c"
    assert out_loop.n == 100
