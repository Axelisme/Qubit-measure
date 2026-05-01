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
    
    prog_list = [
        {"CMD": "__META__", "TYPE": "LOOP_START", "NAME": "loop1", "ARGS": {"counter_reg": "r1", "n": 5}},
        {"LABEL": "loop1_start"},
        {"CMD": "REG_WR", "DST": "r1", "SRC": "imm", "LIT": "#0"},
        {"CMD": "TEST", "OP": "r1 - #5", "UF": "0"},
        {"CMD": "JUMP", "LABEL": "loop1_end", "IF": "NS"},
        {"CMD": "__META__", "TYPE": "LOOP_BODY_START", "NAME": "loop1"},
        {"CMD": "NOP"},
        {"CMD": "REG_WR", "DST": "r1", "SRC": "op", "OP": "r1+#1"},
        {"CMD": "__META__", "TYPE": "LOOP_BODY_END", "NAME": "loop1"},
        {"CMD": "JUMP", "LABEL": "loop1_start"},
        {"LABEL": "loop1_end"},
        {"CMD": "__META__", "TYPE": "LOOP_END", "NAME": "loop1"},
    ]
    labels = {"loop1_start": "0", "loop1_end": "100"}
    
    builder = IRBuilder()
    root = builder.build(prog_list, labels)
    
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
    opt_insts, opt_labels = builder.unbuild(root)
    
    # Verify emitted instructions (no markers)
    # The new IRLoop.emit() outputs:
    # REG_WR(init), Label(start), TEST, JUMP(end), [BODY...], JUMP(start), Label(end)
    
    expected_cmds = ["REG_WR"] # Init counter
    expected_cmds.append("TEST") # Stop check
    expected_cmds.append("JUMP") # CondJump out
    expected_cmds.append("NOP") # Body
    expected_cmds.append("REG_WR") # Body
    expected_cmds.append("JUMP") # Jump back

    # Note: Labels are extracted into a dictionary when creating binprog, but `emit()` outputs them as LabelInst dicts.
    cmds = [inst.get("CMD") for inst in opt_insts if "CMD" in inst]
    assert cmds == expected_cmds

def test_pipeline_roundtrip_with_normalization():
    # Test that the pipeline preserves well-formed loops
    # Use a large trip count to avoid automatic unrolling (default max is 16)
    loop = IRLoop(name="r", counter_reg="c", n=100)
    
    root = RootNode(insts=[loop], labels={"start": "0", "end": "100"})
    
    config = PipeLineConfig()
    pipeline = make_default_pipeline(config)
    
    out_ir, _ctx = pipeline(root)
    
    # Check that it's still well-formed
    assert isinstance(out_ir.insts[0], IRLoop)
    out_loop = out_ir.insts[0]
    assert out_loop.counter_reg == "c"
    assert out_loop.n == 100