import pytest
from zcu_tools.program.v2.ir.builder import IRBuilder
from zcu_tools.program.v2.ir.instructions import TestInst, JumpInst, RegWriteInst, GenericInst
from zcu_tools.program.v2.ir.node import IRLoop, RootNode, BlockNode
from zcu_tools.program.v2.ir.pipeline import PipeLineConfig, make_default_pipeline

def test_structural_loop_roundtrip():
    # This test verifies that we can build a structural IR from instructions,
    # and unbuild it back to instructions.
    # Note: Structural markers (__META__) are currently NOT preserved in unbuild/emit,
    # because emit() is intended for QICK execution.
    
    prog_list = [
        {"CMD": "__META__", "TYPE": "LOOP_START", "NAME": "loop1", "ARGS": {"trip_count": 5}},
        {"CMD": "__META__", "TYPE": "LOOP_SECTION", "NAME": "initial"},
        {"CMD": "REG_WR", "DST": "r1", "SRC": "imm", "LIT": "#0"},
        {"CMD": "__META__", "TYPE": "LOOP_SECTION", "NAME": "stop_check"},
        {"CMD": "TEST", "OP": "r1-#5", "UF": "1"},
        {"CMD": "JUMP", "LABEL": "loop1_end", "IF": "eq"},
        {"CMD": "__META__", "TYPE": "LOOP_SECTION", "NAME": "body"},
        {"CMD": "NOP"},
        {"CMD": "__META__", "TYPE": "LOOP_SECTION", "NAME": "update"},
        {"CMD": "REG_WR", "DST": "r1", "SRC": "op", "OP": "r1+#1"},
        {"CMD": "__META__", "TYPE": "LOOP_SECTION", "NAME": "jump_back"},
        {"CMD": "JUMP", "LABEL": "loop1_start"},
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
    assert loop.trip_count == 5
    assert isinstance(loop.stop_check.insts[0], TestInst)
    assert isinstance(loop.stop_check.insts[1], JumpInst)
    assert loop.stop_check.insts[1].if_cond == "eq"
    
    # Unbuild (emits instructions)
    opt_insts, opt_labels = builder.unbuild(root)
    
    # Verify emitted instructions (no markers)
    # 1 (reg_wr) + 2 (test+jump) + 1 (nop) + 1 (reg_wr) + 1 (jump) = 6
    # Wait, LabelInst is also emitted for start/end if they were captured.
    # In factory.py, start_label is captured if it appeared before initial.
    # But in my prog_list above, there is no LabelInst.
    
    expected_cmds = ["REG_WR", "TEST", "JUMP", "NOP", "REG_WR", "JUMP"]
    assert [inst["CMD"] for inst in opt_insts] == expected_cmds
    assert opt_insts[2]["IF"] == "eq"
    assert opt_insts[5].get("IF") is None # Unconditional jump back

def test_pipeline_roundtrip_with_normalization():
    # Test that the pipeline preserves well-formed loops
    # Use a large trip count to avoid automatic unrolling (default max is 16)
    loop = IRLoop(name="r", trip_count=100)
    loop.stop_check.append(TestInst(op="r1-#100"))
    loop.stop_check.append(JumpInst(label="end", if_cond="eq"))
    loop.jump_back.append(JumpInst(label="start"))
    
    root = RootNode(insts=[loop], labels={"start": "0", "end": "100"})
    
    config = PipeLineConfig()
    pipeline = make_default_pipeline(config)
    
    out_ir, _ctx = pipeline(root)
    
    # Check that it's still well-formed
    assert isinstance(out_ir.insts[0], IRLoop)
    out_loop = out_ir.insts[0]
    assert len(out_loop.stop_check.insts) == 2
    assert isinstance(out_loop.stop_check.insts[0], TestInst)
    assert isinstance(out_loop.stop_check.insts[1], JumpInst)
