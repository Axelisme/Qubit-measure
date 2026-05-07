from __future__ import annotations

from zcu_tools.program.v2.ir.instructions import NopInst, RegWriteInst, TimeInst
from zcu_tools.program.v2.ir.node import BasicBlockNode, RootNode
from zcu_tools.program.v2.ir.passes.dataflow import IncRegMergeLinear
from zcu_tools.program.v2.ir.pipeline import _run_linear_passes


def test_inc_reg_merge_free_basic():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(dst="r1", src="op", op="r1 + #2"),
                    NopInst(),
                    RegWriteInst(dst="r1", src="op", op="r1 + #3"),
                    RegWriteInst(dst="r2", src="op", op="r2 + #5"),
                    RegWriteInst(dst="r2", src="op", op="r2 - #1"),
                ]
            )
        ]
    )
    
    _run_linear_passes([IncRegMergeLinear()], root)
    insts = root.insts[0].insts
    
    assert len(insts) == 3
    assert insts[0] == NopInst()
    assert isinstance(insts[1], RegWriteInst)
    assert insts[1].dst == "r1"
    assert insts[1].op == "r1 + #5"
    assert isinstance(insts[2], RegWriteInst)
    assert insts[2].dst == "r2"
    assert insts[2].op == "r2 + #4"

def test_inc_reg_merge_free_flush_on_read():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(dst="r1", src="op", op="r1 + #2"),
                    TimeInst(c_op="inc_ref", r1="r1"),  # reads r1
                    RegWriteInst(dst="r1", src="op", op="r1 + #3"),
                ]
            )
        ]
    )
    
    _run_linear_passes([IncRegMergeLinear()], root)
    insts = root.insts[0].insts
    
    assert len(insts) == 3
    assert isinstance(insts[0], RegWriteInst)
    assert insts[0].op == "r1 + #2"
    assert isinstance(insts[1], TimeInst)
    assert isinstance(insts[2], RegWriteInst)
    assert insts[2].op == "r1 + #3"

def test_inc_reg_merge_fixed_basic():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(dst="r1", src="op", op="r1 + #2"),
                    RegWriteInst(dst="r1", src="op", op="r1 + #3"),
                    NopInst(),
                    RegWriteInst(dst="r2", src="op", op="r2 + #5"),
                    RegWriteInst(dst="r2", src="op", op="r2 - #5"),
                ],
                fix_addr_size=True
            )
        ]
    )
    
    _run_linear_passes([IncRegMergeLinear()], root)
    insts = root.insts[0].insts
    
    assert len(insts) == 5
    assert isinstance(insts[0], RegWriteInst)
    assert insts[0].op == "r1 + #5"
    assert insts[1] == NopInst()
    assert insts[2] == NopInst()
    assert insts[3] == NopInst()
    assert insts[4] == NopInst()

def test_inc_reg_merge_fixed_non_adjacent():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(dst="r1", src="op", op="r1 + #2"),
                    NopInst(),
                    RegWriteInst(dst="r1", src="op", op="r1 + #3"),
                ],
                fix_addr_size=True
            )
        ]
    )
    
    _run_linear_passes([IncRegMergeLinear()], root)
    insts = root.insts[0].insts
    
    assert len(insts) == 3
    assert insts[0].op == "r1 + #2"
    assert insts[1] == NopInst()
    assert insts[2].op == "r1 + #3"
