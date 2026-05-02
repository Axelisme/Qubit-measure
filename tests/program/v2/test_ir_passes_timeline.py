from __future__ import annotations

from zcu_tools.program.v2.ir.instructions import GenericInst, TimeInst
from zcu_tools.program.v2.ir.node import InstNode, RootNode
from zcu_tools.program.v2.ir.pipeline import PipeLineContext
from zcu_tools.program.v2.ir.passes.timeline import (
    TimedInstructionMergePass,
    ZeroDelayDCEPass,
)


def test_zero_delay_dce_removes_plain_zero_increment():
    root = RootNode(
        insts=[
            InstNode(TimeInst(c_op="inc_ref", lit="#0")),
            InstNode(TimeInst(c_op="inc_ref", lit="#4")),
            InstNode(GenericInst(cmd="NOP")),
        ]
    )

    out = ZeroDelayDCEPass().process(root, PipeLineContext())

    assert len(out.insts) == 2
    assert isinstance(out.insts[0], InstNode)
    assert isinstance(out.insts[0].inst, TimeInst)
    assert out.insts[0].inst.lit == "#4"
    assert isinstance(out.insts[1], InstNode)
    assert isinstance(out.insts[1].inst, GenericInst)


def test_zero_delay_dce_removes_annotated_zero_increment():
    root = RootNode(
        insts=[
            InstNode(TimeInst(c_op="inc_ref", lit="#0", annotations={"IR_X": 1})),
            InstNode(GenericInst(cmd="NOP")),
        ]
    )

    out = ZeroDelayDCEPass().process(root, PipeLineContext())

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], InstNode)
    assert isinstance(out.insts[0].inst, GenericInst)


def test_timed_instruction_merge_merges_plain_adjacent_increments():
    root = RootNode(
        insts=[
            InstNode(TimeInst(c_op="inc_ref", lit="#2")),
            InstNode(TimeInst(c_op="inc_ref", lit="#3")),
            InstNode(GenericInst(cmd="NOP")),
        ]
    )

    out = TimedInstructionMergePass().process(root, PipeLineContext())

    assert len(out.insts) == 2
    assert isinstance(out.insts[0], InstNode)
    assert isinstance(out.insts[0].inst, TimeInst)
    assert out.insts[0].inst.lit == "#5"
    assert isinstance(out.insts[1], InstNode)
    assert isinstance(out.insts[1].inst, GenericInst)


def test_timed_instruction_merge_merges_annotated_adjacent_increments():
    root = RootNode(
        insts=[
            InstNode(TimeInst(c_op="inc_ref", lit="#2", annotations={"IR_X": 1})),
            InstNode(TimeInst(c_op="inc_ref", lit="#3", annotations={"IR_Y": 2})),
        ]
    )

    out = TimedInstructionMergePass().process(root, PipeLineContext())

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], InstNode)
    assert isinstance(out.insts[0].inst, TimeInst)
    assert out.insts[0].inst.lit == "#5"


def test_timed_instruction_merge_keeps_zero_increment_as_boundary():
    root = RootNode(
        insts=[
            InstNode(TimeInst(c_op="inc_ref", lit="#2")),
            InstNode(TimeInst(c_op="inc_ref", lit="#0")),
            InstNode(TimeInst(c_op="inc_ref", lit="#3")),
        ]
    )

    out = TimedInstructionMergePass().process(root, PipeLineContext())

    assert len(out.insts) == 3
    assert [getattr(item.inst, "lit", None) for item in out.insts] == ["#2", "#0", "#3"]
