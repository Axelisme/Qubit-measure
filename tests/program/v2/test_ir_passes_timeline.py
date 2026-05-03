from __future__ import annotations

from typing import cast

from zcu_tools.program.v2.ir.instructions import NopInst, TimeInst, WaitInst
from zcu_tools.program.v2.ir.node import InstNode, RootNode
from zcu_tools.program.v2.ir.passes.timeline import (
    TimedInstructionMergePass,
    ZeroDelayDCEPass,
)
from zcu_tools.program.v2.ir.pipeline import PipeLineConfig, PipeLineContext


def test_zero_delay_dce_removes_plain_zero_increment():
    root = RootNode(
        insts=[
            InstNode(TimeInst(c_op="inc_ref", lit="#0")),
            InstNode(TimeInst(c_op="inc_ref", lit="#4")),
            InstNode(NopInst()),
        ]
    )

    out = ZeroDelayDCEPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert len(out.insts) == 2
    assert isinstance(out.insts[0], InstNode)
    assert isinstance(cast(InstNode, out.insts[0]).inst, TimeInst)
    assert getattr(cast(InstNode, out.insts[0]).inst, "lit") == "#4"
    assert isinstance(out.insts[1], InstNode)
    assert isinstance(cast(InstNode, out.insts[1]).inst, NopInst)


def test_zero_delay_dce_removes_zero_increment_with_extra_args():
    root = RootNode(
        insts=[
            InstNode(TimeInst(c_op="inc_ref", lit="#0", extra_args={"IR_X": 1})),
            InstNode(NopInst()),
        ]
    )

    out = ZeroDelayDCEPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], InstNode)
    assert isinstance(cast(InstNode, out.insts[0]).inst, NopInst)


def test_timed_instruction_merge_merges_plain_adjacent_increments():
    root = RootNode(
        insts=[
            InstNode(TimeInst(c_op="inc_ref", lit="#2")),
            InstNode(TimeInst(c_op="inc_ref", lit="#3")),
            InstNode(NopInst()),
        ]
    )

    out = TimedInstructionMergePass().process(
        root, PipeLineContext(config=PipeLineConfig())
    )

    assert len(out.insts) == 2
    assert isinstance(out.insts[0], InstNode)
    assert isinstance(cast(InstNode, out.insts[0]).inst, TimeInst)
    assert getattr(cast(InstNode, out.insts[0]).inst, "lit") == "#5"
    assert isinstance(out.insts[1], InstNode)
    assert isinstance(cast(InstNode, out.insts[1]).inst, NopInst)


def test_timed_instruction_merge_merges_adjacent_increments_with_extra_args():
    root = RootNode(
        insts=[
            InstNode(TimeInst(c_op="inc_ref", lit="#2", extra_args={"IR_X": 1})),
            InstNode(TimeInst(c_op="inc_ref", lit="#3", extra_args={"IR_Y": 2})),
        ]
    )

    out = TimedInstructionMergePass().process(
        root, PipeLineContext(config=PipeLineConfig())
    )

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], InstNode)
    assert isinstance(cast(InstNode, out.insts[0]).inst, TimeInst)
    assert getattr(cast(InstNode, out.insts[0]).inst, "lit") == "#5"


def test_timed_instruction_merge_keeps_zero_increment_as_boundary():
    root = RootNode(
        insts=[
            InstNode(TimeInst(c_op="inc_ref", lit="#2")),
            InstNode(TimeInst(c_op="inc_ref", lit="#0")),
            InstNode(TimeInst(c_op="inc_ref", lit="#3")),
        ]
    )

    out = TimedInstructionMergePass().process(
        root, PipeLineContext(config=PipeLineConfig())
    )

    assert len(out.insts) == 3
    assert [getattr(cast(InstNode, item).inst, "lit", None) for item in out.insts] == ["#2", "#0", "#3"]


def test_timed_instruction_merge_does_not_cross_wait():
    root = RootNode(
        insts=[
            InstNode(TimeInst(c_op="inc_ref", lit="#2")),
            InstNode(WaitInst(c_op="time")),
            InstNode(TimeInst(c_op="inc_ref", lit="#3")),
        ]
    )

    out = TimedInstructionMergePass().process(
        root, PipeLineContext(config=PipeLineConfig())
    )

    assert len(out.insts) == 3
    assert [getattr(cast(InstNode, item).inst, "lit", None) for item in out.insts] == ["#2", None, "#3"]
