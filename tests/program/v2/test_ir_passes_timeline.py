from __future__ import annotations

from zcu_tools.program.v2.ir.instructions import (
    NopInst,
    TimeInst,
    WaitInst,
)
from zcu_tools.program.v2.ir.node import BasicBlockNode, BlockNode, RootNode
from zcu_tools.program.v2.ir.passes.timeline import TimedMergeLinear, ZeroDelayDCELinear
from zcu_tools.program.v2.ir.pipeline import _run_linear_passes


def _run_dce(root: RootNode) -> RootNode:
    _run_linear_passes([ZeroDelayDCELinear()], root)
    return root


def _run_merge(root: RootNode) -> RootNode:
    _run_linear_passes([TimedMergeLinear()], root)
    return root


def test_zero_delay_dce_removes_plain_zero_increment():
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[
                TimeInst(c_op="inc_ref", lit="#0"),
                TimeInst(c_op="inc_ref", lit="#4"),
                NopInst(),
            ])
        ]
    )

    out = _run_dce(root)

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], TimeInst)
    assert bb.insts[0].lit == "#4"
    assert isinstance(bb.insts[1], NopInst)


def test_zero_delay_dce_removes_zero_increment_with_extra_args():
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[
                TimeInst(c_op="inc_ref", lit="#0", extra_args={"IR_X": 1}),
                NopInst(),
            ])
        ]
    )

    out = _run_dce(root)

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 1
    assert isinstance(bb.insts[0], NopInst)


def test_timed_instruction_merge_merges_plain_adjacent_increments():
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[
                TimeInst(c_op="inc_ref", lit="#2"),
                TimeInst(c_op="inc_ref", lit="#3"),
                NopInst(),
            ])
        ]
    )

    out = _run_merge(root)

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], TimeInst)
    assert bb.insts[0].lit == "#5"
    assert isinstance(bb.insts[1], NopInst)


def test_timed_instruction_merge_merges_adjacent_increments_with_extra_args():
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[
                TimeInst(c_op="inc_ref", lit="#2", extra_args={"IR_X": 1}),
                TimeInst(c_op="inc_ref", lit="#3", extra_args={"IR_Y": 2}),
            ])
        ]
    )

    out = _run_merge(root)

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 1
    assert isinstance(bb.insts[0], TimeInst)
    assert bb.insts[0].lit == "#5"


def test_timed_instruction_merge_keeps_zero_increment_as_boundary():
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[
                TimeInst(c_op="inc_ref", lit="#2"),
                TimeInst(c_op="inc_ref", lit="#0"),
                TimeInst(c_op="inc_ref", lit="#3"),
            ])
        ]
    )

    out = _run_merge(root)

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 3
    assert [inst.lit for inst in bb.insts] == ["#2", "#0", "#3"]


def test_timed_instruction_merge_does_not_cross_block_boundary():
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit="#2")]),
            BasicBlockNode(insts=[WaitInst(c_op="time")]),
            BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit="#3")]),
        ]
    )

    out = _run_merge(root)

    assert len(out.insts) == 3
    assert isinstance(out.insts[0].insts[0], TimeInst)
    assert out.insts[0].insts[0].lit == "#2"
    assert isinstance(out.insts[2].insts[0], TimeInst)
    assert out.insts[2].insts[0].lit == "#3"


# ---------------------------------------------------------------------------
# BasicBlockNode path and fix_addr_size behaviour
# ---------------------------------------------------------------------------

def test_zero_delay_dce_removes_from_basic_block():
    root = RootNode(
        insts=[
            BlockNode(insts=[
                BasicBlockNode(insts=[
                    TimeInst(c_op="inc_ref", lit="#0"),
                    NopInst(),
                    TimeInst(c_op="inc_ref", lit="#0"),
                ]),
            ])
        ]
    )

    out = _run_dce(root)

    bb = out.insts[0].insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 1
    assert isinstance(bb.insts[0], NopInst)


def test_zero_delay_dce_nop_pads_fixed_basic_block():
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit="#0"), NopInst()], fix_addr_size=True),
        ]
    )

    out = _run_dce(root)

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2  # stride preserved
    assert isinstance(bb.insts[0], NopInst)  # TIME#0 replaced with NOP
    assert isinstance(bb.insts[1], NopInst)


def test_timed_merge_merges_in_basic_block():
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[
                TimeInst(c_op="inc_ref", lit="#2"),
                TimeInst(c_op="inc_ref", lit="#3"),
                NopInst(),
            ]),
        ]
    )

    out = _run_merge(root)

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], TimeInst)
    assert bb.insts[0].lit == "#5"
    assert isinstance(bb.insts[1], NopInst)


def test_timed_merge_nop_pads_fixed_basic_block():
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[
                TimeInst(c_op="inc_ref", lit="#2"),
                TimeInst(c_op="inc_ref", lit="#3"),
            ], fix_addr_size=True),
        ]
    )

    out = _run_merge(root)

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2  # stride preserved
    assert isinstance(bb.insts[0], TimeInst)
    assert bb.insts[0].lit == "#5"  # merged value
    assert isinstance(bb.insts[1], NopInst)  # padding
