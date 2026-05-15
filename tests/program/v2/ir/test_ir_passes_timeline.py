from __future__ import annotations

from typing import Iterator

from zcu_tools.program.v2.ir.factory import IRParser
from zcu_tools.program.v2.ir.instructions import (
    NopInst,
    PortWriteInst,
    RegWriteInst,
    TimeInst,
    WaitInst,
)
from zcu_tools.program.v2.ir.node import (
    BasicBlockNode,
    BlockNode,
    IRBranch,
    IRLoop,
    IRNode,
)
from zcu_tools.program.v2.ir.operands import (
    Immediate,
    ImmValue,
    MemAddr,
    Register,
    SrcKeyword,
    TimeOffset,
)
from zcu_tools.program.v2.ir.passes.timeline import TimedMergePass, ZeroDelayDCEPass
from zcu_tools.program.v2.ir.pipeline import PipeLineConfig, PipeLineContext


def _walk_basic_blocks(node: IRNode) -> Iterator[BasicBlockNode]:
    if isinstance(node, BasicBlockNode):
        yield node
    elif isinstance(node, BlockNode):
        for child in node.insts:
            yield from _walk_basic_blocks(child)
    elif isinstance(node, IRLoop):
        yield from _walk_basic_blocks(node.body)
    elif isinstance(node, IRBranch):
        for case in node.cases:
            yield from _walk_basic_blocks(case)


def _run_dce(root: BlockNode) -> BlockNode:
    parser = IRParser()
    chunks = parser.unparse(root)
    chunks, _ = ZeroDelayDCEPass().process(
        chunks, PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    )
    return parser.parse(chunks)


def _run_merge(root: BlockNode) -> BlockNode:
    parser = IRParser()
    chunks = parser.unparse(root)
    chunks, _ = TimedMergePass().process(
        chunks, PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    )
    return parser.parse(chunks)


def test_zero_delay_dce_removes_plain_zero_increment():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Immediate(0)),
                    TimeInst(c_op="inc_ref", lit=Immediate(4)),
                    NopInst(),
                ]
            )
        ]
    )

    out = _run_dce(root)

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], TimeInst)
    assert str(bb.insts[0].lit) == "#4"
    assert isinstance(bb.insts[1], NopInst)


def test_timed_instruction_merge_merges_plain_adjacent_increments():
    # New aggressive behavior: TIME sinks past NopInst, merges at end.
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Immediate(2)),
                    TimeInst(c_op="inc_ref", lit=Immediate(3)),
                    NopInst(),
                ]
            )
        ]
    )

    out = _run_merge(root)

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], NopInst)
    assert isinstance(bb.insts[1], TimeInst)
    assert str(bb.insts[1].lit) == "#5"


def test_timed_instruction_merge_sinks_past_zero_increment():
    # TIME #0 is not a lit-time and not an anchor; both #2 and #3 accumulate
    # across it, producing a single merged TIME at the end.
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Immediate(2)),
                    TimeInst(c_op="inc_ref", lit=Immediate(0)),
                    TimeInst(c_op="inc_ref", lit=Immediate(3)),
                ]
            )
        ]
    )

    out = _run_merge(root)

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], TimeInst)
    assert str(bb.insts[0].lit) == "#0"
    assert isinstance(bb.insts[1], TimeInst)
    assert str(bb.insts[1].lit) == "#5"


def test_timed_instruction_merge_does_not_cross_block_boundary():
    root = BlockNode(
        insts=[
            BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit=Immediate(2))]),
            BasicBlockNode(insts=[WaitInst(c_op="time")]),
            BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit=Immediate(3))]),
        ]
    )

    out = _run_merge(root)

    assert len(out.insts) == 3
    assert isinstance(out.insts[0].insts[0], TimeInst)  # type: ignore
    assert str(out.insts[0].insts[0].lit) == "#2"  # type: ignore
    assert isinstance(out.insts[2].insts[0], TimeInst)  # type: ignore
    assert str(out.insts[2].insts[0].lit) == "#3"  # type: ignore


# ---------------------------------------------------------------------------
# BasicBlockNode path and disable_opt behaviour
# ---------------------------------------------------------------------------


def test_zero_delay_dce_removes_from_basic_block():
    root = BlockNode(
        insts=[
            BlockNode(
                insts=[
                    BasicBlockNode(
                        insts=[
                            TimeInst(c_op="inc_ref", lit=Immediate(0)),
                            NopInst(),
                            TimeInst(c_op="inc_ref", lit=Immediate(0)),
                        ]
                    ),
                ]
            )
        ]
    )

    out = _run_dce(root)

    bb = list(_walk_basic_blocks(out))[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 1
    assert isinstance(bb.insts[0], NopInst)


def test_zero_delay_dce_skips_fixed_basic_block():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[TimeInst(c_op="inc_ref", lit=Immediate(0)), NopInst()],
                disable_opt=True,
            ),
        ]
    )

    out = _run_dce(root)

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], TimeInst)
    assert isinstance(bb.insts[1], NopInst)


def test_timed_merge_merges_in_basic_block():
    # TIME sinks past NopInst; merged TIME appears at end.
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Immediate(2)),
                    TimeInst(c_op="inc_ref", lit=Immediate(3)),
                    NopInst(),
                ]
            ),
        ]
    )

    out = _run_merge(root)

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], NopInst)
    assert isinstance(bb.insts[1], TimeInst)
    assert str(bb.insts[1].lit) == "#5"


def test_timed_merge_skips_fixed_basic_block():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Immediate(2)),
                    TimeInst(c_op="inc_ref", lit=Immediate(3)),
                ],
                disable_opt=True,
            ),
        ]
    )

    out = _run_merge(root)

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], TimeInst)
    assert str(bb.insts[0].lit) == "#2"
    assert isinstance(bb.insts[1], TimeInst)
    assert str(bb.insts[1].lit) == "#3"


# ---------------------------------------------------------------------------
# Aggressive sinking and absorption (new _merge_free behaviour)
# ---------------------------------------------------------------------------


def test_lit_time_sinks_past_reg_write():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Immediate(10)),
                    RegWriteInst(
                        dst=Register("r0"), src=SrcKeyword.IMM, lit=Immediate(1)
                    ),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], RegWriteInst)
    assert isinstance(bb.insts[1], TimeInst)
    assert str(bb.insts[1].lit) == "#10"


def test_lit_time_sinks_past_multiple_non_timed():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Immediate(5)),
                    NopInst(),
                    RegWriteInst(
                        dst=Register("r0"), src=SrcKeyword.IMM, lit=Immediate(1)
                    ),
                    NopInst(),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 4
    assert isinstance(bb.insts[0], NopInst)
    assert isinstance(bb.insts[1], RegWriteInst)
    assert isinstance(bb.insts[2], NopInst)
    assert isinstance(bb.insts[3], TimeInst)
    assert str(bb.insts[3].lit) == "#5"


def test_lit_time_absorbed_into_port_write():
    # TIME sinks to end; PortWrite's @N is adjusted.
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Immediate(43)),
                    PortWriteInst(
                        dst=ImmValue(2),
                        src=SrcKeyword.WMEM,
                        addr=MemAddr(12),
                        time=TimeOffset(0),
                    ),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], PortWriteInst)
    assert str(bb.insts[0].time) == "@43"
    assert isinstance(bb.insts[1], TimeInst)
    assert str(bb.insts[1].lit) == "#43"


def test_lit_time_absorbed_into_wait_inst():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Immediate(83)),
                    WaitInst(c_op="time", time=TimeOffset(0)),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], WaitInst)
    assert str(bb.insts[0].time) == "@83"
    assert isinstance(bb.insts[1], TimeInst)
    assert str(bb.insts[1].lit) == "#83"


def test_accumulated_time_absorbed():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Immediate(10)),
                    TimeInst(c_op="inc_ref", lit=Immediate(20)),
                    PortWriteInst(
                        dst=ImmValue(2),
                        src=SrcKeyword.WMEM,
                        addr=MemAddr(0),
                        time=TimeOffset(0),
                    ),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], PortWriteInst)
    assert str(bb.insts[0].time) == "@30"
    assert isinstance(bb.insts[1], TimeInst)
    assert str(bb.insts[1].lit) == "#30"


def test_time_and_non_timed_then_port_write():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Immediate(10)),
                    RegWriteInst(
                        dst=Register("r0"), src=SrcKeyword.IMM, lit=Immediate(1)
                    ),
                    TimeInst(c_op="inc_ref", lit=Immediate(20)),
                    PortWriteInst(
                        dst=ImmValue(2),
                        src=SrcKeyword.WMEM,
                        addr=MemAddr(0),
                        time=TimeOffset(0),
                    ),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 3
    assert isinstance(bb.insts[0], RegWriteInst)
    assert isinstance(bb.insts[1], PortWriteInst)
    assert str(bb.insts[1].time) == "@30"
    assert isinstance(bb.insts[2], TimeInst)
    assert str(bb.insts[2].lit) == "#30"


def test_all_timed_adjusted_with_same_delta():
    # pending_lit is NOT reset after absorption; all timed insts in the same
    # baseline segment receive the same delta adjustment.
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Immediate(10)),
                    PortWriteInst(
                        dst=ImmValue(2),
                        src=SrcKeyword.WMEM,
                        addr=MemAddr(0),
                        time=TimeOffset(0),
                    ),
                    PortWriteInst(
                        dst=ImmValue(2),
                        src=SrcKeyword.WMEM,
                        addr=MemAddr(1),
                        time=TimeOffset(5),
                    ),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 3
    assert isinstance(bb.insts[0], PortWriteInst)
    assert str(bb.insts[0].time) == "@10"  # 0 + 10
    assert isinstance(bb.insts[1], PortWriteInst)
    assert str(bb.insts[1].time) == "@15"  # 5 + 10 — same delta applied
    assert isinstance(bb.insts[2], TimeInst)
    assert str(bb.insts[2].lit) == "#10"  # single TIME at end


def test_port_write_no_time_acts_as_barrier():
    # WPORT_WR with no explicit @T still emits at s14 (out_usr_time), so a
    # pending TIME inc_ref must be flushed *before* it — moving the inc_ref
    # to the other side would shift the actual emission time by the
    # accumulated delta. PortWriteInst.reg_read declares s14, which the pass
    # uses as its barrier signal.
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Immediate(10)),
                    PortWriteInst(
                        dst=ImmValue(2), src=SrcKeyword.WMEM, addr=MemAddr(0), time=None
                    ),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], TimeInst)
    assert str(bb.insts[0].lit) == "#10"
    assert isinstance(bb.insts[1], PortWriteInst)
    assert bb.insts[1].time is None


def test_port_write_reg_time_acts_as_barrier():
    # Register-driven @T (time=Register) cannot be folded by adjusting a
    # literal offset, and the WPORT_WR still reads s14 implicitly, so the
    # pending TIME inc_ref must remain before the WPORT_WR.
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Immediate(10)),
                    PortWriteInst(
                        dst=ImmValue(2),
                        src=SrcKeyword.WMEM,
                        addr=MemAddr(0),
                        time=Register("s14"),
                    ),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], TimeInst)
    assert str(bb.insts[0].lit) == "#10"
    assert isinstance(bb.insts[1], PortWriteInst)
    assert bb.insts[1].time == Register("s14")


def test_reg_time_flushes_pending_lit():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Immediate(10)),
                    RegWriteInst(
                        dst=Register("r0"), src=SrcKeyword.IMM, lit=Immediate(1)
                    ),
                    TimeInst(c_op="inc_ref", r1=Register("r1")),  # reg-TIME barrier
                    PortWriteInst(
                        dst=ImmValue(2),
                        src=SrcKeyword.WMEM,
                        addr=MemAddr(0),
                        time=TimeOffset(5),
                    ),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 4
    assert isinstance(bb.insts[0], RegWriteInst)
    assert isinstance(bb.insts[1], TimeInst) and str(bb.insts[1].lit) == "#10"
    assert isinstance(bb.insts[2], TimeInst) and bb.insts[2].r1 == Register("r1")
    assert isinstance(bb.insts[3], PortWriteInst) and str(bb.insts[3].time) == "@5"


def test_pending_lit_flushed_at_end_of_block():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Immediate(15)),
                    RegWriteInst(
                        dst=Register("r0"), src=SrcKeyword.IMM, lit=Immediate(1)
                    ),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], RegWriteInst)
    assert isinstance(bb.insts[1], TimeInst)
    assert str(bb.insts[1].lit) == "#15"
