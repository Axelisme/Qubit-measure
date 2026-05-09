from __future__ import annotations

from zcu_tools.program.v2.ir.factory import IRParser
from zcu_tools.program.v2.ir.instructions import (
    NopInst,
    PortWriteInst,
    RegWriteInst,
    TimeInst,
    WaitInst,
)
from zcu_tools.program.v2.ir.node import BasicBlockNode, BlockNode, RootNode
from zcu_tools.program.v2.ir.operands import Literal, Register
from zcu_tools.program.v2.ir.passes.timeline import TimedMergePass, ZeroDelayDCEPass
from zcu_tools.program.v2.ir.pipeline import PipeLineConfig, PipeLineContext
from zcu_tools.program.v2.ir.passes import walk_basic_blocks


def _run_dce(root: RootNode) -> RootNode:
    parser = IRParser()
    chunks = parser.unparse(root)
    chunks, _ = ZeroDelayDCEPass().process(
        chunks, PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    )
    return parser.parse(chunks)


def _run_merge(root: RootNode) -> RootNode:
    parser = IRParser()
    chunks = parser.unparse(root)
    chunks, _ = TimedMergePass().process(
        chunks, PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    )
    return parser.parse(chunks)


def test_zero_delay_dce_removes_plain_zero_increment():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Literal("#0")),
                    TimeInst(c_op="inc_ref", lit=Literal("#4")),
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
    assert bb.insts[0].lit == Literal("#4")
    assert isinstance(bb.insts[1], NopInst)


def test_timed_instruction_merge_merges_plain_adjacent_increments():
    # New aggressive behavior: TIME sinks past NopInst, merges at end.
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Literal("#2")),
                    TimeInst(c_op="inc_ref", lit=Literal("#3")),
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
    assert bb.insts[1].lit == Literal("#5")


def test_timed_instruction_merge_sinks_past_zero_increment():
    # TIME #0 is not a lit-time and not an anchor; both #2 and #3 accumulate
    # across it, producing a single merged TIME at the end.
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Literal("#2")),
                    TimeInst(c_op="inc_ref", lit=Literal("#0")),
                    TimeInst(c_op="inc_ref", lit=Literal("#3")),
                ]
            )
        ]
    )

    out = _run_merge(root)

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], TimeInst)
    assert bb.insts[0].lit == Literal("#0")
    assert isinstance(bb.insts[1], TimeInst)
    assert bb.insts[1].lit == Literal("#5")


def test_timed_instruction_merge_does_not_cross_block_boundary():
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit=Literal("#2"))]),
            BasicBlockNode(insts=[WaitInst(c_op="time")]),
            BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit=Literal("#3"))]),
        ]
    )

    out = _run_merge(root)

    assert len(out.insts) == 3
    assert isinstance(out.insts[0].insts[0], TimeInst)  # type: ignore
    assert out.insts[0].insts[0].lit == Literal("#2")  # type: ignore
    assert isinstance(out.insts[2].insts[0], TimeInst)  # type: ignore
    assert out.insts[2].insts[0].lit == Literal("#3")  # type: ignore


# ---------------------------------------------------------------------------
# BasicBlockNode path and fix_addr_size behaviour
# ---------------------------------------------------------------------------


def test_zero_delay_dce_removes_from_basic_block():
    root = RootNode(
        insts=[
            BlockNode(
                insts=[
                    BasicBlockNode(
                        insts=[
                            TimeInst(c_op="inc_ref", lit=Literal("#0")),
                            NopInst(),
                            TimeInst(c_op="inc_ref", lit=Literal("#0")),
                        ]
                    ),
                ]
            )
        ]
    )

    out = _run_dce(root)

    bb = list(walk_basic_blocks(out))[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 1
    assert isinstance(bb.insts[0], NopInst)


def test_zero_delay_dce_skips_fixed_basic_block():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[TimeInst(c_op="inc_ref", lit=Literal("#0")), NopInst()],
                fix_addr_size=True,
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
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Literal("#2")),
                    TimeInst(c_op="inc_ref", lit=Literal("#3")),
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
    assert bb.insts[1].lit == Literal("#5")


def test_timed_merge_skips_fixed_basic_block():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Literal("#2")),
                    TimeInst(c_op="inc_ref", lit=Literal("#3")),
                ],
                fix_addr_size=True,
            ),
        ]
    )

    out = _run_merge(root)

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], TimeInst)
    assert bb.insts[0].lit == Literal("#2")
    assert isinstance(bb.insts[1], TimeInst)
    assert bb.insts[1].lit == Literal("#3")


# ---------------------------------------------------------------------------
# Aggressive sinking and absorption (new _merge_free behaviour)
# ---------------------------------------------------------------------------


def test_lit_time_sinks_past_reg_write():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Literal("#10")),
                    RegWriteInst(dst=Register("r0"), src="imm", lit=Literal("#1")),
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
    assert bb.insts[1].lit == Literal("#10")


def test_lit_time_sinks_past_multiple_non_timed():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Literal("#5")),
                    NopInst(),
                    RegWriteInst(dst=Register("r0"), src="imm", lit=Literal("#1")),
                    NopInst(),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert len(bb.insts) == 4
    assert isinstance(bb.insts[0], NopInst)
    assert isinstance(bb.insts[1], RegWriteInst)
    assert isinstance(bb.insts[2], NopInst)
    assert isinstance(bb.insts[3], TimeInst)
    assert bb.insts[3].lit == Literal("#5")


def test_lit_time_absorbed_into_port_write():
    # TIME sinks to end; PortWrite's @N is adjusted.
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Literal("#43")),
                    PortWriteInst(
                        dst=Literal("2"),
                        src="wmem",
                        addr=Literal("&12"),
                        time=Literal("@0"),
                    ),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], PortWriteInst)
    assert bb.insts[0].time == Literal("@43")
    assert isinstance(bb.insts[1], TimeInst)
    assert bb.insts[1].lit == Literal("#43")


def test_lit_time_absorbed_into_wait_inst():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Literal("#83")),
                    WaitInst(c_op="time", time=Literal("@0")),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], WaitInst)
    assert bb.insts[0].time == Literal("@83")
    assert isinstance(bb.insts[1], TimeInst)
    assert bb.insts[1].lit == Literal("#83")


def test_accumulated_time_absorbed():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Literal("#10")),
                    TimeInst(c_op="inc_ref", lit=Literal("#20")),
                    PortWriteInst(
                        dst=Literal("2"),
                        src="wmem",
                        addr=Literal("&0"),
                        time=Literal("@0"),
                    ),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], PortWriteInst)
    assert bb.insts[0].time == Literal("@30")
    assert isinstance(bb.insts[1], TimeInst)
    assert bb.insts[1].lit == Literal("#30")


def test_time_and_non_timed_then_port_write():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Literal("#10")),
                    RegWriteInst(dst=Register("r0"), src="imm", lit=Literal("#1")),
                    TimeInst(c_op="inc_ref", lit=Literal("#20")),
                    PortWriteInst(
                        dst=Literal("2"),
                        src="wmem",
                        addr=Literal("&0"),
                        time=Literal("@0"),
                    ),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert len(bb.insts) == 3
    assert isinstance(bb.insts[0], RegWriteInst)
    assert isinstance(bb.insts[1], PortWriteInst)
    assert bb.insts[1].time == Literal("@30")
    assert isinstance(bb.insts[2], TimeInst)
    assert bb.insts[2].lit == Literal("#30")


def test_all_timed_adjusted_with_same_delta():
    # pending_lit is NOT reset after absorption; all timed insts in the same
    # baseline segment receive the same delta adjustment.
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Literal("#10")),
                    PortWriteInst(
                        dst=Literal("2"),
                        src="wmem",
                        addr=Literal("&0"),
                        time=Literal("@0"),
                    ),
                    PortWriteInst(
                        dst=Literal("2"),
                        src="wmem",
                        addr=Literal("&1"),
                        time=Literal("@5"),
                    ),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert len(bb.insts) == 3
    assert isinstance(bb.insts[0], PortWriteInst)
    assert bb.insts[0].time == Literal("@10")  # 0 + 10
    assert isinstance(bb.insts[1], PortWriteInst)
    assert bb.insts[1].time == Literal("@15")  # 5 + 10 — same delta applied
    assert isinstance(bb.insts[2], TimeInst)
    assert bb.insts[2].lit == Literal("#10")  # single TIME at end


def test_port_write_no_time_not_anchor():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Literal("#10")),
                    PortWriteInst(
                        dst=Literal("2"), src="wmem", addr=Literal("&0"), time=None
                    ),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], PortWriteInst)
    assert bb.insts[0].time is None
    assert isinstance(bb.insts[1], TimeInst)
    assert bb.insts[1].lit == Literal("#10")


def test_port_write_reg_time_not_adjusted():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Literal("#10")),
                    PortWriteInst(
                        dst=Literal("2"),
                        src="wmem",
                        addr=Literal("&0"),
                        time=Register("s14"),
                    ),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], PortWriteInst)
    assert bb.insts[0].time == Register("s14")  # register ref: not adjusted
    assert isinstance(bb.insts[1], TimeInst)
    assert bb.insts[1].lit == Literal("#10")


def test_reg_time_flushes_pending_lit():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Literal("#10")),
                    RegWriteInst(dst=Register("r0"), src="imm", lit=Literal("#1")),
                    TimeInst(c_op="inc_ref", r1=Register("r1")),  # reg-TIME barrier
                    PortWriteInst(
                        dst=Literal("2"),
                        src="wmem",
                        addr=Literal("&0"),
                        time=Literal("@5"),
                    ),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert len(bb.insts) == 4
    assert isinstance(bb.insts[0], RegWriteInst)
    assert isinstance(bb.insts[1], TimeInst) and bb.insts[1].lit == Literal("#10")
    assert isinstance(bb.insts[2], TimeInst) and bb.insts[2].r1 == Register("r1")
    assert isinstance(bb.insts[3], PortWriteInst) and bb.insts[3].time == Literal("@5")


def test_pending_lit_flushed_at_end_of_block():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TimeInst(c_op="inc_ref", lit=Literal("#15")),
                    RegWriteInst(dst=Register("r0"), src="imm", lit=Literal("#1")),
                ]
            )
        ]
    )
    out = _run_merge(root)
    bb = out.insts[0]
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], RegWriteInst)
    assert isinstance(bb.insts[1], TimeInst)
    assert bb.insts[1].lit == Literal("#15")
