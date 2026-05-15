"""Unit tests for build_jump_table_blocks."""

from __future__ import annotations

from zcu_tools.program.v2.ir.instructions import (
    Instruction,
    JumpInst,
    MetaInst,
    RegWriteInst,
    TimeInst,
)
from zcu_tools.program.v2.ir.labels import Label
from zcu_tools.program.v2.ir.node import BasicBlockNode, BlockNode
from zcu_tools.program.v2.ir.operands import (
    AluExpr,
    Immediate,
    ImmValue,
    Register,
    SrcKeyword,
)
from zcu_tools.program.v2.ir.passes.loop.dispatch_island import (
    build_jump_table_blocks,
)

# ---------------------------------------------------------------------------
# build_jump_table_blocks helpers
# ---------------------------------------------------------------------------


def _make_jt_blocks(
    k: int, body_words: int, pmem_size: int | None = None
) -> list[BasicBlockNode]:
    Label.reset()
    entry_labels = [Label.make_new(f"e_{i}") for i in range(k)]
    exit_label = Label.make_new("jt_exit")
    bodies = [
        BlockNode(
            insts=[
                BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit=Immediate(1))])
                for _ in range(body_words)
            ]
        )
        for _ in range(k)
    ]
    return build_jump_table_blocks(
        n_reg="r_n",
        counter_reg="r_i",
        k=k,
        entry_labels=entry_labels,
        exit_label=exit_label,
        bodies=bodies,
        pmem_size=pmem_size,
    )


def _flatten_blocks(blocks: list[BasicBlockNode]) -> list[Instruction]:
    """Flatten BasicBlockNodes into a single instruction list."""
    result: list[Instruction] = []
    for bb in blocks:
        result.extend(bb.labels)
        result.extend(bb.insts)
        if bb.branch is not None:
            result.append(bb.branch)
    return result


# ---------------------------------------------------------------------------
# build_jump_table_blocks structural tests
# ---------------------------------------------------------------------------


def test_build_jump_table_blocks_structure_k2():
    blocks = _make_jt_blocks(k=2, body_words=1)
    flat = _flatten_blocks(blocks)

    real = [inst for inst in flat if not isinstance(inst, MetaInst)]

    # New dispatch-table shape:
    # prologue:
    #   JUMP exit -if(Z) -op(r_n - #0)
    #   REG_WR r_i op (r_n AND #1)
    #   JUMP e_0 -if(Z) -op(r_i - #0)
    #   REG_WR r_i op (r_i - #2)
    #   REG_WR r_i op (ABS r_i)
    #   REG_WR s15 label dispatch_0
    #   REG_WR s15 op (s15 + r_i)
    #   REG_WR r_i imm #0
    #   JUMP s15
    # table:
    #   LABEL dispatch_0; JUMP e_0
    #   LABEL dispatch_1; JUMP e_1
    # bodies:
    #   LABEL e_0; TimeInst
    #   LABEL e_1; TimeInst
    # back_edge:
    #   JUMP exit -if(NS) -op(r_i - r_n)
    #   JUMP e_0
    # LABEL exit
    types = [type(inst).__name__ for inst in real]
    assert types == [
        "JumpInst",  # n==0 guard
        "RegWriteInst",
        "JumpInst",  # i=n%k, jump if r==0
        "RegWriteInst",
        "RegWriteInst",  # i=i-#k, i=ABS i
        "RegWriteInst",  # s15=label dispatch_0
        "RegWriteInst",
        "RegWriteInst",  # s15+=i, i:=0
        "JumpInst",  # JUMP s15
        "LabelInst",
        "JumpInst",  # dispatch_0 stub
        "LabelInst",
        "JumpInst",  # dispatch_1 stub
        "LabelInst",
        "TimeInst",  # entry_0: body
        "LabelInst",
        "TimeInst",  # entry_1: body
        "JumpInst",
        "JumpInst",  # back_edge: exit check, → entry_0
        "LabelInst",  # exit
    ]

    # Spot-check key operands.
    prologue_jump = real[0]
    assert isinstance(prologue_jump, JumpInst)
    assert str(prologue_jump.op) == "r_n - #0"
    assert prologue_jump.if_cond == "Z"

    real[1]
    remainder_inst = real[1]
    assert isinstance(remainder_inst, RegWriteInst)
    assert remainder_inst.dst.name == "r_i" and remainder_inst.src == "op"

    label_writes = [
        inst for inst in real if isinstance(inst, RegWriteInst) and inst.src == "label"
    ]
    assert len(label_writes) == 1
    assert label_writes[0].dst.name == "s15"

    dispatch_jumps = [
        inst
        for inst in real
        if isinstance(inst, JumpInst)
        and inst.addr is not None
        and str(inst.addr) == "s15"
    ]
    assert len(dispatch_jumps) == 1


def test_build_jump_table_blocks_body_words_5_uses_shift_add_seq():
    # body width no longer affects dispatch arithmetic.
    blocks = _make_jt_blocks(k=2, body_words=5)
    flat = _flatten_blocks(blocks)

    s15_add_writes = [
        inst
        for inst in flat
        if isinstance(inst, RegWriteInst)
        and inst.dst.name == "s15"
        and str(inst.op) == "s15 + r_i"
    ]
    assert len(s15_add_writes) == 1


def test_build_jump_table_blocks_large_pmem_uses_s15_jumps():
    blocks = _make_jt_blocks(k=2, body_words=1, pmem_size=4096)
    flat = _flatten_blocks(blocks)

    # All conditional jumps should use ADDR=s15 (big-jump mode).
    cond_jumps = [
        inst for inst in flat if isinstance(inst, JumpInst) and inst.if_cond is not None
    ]
    assert cond_jumps
    assert all(inst.addr is not None and str(inst.addr) == "s15" for inst in cond_jumps)
    assert all(j.label is None for j in cond_jumps)

    # Multiple label materializations.
    label_writes = [
        inst for inst in flat if isinstance(inst, RegWriteInst) and inst.src == "label"
    ]
    assert len(label_writes) >= 5


def test_build_jump_table_blocks_invalid_k_raises():
    Label.reset()
    try:
        build_jump_table_blocks(
            n_reg="r_n",
            counter_reg="r_i",
            k=1,  # < 2
            entry_labels=[],
            exit_label=Label("x"),
            bodies=[],
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_build_jump_table_blocks_entry_blocks_have_disable_opt():
    blocks = _make_jt_blocks(k=4, body_words=2)
    fixed_blocks = [b for b in blocks if b.disable_opt]
    assert len(fixed_blocks) == 4
    assert all(b.labels for b in fixed_blocks)
    assert all(b.branch is not None for b in fixed_blocks)
    # Body copies and back-edge blocks must remain free-form.
    free_blocks = [b for b in blocks if not b.disable_opt]
    assert any(
        any(lbl.name.name.startswith("e_") for lbl in b.labels) for b in free_blocks
    )
