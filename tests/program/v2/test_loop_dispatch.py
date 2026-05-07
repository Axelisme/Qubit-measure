"""Unit tests for shift_add_multiply and build_jump_table_blocks."""
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
from zcu_tools.program.v2.ir.passes.loop_dispatch import (
    build_jump_table_blocks,
    shift_add_multiply,
)

# ---------------------------------------------------------------------------
# shift_add_multiply
# ---------------------------------------------------------------------------


def test_shift_add_multiply_power_of_two():
    """body_words=4 → shift src by 2, then add once."""
    seq = shift_add_multiply(src_reg="r1", dst_reg="s15", constant=4, max_words=8)
    assert seq is not None
    ops = [(inst.dst, inst.op) for inst in seq if isinstance(inst, RegWriteInst)]
    assert ops == [
        ("r1", "r1 << #2"),
        ("s15", "s15 + r1"),
    ]


def test_shift_add_multiply_one_is_single_add():
    seq = shift_add_multiply(src_reg="r1", dst_reg="s15", constant=1, max_words=8)
    assert seq is not None
    assert len(seq) == 1
    assert isinstance(seq[0], RegWriteInst)
    assert seq[0].op == "s15 + r1"


def test_shift_add_multiply_five():
    """5 = 0b101 → add at bit 0, shift to bit 2, add."""
    seq = shift_add_multiply(src_reg="r1", dst_reg="s15", constant=5, max_words=8)
    assert seq is not None
    ops = [(inst.dst, inst.op) for inst in seq if isinstance(inst, RegWriteInst)]
    assert ops == [
        ("s15", "s15 + r1"),    # bit 0
        ("r1", "r1 << #2"),     # accumulate shift to bit 2
        ("s15", "s15 + r1"),    # bit 2
    ]


def test_shift_add_multiply_zero_returns_none():
    assert shift_add_multiply(src_reg="r1", dst_reg="s15", constant=0, max_words=8) is None
    assert shift_add_multiply(src_reg="r1", dst_reg="s15", constant=-1, max_words=8) is None


def test_shift_add_multiply_exceeds_word_budget():
    """0xFF = 8 set bits → 8 adds + 7 shifts = 15 instructions; budget 4 → None."""
    assert shift_add_multiply(src_reg="r1", dst_reg="s15", constant=0xFF, max_words=4) is None


def test_shift_add_multiply_shift_amount_too_large():
    """bit 17 requires shift > 15; must return None."""
    assert (
        shift_add_multiply(src_reg="r1", dst_reg="s15", constant=1 << 17, max_words=64)
        is None
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
        BlockNode(insts=[BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit="#1")]) for _ in range(body_words)])
        for _ in range(k)
    ]
    return build_jump_table_blocks(
        name="jt",
        n_reg="r_n",
        counter_reg="r_i",
        k=k,
        body_words=body_words,
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

    # Expected structure (k=2, body_words=1, stride=2=0b10):
    # prologue:
    #   JUMP exit -if(Z) -op(r_n - #0)
    #   REG_WR r_i op (r_n AND #1)
    #   JUMP e_0 -if(Z) -op(r_i - #0)
    #   REG_WR r_i op (r_i - #2)
    #   REG_WR r_i op (ABS r_i)
    #   REG_WR s15 label e_0
    #   REG_WR r_i op (r_i << #1)   } shift-add
    #   REG_WR s15 op (s15 + r_i)   }
    #   REG_WR r_i imm #0
    #   JUMP s15
    # LABEL e_0: TimeInst; REG_WR r_i +1
    # LABEL e_1: TimeInst; REG_WR r_i +1
    # back_edge:
    #   JUMP exit -if(NS) -op(r_i - r_n)
    #   JUMP e_0
    # LABEL exit
    types = [type(inst).__name__ for inst in real]
    assert types == [
        "JumpInst",                                        # n==0 guard
        "RegWriteInst", "JumpInst",                        # i=n%k, jump if r==0
        "RegWriteInst", "RegWriteInst",                    # i=i-#k, i=ABS i
        "RegWriteInst",                                    # s15=label entry_0
        "RegWriteInst", "RegWriteInst",                    # shift-add: i<<=1, s15+=i
        "RegWriteInst", "JumpInst",                        # i:=0, JUMP s15
        "LabelInst", "TimeInst", "RegWriteInst",           # entry_0: body, i++
        "LabelInst", "TimeInst", "RegWriteInst",           # entry_1: body, i++
        "JumpInst", "JumpInst",                            # back_edge: exit check, → entry_0
        "LabelInst",                                       # exit
    ]

    # Spot-check key operands.
    prologue_jump = real[0]
    assert isinstance(prologue_jump, JumpInst)
    assert prologue_jump.op == "r_n - #0"
    assert prologue_jump.if_cond == "Z"

    real[1]
    remainder_inst = real[1]
    assert isinstance(remainder_inst, RegWriteInst)
    assert remainder_inst.dst == "r_i" and remainder_inst.src == "op"

    label_writes = [
        inst for inst in real if isinstance(inst, RegWriteInst) and inst.src == "label"
    ]
    assert len(label_writes) == 1
    assert label_writes[0].dst == "s15"

    dispatch_jumps = [
        inst for inst in real if isinstance(inst, JumpInst) and inst.addr == "s15"
    ]
    assert len(dispatch_jumps) == 1


def test_build_jump_table_blocks_body_words_5_uses_shift_add_seq():
    # stride = body_words + 1 = 5 → helper emits two `s15 += r_i` adds
    blocks = _make_jt_blocks(k=2, body_words=4)
    flat = _flatten_blocks(blocks)

    s15_add_writes = [
        inst
        for inst in flat
        if isinstance(inst, RegWriteInst) and inst.dst == "s15" and inst.op == "s15 + r_i"
    ]
    assert len(s15_add_writes) == 2


def test_build_jump_table_blocks_large_pmem_uses_s15_jumps():
    blocks = _make_jt_blocks(k=2, body_words=1, pmem_size=4096)
    flat = _flatten_blocks(blocks)

    # All conditional jumps should use ADDR=s15 (big-jump mode).
    cond_jumps = [
        inst for inst in flat if isinstance(inst, JumpInst) and inst.if_cond is not None
    ]
    assert cond_jumps
    assert all(j.addr == "s15" for j in cond_jumps)
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
            name="jt",
            n_reg="r_n",
            counter_reg="r_i",
            k=1,  # < 2
            body_words=1,
            entry_labels=[],
            exit_label=Label("x"),
            bodies=[],
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_build_jump_table_blocks_entry_blocks_have_fix_inst_num():
    blocks = _make_jt_blocks(k=4, body_words=2)
    entry_blocks = [b for b in blocks if b.fix_inst_num and b.labels]
    assert len(entry_blocks) == 4
    # Each entry block: body_words + counter_incr = 3 insts.
    for blk in entry_blocks:
        assert len(blk.insts) == 3
