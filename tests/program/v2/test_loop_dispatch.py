"""Phase 8C unit tests: shift_add_multiply helper + IRJumpTableLoop.emit."""
from __future__ import annotations

from zcu_tools.program.v2.ir.instructions import (
    Instruction,
    JumpInst,
    MetaInst,
    RegWriteInst,
    TestInst,
    TimeInst,
)
from zcu_tools.program.v2.ir.labels import Label
from zcu_tools.program.v2.ir.node import BlockNode, InstNode
from zcu_tools.program.v2.ir.passes.loop_dispatch import (
    IRJumpTableLoop,
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
# IRJumpTableLoop.emit
# ---------------------------------------------------------------------------


def _make_jt_loop(k: int, body_words: int) -> IRJumpTableLoop:
    Label.reset()
    entry_labels = [Label.make_new(f"e_{i}") for i in range(k)]
    exit_label = Label.make_new("jt_exit")
    bodies = [
        BlockNode(
            insts=[
                InstNode(TimeInst(c_op="inc_ref", lit="#1")),
                InstNode(RegWriteInst(dst="r_i", src="op", op="r_i + #1")),
            ]
        )
        for _ in range(k)
    ]
    return IRJumpTableLoop(
        n_reg="r_n",
        counter_reg="r_i",
        k=k,
        body_words=body_words,
        entry_labels=entry_labels,
        exit_label=exit_label,
        bodies=bodies,
        name="jt",
    )


def test_irjumptableloop_emit_structure_k2_pow2_body():
    jt = _make_jt_loop(k=2, body_words=1)
    out: list[Instruction] = []
    jt.emit(out)

    # Filter to non-meta for structural inspection.
    real = [inst for inst in out if not isinstance(inst, MetaInst)]

    # Expected sequence:
    #   TEST n - #0
    #   JUMP exit -if(Z)
    #   REG_WR i imm #0
    #   LABEL e_0
    #   TimeInst inc_ref #1   (body 0)
    #   REG_WR i op (i + #1)  (from body 0)
    #   LABEL e_1
    #   TimeInst inc_ref #1   (body 1)
    #   REG_WR i op (i + #1)  (from body 1)
    #   TEST i - n
    #   JUMP exit -if(NS)
    #   REG_WR i op (n - i)
    #   TEST i - #2
    #   JUMP fast_path -if(NS)
    #   REG_WR i op (i - #2)            ; dispatch
    #   REG_WR i op (ABS i)
    #   REG_WR s15 label e_0
    #   REG_WR s15 op (s15 + i)         ; body_words=1 → single add
    #   JUMP s15
    #   LABEL fast_path
    #   REG_WR i op (n - i)
    #   JUMP e_0
    #   LABEL exit
    types = [type(inst).__name__ for inst in real]
    assert types == [
        "TestInst", "JumpInst", "RegWriteInst",
        "LabelInst", "TimeInst", "RegWriteInst",
        "LabelInst", "TimeInst", "RegWriteInst",
        "TestInst", "JumpInst",
        "RegWriteInst", "TestInst", "JumpInst",
        "RegWriteInst", "RegWriteInst", "RegWriteInst",
        "RegWriteInst", "JumpInst",
        "LabelInst", "RegWriteInst", "JumpInst",
        "LabelInst",
    ]

    # Spot-check key operands.
    prologue_test = real[0]
    assert isinstance(prologue_test, TestInst)
    assert prologue_test.op == "r_n - #0"

    init_i = real[2]
    assert isinstance(init_i, RegWriteInst)
    assert init_i.dst == "r_i" and init_i.lit == "#0"

    # The "REG_WR s15 label e_0" instruction: find it by SRC.
    label_writes = [
        inst for inst in real if isinstance(inst, RegWriteInst) and inst.src == "label"
    ]
    assert len(label_writes) == 1
    assert label_writes[0].dst == "s15"
    assert label_writes[0].label is jt.entry_labels[0]

    # Final dispatch jump uses s15.
    dispatch_jumps = [
        inst for inst in real if isinstance(inst, JumpInst) and inst.addr == "s15"
    ]
    assert len(dispatch_jumps) == 1


def test_irjumptableloop_emit_body_words_5_uses_shift_add_seq():
    jt = _make_jt_loop(k=2, body_words=5)
    out: list[Instruction] = []
    jt.emit(out)

    # Among REG_WR with dst=s15: one is `label`, then the shift-add seq
    # contributes 2 adds (`s15 + r_i`). For body_words=5 the helper emits
    # `s15+=src; src<<=2; s15+=src` so two REG_WR-to-s15 with op form.
    s15_op_writes = [
        inst
        for inst in out
        if isinstance(inst, RegWriteInst) and inst.dst == "s15" and inst.src == "op"
    ]
    assert len(s15_op_writes) == 2
    assert all(w.op == "s15 + r_i" for w in s15_op_writes)


def test_irjumptableloop_malformed_k_raises():
    jt = IRJumpTableLoop(
        n_reg="r_n",
        counter_reg="r_i",
        k=1,  # < 2
        body_words=1,
        entry_labels=[],
        exit_label=Label("x"),
        bodies=[],
    )
    try:
        jt.emit([])
    except ValueError:
        return
    raise AssertionError("expected ValueError")
