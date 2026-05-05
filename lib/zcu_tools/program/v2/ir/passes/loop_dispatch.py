"""Phase 8C/8D: Register-driven partial unroll via jump-table dispatch.

`IRJumpTableLoop` itself lives in `..node` so that analysis helpers can
recognize it without circular imports. This module owns the codegen
(`emit_jump_table_loop`) and the `shift_add_multiply` helper.

Emitted shape (k = unroll factor, body_words = pmem words per body copy):

    prologue:
      TEST   n_reg - #0
      JUMP   exit -if(Z)
      REG_WR i imm #0                       ; i := 0
    entry_0: <body copy 0>; REG_WR i op (i + #1)
    entry_1: <body copy 1>; REG_WR i op (i + #1)
    ...
    entry_{k-1}: <body copy k-1>; REG_WR i op (i + #1)
    back_edge:
      TEST   i - n_reg
      JUMP   exit -if(NS)                    ; i >= n: done
      REG_WR i op (n_reg - i)               ; i := r = n - i  (destroys counter)
      TEST   i - #k
      JUMP   fast_path -if(NS)              ; r >= k: take fast path
      ; ── dispatch (last partial round, 0 < r < k) ────────────────
      REG_WR i op (i - #k)                  ; i := r - k  (negative)
      REG_WR i op (ABS i)                   ; i := k - r  (entry offset)
      REG_WR s15 label entry_0              ; s15 := base
      <shift-add multiply: s15 += i * body_words>
      JUMP s15
    fast_path:
      REG_WR i op (n_reg - i)               ; restore i = original counter
      JUMP entry_0
    exit:

Constraints:

* tProc v2 ALU is binary (`A op B`); first operand must be a register;
  every binary op needs its own REG_WR word.
* Shift amount in `SL` is capped at 15 by the assembler.
* The counter register is reused as scratch in the dispatch block.
  Body copies have already executed by then so the runtime value of
  the counter is no longer observed.
* `s15` is the only legal big-jump target (`JUMP s15`) and is treated
  by QICK as a scratch big-jump register.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..instructions import (
    Instruction,
    JumpInst,
    LabelInst,
    MetaInst,
    RegWriteInst,
    TestInst,
)
from ..labels import Label

if TYPE_CHECKING:
    from ..node import IRJumpTableLoop


def shift_add_multiply(
    src_reg: str, dst_reg: str, constant: int, max_words: int
) -> Optional[list[Instruction]]:
    """Emit a shift-add sequence so that `dst_reg += src_reg * constant`.

    `src_reg` is treated as a scratch register (shifted in place). Returns
    None when the resulting sequence would exceed `max_words` REG_WR
    instructions, when constant <= 0, or when any required shift > 15.
    """
    if constant <= 0:
        return None

    insts: list[Instruction] = []
    bits: list[int] = []
    b = 0
    c = constant
    while c > 0:
        if c & 1:
            bits.append(b)
        c >>= 1
        b += 1

    prev_bit = 0
    for bit in bits:
        delta = bit - prev_bit
        if delta > 0:
            if delta > 15:
                return None
            insts.append(
                RegWriteInst(dst=src_reg, src="op", op=f"{src_reg} << #{delta}")
            )
            prev_bit = bit
        insts.append(
            RegWriteInst(dst=dst_reg, src="op", op=f"{dst_reg} + {src_reg}")
        )
        if len(insts) > max_words:
            return None

    return insts


def emit_jump_table_loop(
    node: "IRJumpTableLoop", inst_list: list[Instruction]
) -> None:
    """Codegen for IRJumpTableLoop.emit(). See module docstring for shape."""
    if (
        node.k < 2
        or len(node.entry_labels) != node.k
        or len(node.bodies) != node.k
        or node.exit_label is None
        or node.body_words <= 0
    ):
        raise ValueError(
            f"IRJumpTableLoop is malformed: k={node.k}, "
            f"entries={len(node.entry_labels)}, bodies={len(node.bodies)}, "
            f"body_words={node.body_words}, exit={node.exit_label!r}"
        )

    i = node.counter_reg
    n = node.n_reg
    entry0 = node.entry_labels[0]
    exit_label = node.exit_label
    fast_path = Label.make_new(f"{node.name}_jt_fast")

    inst_list.append(
        MetaInst(
            type="JUMP_TABLE_LOOP_START",
            name=node.name,
            info=dict(
                n_reg=n,
                counter_reg=i,
                k=node.k,
                body_words=node.body_words,
            ),
        )
    )

    # ── prologue ──
    inst_list.append(TestInst(op=f"{n} - #0"))
    inst_list.append(JumpInst(label=exit_label, if_cond="Z"))
    inst_list.append(RegWriteInst(dst=i, src="imm", lit="#0"))

    # ── k body copies ──
    for idx in range(node.k):
        inst_list.append(LabelInst(name=node.entry_labels[idx]))
        node.bodies[idx].emit(inst_list)
        inst_list.append(RegWriteInst(dst=i, src="op", op=f"{i} + #1"))

    # ── back edge ──
    inst_list.append(TestInst(op=f"{i} - {n}"))
    inst_list.append(JumpInst(label=exit_label, if_cond="NS"))
    inst_list.append(RegWriteInst(dst=i, src="op", op=f"{n} - {i}"))
    inst_list.append(TestInst(op=f"{i} - #{node.k}"))
    inst_list.append(JumpInst(label=fast_path, if_cond="NS"))

    # ── dispatch (0 < r < k) ──
    inst_list.append(RegWriteInst(dst=i, src="op", op=f"{i} - #{node.k}"))
    inst_list.append(RegWriteInst(dst=i, src="op", op=f"ABS {i}"))
    inst_list.append(RegWriteInst(dst="s15", src="label", label=entry0))

    shift_add = shift_add_multiply(
        src_reg=i, dst_reg="s15", constant=node.body_words, max_words=64
    )
    if shift_add is None:
        raise ValueError(
            f"IRJumpTableLoop: shift-add for body_words={node.body_words} "
            "failed at emit time"
        )
    inst_list.extend(shift_add)
    inst_list.append(JumpInst(addr="s15"))

    # ── fast path: r >= k, restore i and continue ──
    inst_list.append(LabelInst(name=fast_path))
    inst_list.append(RegWriteInst(dst=i, src="op", op=f"{n} - {i}"))
    inst_list.append(JumpInst(label=entry0))

    # ── exit ──
    inst_list.append(LabelInst(name=exit_label))
    inst_list.append(MetaInst(type="JUMP_TABLE_LOOP_END", name=node.name))


# Re-export for backwards compatibility with tests/imports that expect
# IRJumpTableLoop in this module.
from ..node import IRJumpTableLoop  # noqa: E402, F401
