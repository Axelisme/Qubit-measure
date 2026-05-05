"""Phase 8C/8D: Register-driven partial unroll via jump-table dispatch.

This module defines the IR node `IRJumpTableLoop` and the codegen helper
`shift_add_multiply` used to materialize a register-driven unrolled loop
whose final partial round dispatches via a jump table.

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
      TEST   i - n_reg                      ; flags from i - n
      JUMP   exit -if(NS)                    ; i >= n: done
      REG_WR i op (n_reg - i)               ; i := r = n - i  (destroys counter)
      TEST   i - #k                         ; compare r vs k
      JUMP   fast_path -if(NS)              ; r >= k: take fast path
      ; ── dispatch (last partial round, 0 < r < k) ────────────────
      REG_WR i op (i - #k)                  ; i := r - k     (negative)
      REG_WR i op (ABS i)                   ; i := k - r     (entry offset)
      REG_WR s15 label entry_0              ; s15 := base
      <shift-add multiply: s15 += i * body_words>
      JUMP s15
    fast_path:
      REG_WR i op (n_reg - i)               ; i := n - r = original i (restore)
      JUMP entry_0
    exit:

Constraints encoded here:

* tProc v2 ALU is binary (`A op B`). First operand must be a register;
  the second can be a register or a literal `#N`. So expressions like
  `#k - i` or `(a op b) op c` are illegal — every binary op needs its own
  REG_WR word.
* Shift amounts (`SL`) max 15.
* `i` (the counter register) is reused as scratch within the dispatch
  block. The body copies have already finished executing by then, so
  the counter's runtime value is no longer observed by user code.
* `s15` is the only legal big-jump target on tProc v2 (`JUMP s15`), and
  is reserved by QICK as a scratch big-jump register, so writing to it
  inside the back edge is safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

from ..instructions import (
    Instruction,
    JumpInst,
    LabelInst,
    MetaInst,
    RegWriteInst,
    TestInst,
)
from ..labels import Label
from ..node import BlockNode, IRNode


def shift_add_multiply(
    src_reg: str, dst_reg: str, constant: int, max_words: int
) -> Optional[list[Instruction]]:
    """Emit a shift-add sequence so that `dst_reg += src_reg * constant`.

    `src_reg` is treated as a scratch register: it is shifted in place
    between bit emissions, so its value is destroyed.

    Returns None when the resulting sequence would exceed `max_words`
    REG_WR instructions, or when `constant <= 0`.

    The encoding processes `constant` from LSB to MSB. For each set bit
    `b`, we emit `dst += src` (after src has been shifted to bit b). The
    shifts are accumulated, not re-applied, so a constant like `0b101`
    costs 1 add + 1 shift + 1 add = 3 words rather than 2 shifts + 2 adds.
    """
    if constant <= 0:
        return None

    insts: list[Instruction] = []
    prev_bit = 0
    bits = []
    b = 0
    c = constant
    while c > 0:
        if c & 1:
            bits.append(b)
        c >>= 1
        b += 1

    for bit in bits:
        delta = bit - prev_bit
        if delta > 0:
            if delta > 15:
                return None  # ALU shift amount max 15
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


@dataclass
class IRJumpTableLoop(IRNode):
    """Register-driven loop unrolled into k body copies with last-round dispatch.

    Fields:
      n_reg:        register holding the runtime iteration count (read-only).
      counter_reg:  loop counter register `i`. Destroyed inside the dispatch
                    block; user code must not read it after the loop.
      k:            unroll factor (must be >= 2; for register-driven loops
                    Phase 8D forces k to be a power of 2, but this node
                    itself does not enforce the constraint).
      body_words:   pmem words per single body copy. Baked into the
                    dispatch shift-add multiply.
      entry_labels: k labels marking the head of each body copy.
      exit_label:   single exit label.
      bodies:       k BlockNodes — independent deepcopies of the source body.
      dispatch_words: shift-add sequence length cap (carried for diagnostics).
    """

    n_reg: str = ""
    counter_reg: str = ""
    k: int = 0
    body_words: int = 0
    entry_labels: list[Label] = field(default_factory=list)
    exit_label: Optional[Label] = None
    bodies: list[BlockNode] = field(default_factory=list)
    name: str = ""

    def children(self) -> Iterator[IRNode]:
        yield from self.bodies

    def emit(self, inst_list: list[Instruction]) -> None:
        if (
            self.k < 2
            or len(self.entry_labels) != self.k
            or len(self.bodies) != self.k
            or self.exit_label is None
            or self.body_words <= 0
        ):
            raise ValueError(
                f"IRJumpTableLoop is malformed: k={self.k}, "
                f"entries={len(self.entry_labels)}, bodies={len(self.bodies)}, "
                f"body_words={self.body_words}, exit={self.exit_label!r}"
            )

        i = self.counter_reg
        n = self.n_reg
        entry0 = self.entry_labels[0]
        exit_label = self.exit_label
        fast_path = Label.make_new(f"{self.name}_jt_fast")

        inst_list.append(
            MetaInst(
                type="JUMP_TABLE_LOOP_START",
                name=self.name,
                info=dict(
                    n_reg=n,
                    counter_reg=i,
                    k=self.k,
                    body_words=self.body_words,
                ),
            )
        )

        # ── prologue ────────────────────────────────────────────────
        inst_list.append(TestInst(op=f"{n} - #0"))
        inst_list.append(JumpInst(label=exit_label, if_cond="Z"))
        inst_list.append(RegWriteInst(dst=i, src="imm", lit="#0"))

        # ── k body copies ───────────────────────────────────────────
        for idx in range(self.k):
            inst_list.append(LabelInst(name=self.entry_labels[idx]))
            self.bodies[idx].emit(inst_list)
            inst_list.append(
                RegWriteInst(dst=i, src="op", op=f"{i} + #1")
            )

        # ── back edge ───────────────────────────────────────────────
        inst_list.append(TestInst(op=f"{i} - {n}"))
        inst_list.append(JumpInst(label=exit_label, if_cond="NS"))
        # i := r = n - i  (destroys counter)
        inst_list.append(RegWriteInst(dst=i, src="op", op=f"{n} - {i}"))
        inst_list.append(TestInst(op=f"{i} - #{self.k}"))
        inst_list.append(JumpInst(label=fast_path, if_cond="NS"))

        # ── dispatch (0 < r < k) ────────────────────────────────────
        # i := r - k       (negative)
        inst_list.append(RegWriteInst(dst=i, src="op", op=f"{i} - #{self.k}"))
        # i := |r - k| = k - r
        inst_list.append(RegWriteInst(dst=i, src="op", op=f"ABS {i}"))
        # s15 := base address of entry_0
        inst_list.append(RegWriteInst(dst="s15", src="label", label=entry0))

        shift_add = shift_add_multiply(
            src_reg=i, dst_reg="s15", constant=self.body_words, max_words=64
        )
        if shift_add is None:
            # Defensive: caller (UnrollSmallLoopPass) should have rejected
            # this case before constructing the node.
            raise ValueError(
                f"IRJumpTableLoop: shift-add for body_words={self.body_words} "
                "failed at emit time"
            )
        inst_list.extend(shift_add)
        inst_list.append(JumpInst(addr="s15"))

        # ── fast path: r >= k, restore i and continue ───────────────
        inst_list.append(LabelInst(name=fast_path))
        # i := n - r = original i value (since r = n - i_orig)
        inst_list.append(RegWriteInst(dst=i, src="op", op=f"{n} - {i}"))
        inst_list.append(JumpInst(label=entry0))

        # ── exit ────────────────────────────────────────────────────
        inst_list.append(LabelInst(name=exit_label))
        inst_list.append(MetaInst(type="JUMP_TABLE_LOOP_END", name=self.name))
