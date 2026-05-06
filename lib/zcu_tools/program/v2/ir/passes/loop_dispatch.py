"""Phase 8C/8D: Register-driven partial unroll via jump-table dispatch.

`IRJumpTableLoop` itself lives in `..node` so that analysis helpers can
recognize it without circular imports. This module owns the codegen
(`emit_jump_table_loop`) and the `shift_add_multiply` helper.

Emitted shape (k = unroll factor, must be a power of 2;
               body_words = pmem words per pure body):

    prologue:
      JUMP   exit    -if(Z)   -op(n_reg - #0)   ; skip if n == 0
      REG_WR i       op  (n_reg AND #(k-1))       ; i := n % k  (remainder r; i is scratch)
      JUMP   entry_0 -if(Z)   -op(i - #0)        ; r == 0: run full rounds from entry_0
      ; dispatch: compute entry offset = k - r, jump to entry_{k-r}
      REG_WR i       op  (i - #k)                ; i := r - k  (< 0)
      REG_WR i       op  (ABS i)                 ; i := k - r  (entry offset)
      REG_WR s15     label  entry_0              ; s15 := base address of entry_0
      <shift-add: s15 += i * stride>              ; stride = body_words + 1
      REG_WR i       imm  #0                     ; i := 0  (reset counter before first entry)
      JUMP   s15                                  ; jump to entry_{k-r}
    entry_0: <body copy 0>
              REG_WR i op (i + #1)
    entry_1: <body copy 1>
              REG_WR i op (i + #1)
    ...
    entry_{k-1}: <body copy k-1>
              REG_WR i op (i + #1)
    back_edge:
      JUMP   exit    -if(NS)  -op(i - n_reg)     ; i >= n: done
      JUMP   entry_0                              ; continue next full round
    exit:

Correctness argument (k is a power of 2, so n AND #(k-1) == n % k):
  * n == 0: guard exits immediately.
  * n % k == 0: jump straight to entry_0; run n/k full rounds of k bodies;
    back_edge fires when i == n.
  * n % k == r > 0: dispatch jumps to entry_{k-r}; first partial round
    executes r bodies so i == r at back_edge; each subsequent full round
    adds k; i reaches n after exactly (n-r)/k more rounds.

Constraints:

* k MUST be a power of 2 so that `n AND #(k-1)` equals `n % k`.
* tProc v2 ALU is binary (`A op B`); first operand must be a register;
  every binary op needs its own REG_WR word.
* Shift amount in `SL` is capped at 15 by the assembler.
* Counter `i` is reused as scratch during prologue dispatch and is reset
  to 0 before the dispatch JUMP, so every entry sees i == 0 on entry.
* `s15` is the only legal big-jump target (`JUMP s15`) and is treated
  by QICK as a scratch big-jump register.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..instructions import Instruction, JumpInst, LabelInst, MetaInst, RegWriteInst
from ..labels import Label

if TYPE_CHECKING:
    from ..node import IRJumpTableLoop

_BIG_JUMP_PMEM_THRESHOLD = 2**11


def _needs_big_jump(pmem_size: Optional[int]) -> bool:
    return pmem_size is not None and pmem_size > _BIG_JUMP_PMEM_THRESHOLD


def _emit_label_jump(
    inst_list: list[Instruction],
    *,
    target: Label,
    pmem_size: Optional[int],
    if_cond: Optional[str] = None,
    op: Optional[str] = None,
) -> None:
    if _needs_big_jump(pmem_size):
        inst_list.append(RegWriteInst(dst="s15", src="label", label=target))
        inst_list.append(JumpInst(addr="s15", if_cond=if_cond, op=op))
        return
    inst_list.append(JumpInst(label=target, if_cond=if_cond, op=op))


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
        insts.append(RegWriteInst(dst=dst_reg, src="op", op=f"{dst_reg} + {src_reg}"))
        if len(insts) > max_words:
            return None

    return insts


def emit_jump_table_loop(
    node: "IRJumpTableLoop",
    inst_list: list[Instruction],
    *,
    pmem_size: Optional[int] = None,
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
    k = node.k
    entry0 = node.entry_labels[0]
    exit_label = node.exit_label

    inst_list.append(
        MetaInst(
            type="JUMP_TABLE_LOOP_START",
            name=node.name,
            info=dict(
                n_reg=n,
                counter_reg=i,
                k=k,
                body_words=node.body_words,
            ),
        )
    )

    stride = node.body_words + 1  # body + per-iteration counter increment
    shift_add = shift_add_multiply(
        src_reg=i, dst_reg="s15", constant=stride, max_words=64
    )
    if shift_add is None:
        raise ValueError(
            f"IRJumpTableLoop: shift-add for stride={stride} "
            f"(body_words={node.body_words} + 1) failed at emit time"
        )

    # ── prologue ──
    # Guard: skip entire loop when n == 0.
    _emit_label_jump(
        inst_list, target=exit_label, pmem_size=pmem_size, if_cond="Z", op=f"{n} - #0"
    )
    # Compute remainder r = n % k into i (k is a power of 2, so AND is exact).
    # i is used as scratch here and reset to 0 before the dispatch JUMP.
    # Critically, i (not s15) holds the remainder so that _emit_label_jump
    # below can use -op(i - #0) without touching s15 — a big-pmem cond-jump
    # would overwrite s15 with the label address before the test fires.
    inst_list.append(RegWriteInst(dst=i, src="op", op=f"{n} AND #{k - 1}"))  # i = n % k
    # If r == 0 the loop is perfectly divisible; jump straight to entry_0.
    _emit_label_jump(
        inst_list, target=entry0, pmem_size=pmem_size, if_cond="Z", op=f"{i} - #0"
    )
    # Compute entry offset = k - r, still in i.
    inst_list.append(RegWriteInst(dst=i, src="op", op=f"{i} - #{k}"))  # i = r - k (< 0)
    inst_list.append(RegWriteInst(dst=i, src="op", op=f"ABS {i}"))     # i = k - r (offset)
    inst_list.append(RegWriteInst(dst="s15", src="label", label=entry0))
    inst_list.extend(shift_add)                                          # s15 += i * stride
    inst_list.append(RegWriteInst(dst=i, src="imm", lit="#0"))           # reset counter
    inst_list.append(JumpInst(addr="s15"))                               # → entry_{k-r}

    # ── k body copies (each followed by counter += 1) ──
    for idx in range(k):
        inst_list.append(LabelInst(name=node.entry_labels[idx]))
        node.bodies[idx].emit(inst_list, pmem_size=pmem_size)
        inst_list.append(RegWriteInst(dst=i, src="op", op=f"{i} + #1"))

    # ── back edge ──
    _emit_label_jump(
        inst_list, target=exit_label, pmem_size=pmem_size, if_cond="NS", op=f"{i} - {n}"
    )
    _emit_label_jump(inst_list, target=entry0, pmem_size=pmem_size)

    # ── exit ──
    inst_list.append(LabelInst(name=exit_label))
    inst_list.append(MetaInst(type="JUMP_TABLE_LOOP_END", name=node.name))


# Re-export for backwards compatibility with tests/imports that expect
# IRJumpTableLoop in this module.
from ..node import IRJumpTableLoop  # noqa: E402, F401
