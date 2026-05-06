"""Phase 8C/8D: Register-driven partial unroll via jump-table dispatch.

Emitted shape (k = unroll factor, must be a power of 2;
               body_words = pmem words per pure body):

    prologue:
      JUMP   exit    -if(Z)   -op(n_reg - #0)   ; skip if n == 0
      REG_WR i       op  (n_reg AND #(k-1))       ; i := n % k
      JUMP   entry_0 -if(Z)   -op(i - #0)        ; r == 0: full rounds from entry_0
      REG_WR i       op  (i - #k)                ; i := r - k  (< 0)
      REG_WR i       op  (ABS i)                 ; i := k - r  (entry offset)
      REG_WR s15     label  entry_0
      <shift-add: s15 += i * stride>              ; stride = body_words + 1
      REG_WR i       imm  #0                     ; reset counter
      JUMP   s15
    entry_0: <body copy 0>
              REG_WR i op (i + #1)
    ...
    entry_{k-1}: <body copy k-1>
              REG_WR i op (i + #1)
    back_edge:
      JUMP   exit    -if(NS)  -op(i - n_reg)
      JUMP   entry_0
    exit:

All blocks belonging to the jump-table loop body (entries + back-edge) have
`fix_inst_num=True` so that Post-LIR passes preserve stride accuracy.
"""

from __future__ import annotations

from typing import Optional

from ..instructions import Instruction, JumpInst, LabelInst, MetaInst, RegWriteInst
from ..labels import Label
from ..node import BasicBlockNode, BlockNode, _lower_block_node

_BIG_JUMP_PMEM_THRESHOLD = 2**11


def _needs_big_jump(pmem_size: Optional[int]) -> bool:
    return pmem_size is not None and pmem_size > _BIG_JUMP_PMEM_THRESHOLD


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


def build_jump_table_blocks(
    *,
    name: str,
    n_reg: str,
    counter_reg: str,
    k: int,
    body_words: int,
    entry_labels: list[Label],
    exit_label: Label,
    bodies: list[BlockNode],
    pmem_size: Optional[int] = None,
) -> list[BasicBlockNode]:
    """Build the BasicBlockNode list for a jump-table loop.

    All blocks that form the loop body and back-edge have `fix_inst_num=True`
    to preserve stride accuracy during Post-LIR optimisation.

    See module docstring for the full emitted shape.
    """
    if (
        k < 2
        or len(entry_labels) != k
        or len(bodies) != k
        or body_words <= 0
    ):
        raise ValueError(
            f"build_jump_table_blocks: invalid args k={k}, "
            f"entries={len(entry_labels)}, bodies={len(bodies)}, "
            f"body_words={body_words}"
        )

    i = counter_reg
    n = n_reg
    entry0 = entry_labels[0]

    stride = body_words + 1
    shift_add = shift_add_multiply(
        src_reg=i, dst_reg="s15", constant=stride, max_words=64
    )
    if shift_add is None:
        raise ValueError(
            f"build_jump_table_blocks: shift-add for stride={stride} "
            f"(body_words={body_words} + 1) failed"
        )

    result: list[BasicBlockNode] = []

    # ── meta marker ──
    result.append(
        BasicBlockNode(
            insts=[
                MetaInst(
                    type="JUMP_TABLE_LOOP_START",
                    name=name,
                    info=dict(n_reg=n, counter_reg=i, k=k, body_words=body_words),
                )
            ]
        )
    )

    # ── prologue ──
    # Guard: skip when n == 0.
    if _needs_big_jump(pmem_size):
        result.append(
            BasicBlockNode(
                insts=[RegWriteInst(dst="s15", src="label", label=exit_label)],
                branch=JumpInst(addr="s15", if_cond="Z", op=f"{n} - #0"),
            )
        )
    else:
        result.append(
            BasicBlockNode(
                branch=JumpInst(label=exit_label, if_cond="Z", op=f"{n} - #0"),
            )
        )

    # Compute remainder + dispatch into entry_{k-r}.
    # Uses i as scratch; reset to 0 before the dispatch JUMP.
    if _needs_big_jump(pmem_size):
        dispatch_insts: list[Instruction] = [
            RegWriteInst(dst=i, src="op", op=f"{n} AND #{k - 1}"),
        ]
        # r == 0: jump straight to entry_0
        result.append(BasicBlockNode(insts=dispatch_insts))
        result.append(
            BasicBlockNode(
                insts=[RegWriteInst(dst="s15", src="label", label=entry0)],
                branch=JumpInst(addr="s15", if_cond="Z", op=f"{i} - #0"),
            )
        )
    else:
        result.append(
            BasicBlockNode(
                insts=[RegWriteInst(dst=i, src="op", op=f"{n} AND #{k - 1}")],
                branch=JumpInst(label=entry0, if_cond="Z", op=f"{i} - #0"),
            )
        )

    # Compute entry offset and jump.
    offset_insts: list[Instruction] = [
        RegWriteInst(dst=i, src="op", op=f"{i} - #{k}"),   # i = r - k (< 0)
        RegWriteInst(dst=i, src="op", op=f"ABS {i}"),      # i = k - r (offset)
        RegWriteInst(dst="s15", src="label", label=entry0),
        *shift_add,                                          # s15 += i * stride
        RegWriteInst(dst=i, src="imm", lit="#0"),            # reset counter
    ]
    result.append(
        BasicBlockNode(
            insts=offset_insts,
            branch=JumpInst(addr="s15"),
        )
    )

    # ── k body copies (fix_inst_num=True to lock stride) ──
    for idx in range(k):
        entry_label = entry_labels[idx]
        # Lower the body BlockNode into BasicBlockNodes, then wrap with fix_inst_num.
        body_blocks = _lower_block_node(bodies[idx], pmem_size)
        # Merge into a single fixed block per entry: label + body insts + counter++.
        # Flatten all body insts into one BasicBlockNode so stride is exact.
        entry_insts = []
        for bb in body_blocks:
            entry_insts.extend(bb.labels)   # labels become inline (not entry labels)
            entry_insts.extend(bb.insts)
            if bb.branch is not None:
                entry_insts.append(bb.branch)
        entry_insts.append(RegWriteInst(dst=i, src="op", op=f"{i} + #1"))

        result.append(
            BasicBlockNode(
                labels=[LabelInst(name=entry_label)],
                insts=entry_insts,
                fix_inst_num=True,
            )
        )

    # ── back edge (fix_inst_num=True) ──
    if _needs_big_jump(pmem_size):
        back_insts: list[Instruction] = [RegWriteInst(dst="s15", src="label", label=exit_label)]
        result.append(
            BasicBlockNode(
                insts=back_insts,
                branch=JumpInst(addr="s15", if_cond="NS", op=f"{i} - {n}"),
                fix_inst_num=True,
            )
        )
        result.append(
            BasicBlockNode(
                insts=[RegWriteInst(dst="s15", src="label", label=entry0)],
                branch=JumpInst(addr="s15"),
                fix_inst_num=True,
            )
        )
    else:
        result.append(
            BasicBlockNode(
                branch=JumpInst(label=exit_label, if_cond="NS", op=f"{i} - {n}"),
                fix_inst_num=True,
            )
        )
        result.append(
            BasicBlockNode(
                branch=JumpInst(label=entry0),
                fix_inst_num=True,
            )
        )

    # ── exit ──
    result.append(
        BasicBlockNode(
            labels=[LabelInst(name=exit_label)],
            insts=[MetaInst(type="JUMP_TABLE_LOOP_END", name=name)],
        )
    )

    return result
