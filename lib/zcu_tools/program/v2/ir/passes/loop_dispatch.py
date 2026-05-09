"""Register-driven partial unroll via dispatch-table island.

The old "calculated stride" scheme required every body copy to keep the same
physical width. The new shape keeps only the tiny dispatch-table stubs fixed:

    prologue:
      JUMP exit -if(Z) -op(n_reg - #0)
      REG_WR i op (n_reg AND #(k-1))             ; r := n % k
      JUMP entry_0 -if(Z) -op(i - #0)           ; r == 0 -> full rounds
      REG_WR i op (i - #k)
      REG_WR i op (ABS i)                        ; i := k - r
      REG_WR s15 label table_0
      REG_WR s15 op (s15 + i) [x entry_words]
      REG_WR i imm #0
      JUMP s15

    table_0: JUMP entry_0
    table_1: JUMP entry_1
    ...

    entry_0: <free-form body copy 0>
    ...
    entry_{k-1}: <free-form body copy k-1>

    back_edge:
      JUMP exit -if(NS) -op(i - n_reg)
      JUMP entry_0

Only the dispatch-table stub blocks have ``fix_addr_size=True``.
"""

from __future__ import annotations

from typing import Optional

from ..dispatch import build_dispatch_table_island, emit_dispatch_address_setup
from ..factory import IRParser, _needs_big_jump
from ..instructions import BaseInst, JumpInst, LabelInst, RegWriteInst
from ..labels import Label
from ..node import BasicBlockNode, BlockNode
from ..operands import AluExpr, Literal, Register


def build_jump_table_blocks(
    *,
    n_reg: str,
    counter_reg: str,
    k: int,
    entry_labels: list[Label],
    exit_label: Label,
    bodies: list[BlockNode],
    pmem_size: Optional[int] = None,
) -> list[BasicBlockNode]:
    """Build the BasicBlockNode list for a dispatch-table loop."""
    if k < 2 or len(entry_labels) != k or len(bodies) != k:
        raise ValueError(
            f"build_jump_table_blocks: invalid args k={k}, "
            f"entries={len(entry_labels)}, bodies={len(bodies)}, "
            f"entry_labels={len(entry_labels)}"
        )

    i = Register(counter_reg)
    n = Register(n_reg)
    entry0 = entry_labels[0]
    table_labels = [Label.make_new(f"{entry0.name}_dispatch_{idx}") for idx in range(k)]

    result: list[BasicBlockNode] = []

    # ── prologue ──
    # Guard: skip when n == 0.
    if _needs_big_jump(pmem_size):
        result.append(
            BasicBlockNode(
                insts=[
                    RegWriteInst(dst=Register("s15"), src="label", label=exit_label)
                ],
                branch=JumpInst(
                    addr=Register("s15"), if_cond="Z", op=AluExpr(n, "-", Literal("#0"))
                ),
            )
        )
    else:
        result.append(
            BasicBlockNode(
                branch=JumpInst(
                    label=exit_label, if_cond="Z", op=AluExpr(n, "-", Literal("#0"))
                ),
            )
        )

    # Compute remainder + dispatch into entry_{k-r}.
    # Uses counter_reg as scratch only until it is reset to 0.
    if _needs_big_jump(pmem_size):
        dispatch_insts: list[BaseInst] = [
            RegWriteInst(dst=i, src="op", op=AluExpr(n, "AND", Literal(f"#{k - 1}"))),
        ]
        # r == 0: jump straight to entry_0
        result.append(BasicBlockNode(insts=dispatch_insts))
        result.append(
            BasicBlockNode(
                insts=[RegWriteInst(dst=Register("s15"), src="label", label=entry0)],
                branch=JumpInst(
                    addr=Register("s15"), if_cond="Z", op=AluExpr(i, "-", Literal("#0"))
                ),
            )
        )
    else:
        result.append(
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=i, src="op", op=AluExpr(n, "AND", Literal(f"#{k - 1}"))
                    )
                ],
                branch=JumpInst(
                    label=entry0, if_cond="Z", op=AluExpr(i, "-", Literal("#0"))
                ),
            )
        )

    # Compute entry offset and jump.
    offset_insts: list[BaseInst] = [
        RegWriteInst(
            dst=i, src="op", op=AluExpr(i, "-", Literal(f"#{k}"))
        ),  # i = r - k (< 0)
        RegWriteInst(dst=i, src="op", op=AluExpr(i, "ABS")),  # i = k - r (offset)
        *emit_dispatch_address_setup(
            index_reg=counter_reg,
            table_base=table_labels[0],
            pmem_size=pmem_size,
        ),
        RegWriteInst(dst=i, src="imm", lit=Literal("#0")),  # reset counter
    ]
    result.append(
        BasicBlockNode(
            insts=offset_insts,
            branch=JumpInst(addr=Register("s15")),
        )
    )

    # ── dispatch-table island (the only fixed-width region) ──
    result.extend(
        build_dispatch_table_island(
            table_labels=table_labels,
            target_labels=entry_labels,
            pmem_size=pmem_size,
        )
    )

    # ── k body copies (free-form; no fix_addr_size requirement) ──
    for idx in range(k):
        entry_label = entry_labels[idx]
        body_blocks = IRParser(pmem_size=pmem_size).lower_block(bodies[idx])
        if not body_blocks:
            result.append(BasicBlockNode(labels=[LabelInst(name=entry_label)]))
            continue
        body_blocks[0].labels.insert(0, LabelInst(name=entry_label))
        result.extend(body_blocks)

    # ── back edge (free-form) ──
    if _needs_big_jump(pmem_size):
        back_insts: list[BaseInst] = [
            RegWriteInst(dst=Register("s15"), src="label", label=exit_label)
        ]
        result.append(
            BasicBlockNode(
                insts=back_insts,
                branch=JumpInst(
                    addr=Register("s15"), if_cond="NS", op=AluExpr(i, "-", n)
                ),
            )
        )
        result.append(
            BasicBlockNode(
                insts=[RegWriteInst(dst=Register("s15"), src="label", label=entry0)],
                branch=JumpInst(addr=Register("s15")),
            )
        )
    else:
        result.append(
            BasicBlockNode(
                branch=JumpInst(label=exit_label, if_cond="NS", op=AluExpr(i, "-", n)),
            )
        )
        result.append(
            BasicBlockNode(
                branch=JumpInst(label=entry0),
            )
        )

    # ── exit ──
    result.append(
        BasicBlockNode(
            labels=[LabelInst(name=exit_label, can_remove=True)],
        )
    )

    return result
