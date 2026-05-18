"""DmemDispatchPass: lower an IRDispatch into a dmem-backed jump table.

Purpose
-------
The default dispatch lowering (``IRParser._lower_dispatch``) builds a fixed-width
*program-memory* table island: ``n`` stub blocks, each a one-jump forwarder,
guarded by ``disable_opt=True`` so their physical width stays constant. Reaching
case ``i`` costs 2 jumps (into the stub, then the stub forwards) and ``O(n)``
program-memory words.

This pass replaces that with a *data-memory* address table. The table is a run
of dmem words, each holding the resolved program address of one case entry.
Dispatch becomes:

    REG_WR s15 op (index + table_base)   ; s15 = dmem address of table[index]
    REG_WR s15 dmem [s15]                ; s15 = dmem[s15] = entry program addr
    JUMP [s15]                           ; one jump

Cost: 1 jump, 3 instructions (independent of ``n``), ``n`` dmem words. The
``disable_opt`` invariant disappears — these are ordinary instructions.

QICK Hardware Notes
-------------------
- ``s15`` doubles as the address scratch: the address computation and the dmem
  read are adjacent and transient, and ``value_reg`` is a general register
  (loop counter / branch compare reg), never ``s15`` — so no separate scratch
  register is needed.
- ``REG_WR s15 dmem [s15]`` is legal: ``dst`` and the ``[addr]`` register are
  independent instruction fields; the assembler accepts an ``sN`` dmem address.
- The dmem table holds *program addresses*; ``DmemAddr`` is an unresolved
  reference to it. The pipeline resolve step (after every clone-capable pass)
  assigns the concrete dmem base offset and writes the entry addresses.

Decision Notes
--------------
This is an optimization pass, peer to ``SimplifyDispatchPass``: that one handles
``k == 2`` with a single conditional jump (runs earlier in the pass chain), this
one handles ``k >= 2`` as a dmem table.  When both are active, ``SimplifyDispatchPass``
wins for k==2; if it is disabled, ``DmemDispatchPass`` handles k==2 as well.
An ``IRDispatch`` not transformed by either pass (``disable_all_opt`` bypass,
direct ``unparse`` in tests) still falls back to the program-memory island in
``_lower_dispatch`` — the two dispatch implementations coexist.

``DmemDispatchPass`` returns a ``BlockNode`` of plain instructions with **no**
structural ``MetaInst``: a dmem jump table is lowered code, not a structure.
"""

from __future__ import annotations

from typing_extensions import Optional

from ...hw_semantics import needs_big_jump
from ...instructions import DmemReadInst, JumpInst, RegWriteInst
from ...labels import LabelRef
from ...node import BasicBlockNode, BlockNode, IRDispatch, IRNode
from ...operands import AluExpr, AluOp, DmemAddr, Immediate, Register, SrcKeyword
from ...pipeline import AbsIRTreePass, PipeLineContext


class DmemDispatchPass(AbsIRTreePass):
    """Lower an IRDispatch (k >= 2) into a dmem-backed jump table BlockNode."""

    def transform(
        self,
        node: IRNode,
        ctx: PipeLineContext,
    ) -> Optional[IRNode]:
        if not isinstance(node, IRDispatch):
            return None
        # k == 1 is not a meaningful dispatch (nothing to choose between).
        if len(node.target_labels) <= 1:
            return None

        pmem_size = ctx.config.pmem_capacity
        n = len(node.target_labels)
        value_reg = node.value_reg
        s15 = Register("s15")

        blocks: list[BasicBlockNode] = []

        # Out-of-range guard: index >= n → jump to the last case.
        # NS (non-negative) on (value_reg - n) means value_reg >= n.
        last_label = node.target_labels[-1]
        op_guard = AluExpr(value_reg, AluOp.SUB, Immediate(n))
        if needs_big_jump(pmem_size):
            blocks.append(
                BasicBlockNode(
                    insts=[
                        RegWriteInst(
                            dst=s15,
                            src=SrcKeyword.LABEL,
                            label=LabelRef(last_label),
                        )
                    ],
                    branch=JumpInst(addr=s15, if_cond="NS", op=op_guard),
                )
            )
        else:
            blocks.append(
                BasicBlockNode(
                    branch=JumpInst(
                        label=LabelRef(last_label), if_cond="NS", op=op_guard
                    )
                )
            )

        # dmem table lookup + indirect jump.
        #   s15 = index + table_base   (DmemAddr resolves to the base offset)
        #   s15 = dmem[s15]            (the entry's program address)
        #   JUMP [s15]
        table_ref = DmemAddr(table_labels=tuple(node.target_labels))
        blocks.append(
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=s15,
                        src=SrcKeyword.OP,
                        op=AluExpr(value_reg, AluOp.ADD, table_ref),
                    ),
                    DmemReadInst(dst=s15, addr=s15),
                ],
                branch=JumpInst(addr=s15),
            )
        )

        return BlockNode(insts=list(blocks))
