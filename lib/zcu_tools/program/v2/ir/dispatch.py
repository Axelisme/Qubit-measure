from __future__ import annotations

from .hw_semantics import needs_big_jump
from .instructions import BaseInst, JumpInst, LabelInst, RegWriteInst
from .labels import Label, LabelRef
from .node import BasicBlockNode
from .operands import AluExpr, AluOp, Register, SrcKeyword


def dispatch_entry_words(pmem_size: int | None) -> int:
    """Program-memory words per dispatch-table entry stub."""
    return 2 if needs_big_jump(pmem_size) else 1


def emit_dispatch_address_setup(
    *, index_reg: str, table_base: Label, pmem_size: int | None = None
) -> list[BaseInst]:
    """Compute ``s15 = &table_base + index_reg * entry_words``.

    Small-PMEM tables use 1-word ``JUMP label`` stubs, so one add is enough.
    Big-PMEM tables use 2-word ``WriteLabel + JUMP s15`` stubs, so we add the
    index twice instead of mutating the semantic index register in place.
    """
    index = Register(index_reg)
    s15 = Register("s15")

    insts: list[BaseInst] = [
        RegWriteInst(dst=s15, src=SrcKeyword.LABEL, label=LabelRef(table_base)),
    ]
    for _ in range(dispatch_entry_words(pmem_size)):
        insts.append(
            RegWriteInst(dst=s15, src=SrcKeyword.OP, op=AluExpr(s15, AluOp.ADD, index))
        )
    return insts


def build_dispatch_table_island(
    *,
    table_labels: list[Label],
    target_labels: list[Label],
    pmem_size: int | None = None,
) -> list[BasicBlockNode]:
    """Build fixed-width dispatch-table entry stubs.

    The table is the only place that still requires ``disable_opt=True``:
    every entry must keep the same physical width so computed jumps land on a
    valid stub, while the real target bodies remain completely free-form.
    """
    if len(table_labels) != len(target_labels):
        raise ValueError(
            "build_dispatch_table_island: table_labels and target_labels must "
            f"have the same length, got {len(table_labels)} and {len(target_labels)}"
        )
    if not table_labels:
        raise ValueError("build_dispatch_table_island: at least one entry is required")

    blocks: list[BasicBlockNode] = []
    for table_label, target_label in zip(table_labels, target_labels):
        if needs_big_jump(pmem_size):
            blocks.append(
                BasicBlockNode(
                    labels=[LabelInst(name=table_label, can_remove=False)],
                    insts=[
                        RegWriteInst(
                            dst=Register("s15"),
                            src=SrcKeyword.LABEL,
                            label=LabelRef(target_label),
                        )
                    ],
                    branch=JumpInst(addr=Register("s15")),
                    disable_opt=True,
                )
            )
        else:
            blocks.append(
                BasicBlockNode(
                    labels=[LabelInst(name=table_label, can_remove=False)],
                    branch=JumpInst(label=LabelRef(target_label)),
                    disable_opt=True,
                )
            )

    return blocks
