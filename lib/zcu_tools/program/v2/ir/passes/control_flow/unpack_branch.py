"""UnpackIRBranchPass: expand an IRBranch into a dispatch node + case bodies.

Purpose
-------
``IRBranch`` is a structural node: a value-indexed multi-way branch. Its
lowering needs a dispatch mechanism (jump table) plus the case bodies.

Historically ``IRBranch`` was lowered entirely inside ``IRParser._unparse_node``
— it built an ``IRDispatch`` there and immediately flattened it. That made the
branch's ``IRDispatch`` invisible to IR-tree passes: ``DmemDispatchPass`` and
``SimplifyDispatchPass`` run on the tree, *before* unparse, so they only ever
saw dispatch nodes produced by ``UnrollLoopPass`` — never a branch's.

This pass fixes that asymmetry. It expands ``IRBranch`` *on the tree* into a
``BlockNode``::

    BlockNode([
        IRDispatch(value_reg, case_entry_labels),   # a real tree node
        case_0  (BlockNode, entry label prepended, jump-to-end appended),
        case_1  ...,
        ...
        BasicBlockNode(end_label),                  # fall-through target
    ])

The pipeline then re-recurses this BlockNode, so the ``IRDispatch`` is visited
by ``SimplifyDispatchPass`` (k==2) / ``DmemDispatchPass`` (k>=3) like any other
dispatch. ``IRBranch`` thus becomes a pass-expanded structural node, symmetric
with how ``UnrollLoopPass`` expands ``IRLoop``.

Decision Notes
--------------
Runs *before* ``SimplifyDispatchPass`` / ``DmemDispatchPass`` so the dispatch
node it produces is seen by them. ``IRParser._unparse_node``'s ``IRBranch``
branch is kept as a fallback: when IR optimization is disabled (no tree passes
run) or a test unparses an ``IRBranch`` tree directly, that path still lowers
the branch — symmetric with ``_lower_dispatch`` keeping the pmem island.

The expansion emits **no** structural ``MetaInst``: an unpacked branch is
lowered structure, and (like ``UnrollLoopPass`` output) must not be re-parsed
back into an ``IRBranch``, which would cause the pipeline to expand it again.
"""

from __future__ import annotations

from typing_extensions import Optional

from ...dispatch import needs_big_jump
from ...instructions import JumpInst, LabelInst, RegWriteInst
from ...labels import Label, LabelRef, make_label
from ...node import (
    BasicBlockNode,
    BlockNode,
    IRBranch,
    IRDispatch,
    IRNode,
    _collect_subtree_names,
)
from ...operands import Register, SrcKeyword
from ...pipeline import AbsIRTreePass, PipeLineContext


def _first_basic_block(node: IRNode) -> Optional[BasicBlockNode]:
    """Return the first BasicBlockNode reachable as the head of `node`, if any."""
    if isinstance(node, BasicBlockNode):
        return node
    if isinstance(node, BlockNode) and node.insts:
        return _first_basic_block(node.insts[0])
    return None


def _case_entry_label(
    case: IRNode, fallback_name: str, allocated: set[str]
) -> tuple[IRNode, Label]:
    """Return (case_with_entry_label, entry_label).

    A case body usually already carries an entry label (the macro layer emits
    ``{branch}_case_entry_{i}``); reuse it so the dispatch target stays in sync
    with the label that physically marks the case. Only when the case has no
    leading label do we synthesise a fresh one and prepend it.
    """
    head = _first_basic_block(case)
    if head is not None and head.labels:
        return case, head.labels[0].name

    label = make_label(fallback_name, allocated)
    new_label = LabelInst(name=label, can_remove=True)
    if head is not None:
        head.labels.insert(0, new_label)
        return case, label
    # Case has no BasicBlockNode head: prepend a fresh labelled block.
    entry = BasicBlockNode(labels=[new_label])
    if isinstance(case, BlockNode):
        return BlockNode(insts=[entry, *case.insts]), label
    return BlockNode(insts=[entry, case]), label


class UnpackIRBranchPass(AbsIRTreePass):
    """Expand an IRBranch into an IRDispatch node followed by case bodies."""

    def transform(
        self,
        node: IRNode,
        ctx: PipeLineContext,
    ) -> Optional[IRNode]:
        if not isinstance(node, IRBranch):
            return None

        pmem_size = ctx.config.pmem_capacity
        n = len(node.cases)

        # Local allocated set: seed from every label / structure name already
        # present in the branch subtree so any synthesised label is unique.
        allocated: set[str] = set()
        for case in node.cases:
            labels, structs = _collect_subtree_names(case)
            allocated |= labels | structs

        # Resolve each case's entry label (reuse the macro-emitted one when
        # present, synthesise otherwise) so the dispatch targets stay in sync
        # with the labels that physically mark the cases.
        cases_with_entry: list[IRNode] = []
        case_entry_labels: list[Label] = []
        for idx, case in enumerate(node.cases):
            case2, entry = _case_entry_label(
                case, f"{node.name}_case_entry_{idx}", allocated
            )
            cases_with_entry.append(case2)
            case_entry_labels.append(entry)

        end_label = make_label(f"{node.name}_end", allocated)

        result: list[IRNode] = [
            IRDispatch(
                name=node.name,
                value_reg=node.compare_reg,
                target_labels=list(case_entry_labels),
            )
        ]

        for idx, case in enumerate(cases_with_entry):
            is_last = idx == n - 1
            result.append(case)
            # Every case but the last jumps to end_label so it does not fall
            # through into the next case body.
            if not is_last:
                if needs_big_jump(pmem_size):
                    result.append(
                        BasicBlockNode(
                            insts=[
                                RegWriteInst(
                                    dst=Register("s15"),
                                    src=SrcKeyword.LABEL,
                                    label=LabelRef(end_label),
                                )
                            ],
                            branch=JumpInst(addr=Register("s15")),
                        )
                    )
                else:
                    result.append(
                        BasicBlockNode(branch=JumpInst(label=LabelRef(end_label)))
                    )

        # Fall-through target for the jump-to-end branches.
        result.append(
            BasicBlockNode(labels=[LabelInst(name=end_label, can_remove=True)])
        )

        return BlockNode(insts=result)
