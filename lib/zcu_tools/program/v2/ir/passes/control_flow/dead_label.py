"""DeadLabelEliminationPass: remove labels that are never referenced.

Purpose
-------
After optimizations remove jumps or restructure control flow, some labels
become orphaned — no instruction refers to them. This pass collects all
labels that appear in ``need_label`` fields across all chunks, then drops
any ``BasicBlockNode.labels`` entry that is not in that set.

QICK Hardware Notes
-------------------
- Pseudo labels (``HERE`` / ``NEXT``) are hardware-reserved jump targets used
  by the QICK assembler to encode short relative offsets.  They must never be
  removed even if no IR instruction references them explicitly.
- Labels with ``can_remove=False`` are anchoring markers (e.g. loop entry
  points, dispatch-table entries) that must survive independently of whether
  any jump currently targets them.

Decision Notes
--------------
Labels live only in ``BasicBlockNode.labels`` — ``BasicBlockNode.__post_init__``
forbids ``LabelInst`` inside ``insts``.  A single linear scan over all chunks
is sufficient to collect referenced labels and filter dead ones.
"""

from __future__ import annotations

from ...instructions import BaseInst
from ...labels import Label
from ...node import BasicBlockNode
from ...pipeline import ChunkList, PipeLineContext
from ..base import BlockChunkPass


def _collect_referenced_labels(chunks: ChunkList) -> set[Label]:
    refs: set[Label] = set()
    for chunk in chunks:
        if not isinstance(chunk, BasicBlockNode):
            continue
        for inst in (*chunk.labels, *chunk.insts, *(
            [chunk.branch] if chunk.branch else []
        )):
            if isinstance(inst, BaseInst) and inst.need_label is not None:
                refs.add(inst.need_label)
    return refs


class DeadLabelEliminationPass(BlockChunkPass):
    """Remove labels that are never referenced by any instruction."""

    def process(self, chunks: ChunkList, ctx: PipeLineContext) -> tuple[ChunkList, bool]:
        self._referenced = _collect_referenced_labels(chunks)
        return super().process(chunks, ctx)

    def _process_block(self, block: BasicBlockNode) -> bool:
        before = len(block.labels)
        block.labels = [
            lbl
            for lbl in block.labels
            if lbl.name.is_pseudo_name()
            or not lbl.can_remove
            or lbl.name in self._referenced
        ]
        return len(block.labels) != before
