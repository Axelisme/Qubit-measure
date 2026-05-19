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

from ...analysis import collect_referenced_labels
from ...node import BasicBlockNode
from ...pipeline import AbsChunkListPass, ChunkList, PipeLineContext


class DeadLabelEliminationPass(AbsChunkListPass):
    """Remove labels that are never referenced by any instruction."""

    def process(
        self,
        chunks: ChunkList,
        ctx: PipeLineContext,  # noqa: ARG002
    ) -> tuple[ChunkList, bool]:
        referenced = collect_referenced_labels(chunks)
        changed = False
        for chunk in chunks:
            if not isinstance(chunk, BasicBlockNode):
                continue
            before = len(chunk.labels)
            chunk.labels = [
                lbl
                for lbl in chunk.labels
                if not lbl.can_remove or lbl.name in referenced
            ]
            changed |= len(chunk.labels) != before
        return chunks, changed
