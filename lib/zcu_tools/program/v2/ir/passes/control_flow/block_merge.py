"""BlockMergePass: merge adjacent BasicBlockNodes when safe.

Purpose
-------
Adjacent BasicBlockNodes with no labels on the second block and no branch on
the first can be merged into a single block.  Fewer blocks means fewer
boundary constraints for downstream passes (e.g., IncRegMergePass can sink
increments across what used to be block boundaries).

Example
-------
Before::

    block_a:               ; no branch (falls through)
      REG_WR r0 imm #1

    block_b:               ; no labels → not a jump target
      TIME inc_ref #50

After::

    block_a:
      REG_WR r0 imm #1
      TIME inc_ref #50

QICK Hardware Notes
-------------------
- A block with ``disable_opt=True`` is a dispatch-table stub whose pmem
  word count is fixed by the jump-table encoding.  Such blocks must not be
  merged with neighbours even if the structural conditions are met.
- A block that has any alive label (referenced by a jump, not pseudo) is a
  potential branch target; merging it would shift its start address and break
  any jump pointing there.

Decision Notes
--------------
Runs at the chunk layer (after ``IRParser.unparse()``): at that point all
``IRLoop``/``IRBranch`` bodies are already flattened into the same chunk list,
so a linear scan is equivalent to the previous recursive tree walk.  MetaInst
boundaries between structural regions are skipped automatically because only
``BasicBlockNode`` pairs are considered for merging.

Merging is done with an inner fixed-point loop so that a chain
``A → B → C → D`` (all mergeable) collapses to one block in a single pass
invocation rather than requiring O(n) pipeline iterations.
"""

from __future__ import annotations

from ...instructions import BaseInst
from ...labels import Label
from ...node import BasicBlockNode
from ...pipeline import AbsChunkPass, ChunkList, PipeLineContext


def _collect_referenced_labels(chunks: ChunkList) -> set[Label]:
    """Collect all labels referenced by any instruction in the chunk list."""
    refs: set[Label] = set()
    for chunk in chunks:
        if not isinstance(chunk, BasicBlockNode):
            continue
        for inst in (
            *chunk.labels,
            *chunk.insts,
            *([chunk.branch] if chunk.branch else []),
        ):
            if isinstance(inst, BaseInst) and inst.need_label is not None:
                refs.add(inst.need_label)
    return refs


class BlockMergePass(AbsChunkPass):
    """Merge adjacent BasicBlockNodes when safe.

    Block A and Block B can be merged when:
      - Block A has no branch (falls through).
      - Block B has no alive labels (not a jump target).
    """

    def process(
        self, chunks: ChunkList, ctx: PipeLineContext
    ) -> tuple[ChunkList, bool]:
        referenced = _collect_referenced_labels(chunks)
        changed_any = False
        changed = True
        while changed:
            changed = self._merge_pass(chunks, referenced)
            changed_any |= changed
        return chunks, changed_any

    def _merge_pass(self, chunks: ChunkList, referenced: set[Label]) -> bool:
        i = 0
        changed = False
        while i < len(chunks) - 1:
            a = chunks[i]
            b = chunks[i + 1]
            if (
                isinstance(a, BasicBlockNode)
                and isinstance(b, BasicBlockNode)
                and a.branch is None
                and not _has_alive_labels(b, referenced)
                and not a.disable_opt
                and not b.disable_opt
            ):
                a.insts.extend(b.insts)
                a.branch = b.branch
                del chunks[i + 1]
                changed = True
            else:
                i += 1
        return changed


def _has_alive_labels(block: BasicBlockNode, referenced: set[Label]) -> bool:
    return any(
        lbl.name in referenced and not lbl.name.is_pseudo_name() for lbl in block.labels
    )
