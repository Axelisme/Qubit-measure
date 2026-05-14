from __future__ import annotations

from abc import ABC, abstractmethod

from typing_extensions import List, Union

from ..instructions import (
    ArithInst,
    BaseInst,
    CallInst,
    ClearInst,
    ComInst,
    CustomPeripheralInst,
    DivInst,
    DmemReadInst,
    DmemWriteInst,
    DportReadInst,
    FlagInst,
    NetInst,
    NopInst,
    PortWriteInst,
    RegWriteInst,
    RetInst,
    TimeInst,
    TrigInst,
    WaitInst,
    WmemWriteInst,
)
from ..node import BasicBlockNode, BlockNode, IRBranch, IRDispatch, IRLoop, IRNode
from ..pipeline import AbsChunkPass, ChunkList, PipeLineContext

# ---------------------------------------------------------------------------
# Shared dataflow transparency list (R1)
# ---------------------------------------------------------------------------

# Instructions that are transparent to dataflow tracking: passes can freely
# sink/hoist/reorder across these without flushing pending state.
# Excluded (barrier by omission): JumpInst, TestInst, LabelInst, DportWriteInst.
DATAFLOW_TRANSPARENT_INSTS: tuple[type[BaseInst], ...] = (
    TimeInst,
    WaitInst,
    RegWriteInst,
    DmemReadInst,
    DmemWriteInst,
    PortWriteInst,
    WmemWriteInst,
    NopInst,
    ArithInst,
    CallInst,
    ClearInst,
    ComInst,
    CustomPeripheralInst,
    DivInst,
    DportReadInst,
    FlagInst,
    NetInst,
    RetInst,
    TrigInst,
)

# ---------------------------------------------------------------------------
# BlockChunkPass (R2)
# ---------------------------------------------------------------------------


class BlockChunkPass(AbsChunkPass, ABC):
    """AbsChunkPass that iterates over BasicBlockNode chunks only.

    Subclasses implement ``_process_block`` and get the per-chunk loop for free.
    ``UnreachableEliminationPass`` uses a stateful linear scan instead and does
    not inherit from this class.
    """

    def process(
        self, chunks: ChunkList, ctx: PipeLineContext
    ) -> tuple[ChunkList, bool]:
        changed = False
        for chunk in chunks:
            if isinstance(chunk, BasicBlockNode):
                changed |= self._process_block(chunk)
        return chunks, changed

    @abstractmethod
    def _process_block(self, block: BasicBlockNode) -> bool: ...


# ---------------------------------------------------------------------------
# IRTransformer (R3-B)
# ---------------------------------------------------------------------------


class IRTransformer:
    """Base class for IR tree transformations using the children()/replace_child() interface.

    Subclasses override ``visit_<ClassName>`` methods to transform specific
    node types.  The default behaviour recurses into every child automatically
    via ``IRNode.children()`` and ``IRNode.replace_child()``, so overrides only
    need to handle the node itself — not its descendants.

    Traversal order: pre-order on the *result* of each visitor.  If
    ``visit_Foo`` returns a new node, that new node's children are still
    visited.  This ensures that a visitor that expands one node type into a
    subtree containing further transformable nodes is handled in a single pass.

    ``_changed`` is set whenever a child is replaced during traversal.
    Reset it to ``False`` before a top-level call and inspect it afterwards.
    """

    _changed: bool = False

    def visit(self, node: IRNode) -> IRNode:
        """Dispatch to ``visit_<ClassName>`` then recurse into result's children."""
        method = getattr(self, f"visit_{type(node).__name__}", None)
        result = method(node) if method is not None else node

        # Recurse into children of the result (not the original node).
        # Leaf nodes (BasicBlockNode, IRDispatch) return [] so the loop is a no-op.
        for child in list(result.children()):
            new_child = self.visit(child)
            if new_child is not child:
                result.replace_child(child, new_child)
                self._changed = True

        return result
