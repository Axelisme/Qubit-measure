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
from ..node import BasicBlockNode, BlockNode, IRBranch, IRLoop, IRNode, RootNode
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
    """Base class for IR tree transformations using explicit per-node visitors.

    Subclasses override ``visit_<ClassName>`` methods to transform specific
    node types.  Default visitors recurse into child nodes automatically.

    ``_changed`` is set whenever a structural change (insert / remove / replace)
    is made during traversal.  Reset it to ``False`` before a top-level call
    and inspect it afterwards.
    """

    _changed: bool = False

    def visit(self, node: IRNode) -> Union[IRNode, List[IRNode], None]:
        """Dispatch to ``visit_<ClassName>`` or the default handler."""
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self._unhandled_visit)
        return visitor(node)

    def _unhandled_visit(self, node: IRNode) -> IRNode:
        """Fallback for unknown IRNode types — return unchanged."""
        return node

    # -- Default visitors (recurse into children) --

    def visit_BasicBlockNode(
        self, node: BasicBlockNode
    ) -> Union[IRNode, List[IRNode], None]:
        """Leaf node: no IRNode children to recurse into."""
        return node

    def visit_BlockNode(self, node: BlockNode) -> Union[IRNode, List[IRNode], None]:
        return self._visit_block(node)

    def visit_RootNode(self, node: RootNode) -> Union[IRNode, List[IRNode], None]:
        return self._visit_block(node)

    def visit_IRLoop(self, node: IRLoop) -> Union[IRNode, List[IRNode], None]:
        res = self.visit(node.body)
        if res is None:
            self._changed = True
            node.body = BlockNode()
        elif isinstance(res, list):
            self._changed = True
            node.body = BlockNode(insts=res)
        else:
            if res is not node.body:
                self._changed = True
            node.body = res  # type: ignore[assignment]
        return node

    def visit_IRBranch(self, node: IRBranch) -> Union[IRNode, List[IRNode], None]:
        new_cases: list[BlockNode] = []
        for case in node.cases:
            res = self.visit(case)
            if res is None:
                self._changed = True
                new_cases.append(BlockNode())
            elif isinstance(res, list):
                self._changed = True
                new_cases.append(BlockNode(insts=res))
            else:
                if res is not case:
                    self._changed = True
                new_cases.append(res)  # type: ignore[arg-type]
        node.cases = new_cases
        return node

    def _visit_block(self, node: BlockNode) -> Union[IRNode, List[IRNode], None]:
        new_insts: list[IRNode] = []
        list_changed = False
        for item in node.insts:
            res = self.visit(item)
            if res is None:
                list_changed = True
            elif isinstance(res, list):
                list_changed = True
                new_insts.extend(res)
            else:
                if res is not item:
                    list_changed = True
                new_insts.append(res)
        if list_changed:
            self._changed = True
            node.insts = new_insts
        return node
