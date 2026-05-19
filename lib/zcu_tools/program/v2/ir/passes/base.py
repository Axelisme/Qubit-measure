from __future__ import annotations

from abc import ABC, abstractmethod

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
from ..node import BasicBlockNode
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
        self,
        chunks: ChunkList,
        ctx: PipeLineContext,  # noqa: ARG002
    ) -> tuple[ChunkList, bool]:
        changed = False
        for chunk in chunks:
            if isinstance(chunk, BasicBlockNode):
                changed |= self._process_block(chunk)
        return chunks, changed

    @abstractmethod
    def _process_block(self, block: BasicBlockNode) -> bool: ...
