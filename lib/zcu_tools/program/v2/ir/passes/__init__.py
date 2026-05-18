from ..pipeline import AbsChunkListPass, AbsIRTreePass
from .base import (
    DATAFLOW_TRANSPARENT_INSTS,
    BlockChunkPass,
)
from .control_flow import (
    BlockMergePass,
    BranchEliminationPass,
    DeadLabelEliminationPass,
    DmemDispatchPass,
    SimplifyDispatchPass,
    UnpackIRBranchPass,
    UnreachableEliminationPass,
)
from .dataflow import (
    DeadTestEliminationPass,
    DeadWriteEliminationPass,
    IncRegMergePass,
)
from .loop import UnrollLoopPass
from .timeline import TimedMergePass, ZeroDelayDCEPass

__all__ = [
    # base
    "BlockChunkPass",
    "AbsChunkListPass",
    "AbsIRTreePass",
    "DATAFLOW_TRANSPARENT_INSTS",
    # control_flow
    "BlockMergePass",
    "BranchEliminationPass",
    "DeadLabelEliminationPass",
    "DmemDispatchPass",
    "SimplifyDispatchPass",
    "UnpackIRBranchPass",
    "UnreachableEliminationPass",
    # dataflow
    "DeadTestEliminationPass",
    "DeadWriteEliminationPass",
    "IncRegMergePass",
    # loop
    "UnrollLoopPass",
    # timeline
    "TimedMergePass",
    "ZeroDelayDCEPass",
]
