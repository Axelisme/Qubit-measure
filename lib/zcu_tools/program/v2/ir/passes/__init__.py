from ..pipeline import AbsChunkListPass, AbsIRTreePass
from .base import (
    DATAFLOW_TRANSPARENT_INSTS,
    BlockChunkPass,
)
from .control_flow import (
    BlockMergePass,
    BranchEliminationPass,
    DeadLabelEliminationPass,
    SimplifyDispatchPass,
    UnreachableEliminationPass,
)
from .dataflow import (
    DeadTestEliminationPass,
    DeadWriteEliminationPass,
    IncRegMergePass,
)
from .loop import LoopConditionMergePass, UnrollLoopPass
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
    "SimplifyDispatchPass",
    "UnreachableEliminationPass",
    # dataflow
    "DeadTestEliminationPass",
    "DeadWriteEliminationPass",
    "IncRegMergePass",
    # loop
    "UnrollLoopPass",
    "LoopConditionMergePass",
    # timeline
    "TimedMergePass",
    "ZeroDelayDCEPass",
]
