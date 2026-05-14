from .base import (
    DATAFLOW_TRANSPARENT_INSTS,
    BlockChunkPass,
    IRTransformer,
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
    "IRTransformer",
    "BlockChunkPass",
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
