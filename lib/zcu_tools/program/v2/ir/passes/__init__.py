from .base import (
    BlockChunkPass,
    IRTransformer,
    _DATAFLOW_TRANSPARENT_INSTS,
)
from .control_flow import (
    BlockMergePass,
    BranchEliminationPass,
    DeadLabelEliminationPass,
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
    "_DATAFLOW_TRANSPARENT_INSTS",
    # control_flow
    "BlockMergePass",
    "BranchEliminationPass",
    "DeadLabelEliminationPass",
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
