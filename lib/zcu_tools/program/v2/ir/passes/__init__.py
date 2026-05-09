from .base import (
    IRTransformer,
    OptimizationPassBase,
    walk_basic_blocks,
    walk_instructions,
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
from .loop import UnrollLoopPass
from .loop_merge import LoopConditionMergePass
from .timeline import TimedMergePass, ZeroDelayDCEPass

__all__ = [
    # base
    "OptimizationPassBase",
    "IRTransformer",
    "walk_basic_blocks",
    "walk_instructions",
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
