from .base import OptimizationPassBase
from .control_flow import (
    BlockMergePass,
    BranchEliminationPass,
    DeadLabelEliminationPass,
)
from .dataflow import (
    DeadTestEliminationLinear,
    DeadWriteEliminationLinear,
    IncRegMergeLinear,
)
from .loop import UnrollLoopPass
from .loop_merge import LoopConditionMergeLinear
from .timeline import TimedMergeLinear, ZeroDelayDCELinear

__all__ = [
    # base
    "OptimizationPassBase",
    # control_flow
    "BlockMergePass",
    "BranchEliminationPass",
    "DeadLabelEliminationPass",
    # dataflow
    "DeadTestEliminationLinear",
    "DeadWriteEliminationLinear",
    "IncRegMergeLinear",
    # loop
    "UnrollLoopPass",
    "LoopConditionMergeLinear",
    # timeline
    "TimedMergeLinear",
    "ZeroDelayDCELinear",
]
