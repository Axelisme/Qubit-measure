from .base import OptimizationPassBase
from .control_flow import (
    BlockMergePass,
    BranchEliminationPass,
    DeadLabelEliminationPass,
)
from .dataflow import DeadTestEliminationLinear, DeadWriteEliminationLinear
from .loop import UnrollSmallLoopPass
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
    # loop
    "UnrollSmallLoopPass",
    # timeline
    "TimedMergeLinear",
    "ZeroDelayDCELinear",
]
