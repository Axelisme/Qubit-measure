from .base import (
    LinearPassAdapter,
    OptimizationPassBase,
    block_contains_structural_node,
)
from .control_flow import (
    BlockMergePass,
    BranchEliminationPass,
    DeadLabelEliminationPass,
)
from .dataflow import (
    DeadWriteEliminationLegacyPass,
    DeadWriteEliminationLinear,
    DeadWriteEliminationPass,
)
from .loop import UnrollSmallLoopPass
from .timeline import (
    TimedMergeLegacyPass,
    TimedMergeLinear,
    TimedMergePass,
    ZeroDelayDCELegacyPass,
    ZeroDelayDCELinear,
    ZeroDelayDCEPass,
)

__all__ = [
    # base
    "LinearPassAdapter",
    "OptimizationPassBase",
    "block_contains_structural_node",
    # control_flow
    "BlockMergePass",
    "BranchEliminationPass",
    "DeadLabelEliminationPass",
    # dataflow
    "DeadWriteEliminationLegacyPass",
    "DeadWriteEliminationLinear",
    "DeadWriteEliminationPass",
    # loop
    "UnrollSmallLoopPass",
    # timeline
    "TimedMergeLegacyPass",
    "TimedMergeLinear",
    "TimedMergePass",
    "ZeroDelayDCELegacyPass",
    "ZeroDelayDCELinear",
    "ZeroDelayDCEPass",
]
