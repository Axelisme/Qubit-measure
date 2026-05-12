from .block_merge import BlockMergePass
from .branch_elimination import BranchEliminationPass
from .dead_label import DeadLabelEliminationPass
from .unreachable import UnreachableEliminationPass

__all__ = [
    "BlockMergePass",
    "BranchEliminationPass",
    "DeadLabelEliminationPass",
    "UnreachableEliminationPass",
]
