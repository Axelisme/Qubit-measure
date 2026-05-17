from .block_merge import BlockMergePass
from .branch_elimination import BranchEliminationPass
from .dead_label import DeadLabelEliminationPass
from .dmem_dispatch import DmemDispatchPass
from .simplify_dispatch import SimplifyDispatchPass
from .unpack_branch import UnpackIRBranchPass
from .unreachable import UnreachableEliminationPass

__all__ = [
    "BlockMergePass",
    "BranchEliminationPass",
    "DeadLabelEliminationPass",
    "DmemDispatchPass",
    "SimplifyDispatchPass",
    "UnpackIRBranchPass",
    "UnreachableEliminationPass",
]
