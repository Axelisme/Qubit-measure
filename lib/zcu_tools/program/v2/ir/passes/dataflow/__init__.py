from .dead_test import DeadTestEliminationPass
from .dead_write import DeadWriteEliminationPass
from .inc_reg_merge import IncRegMergePass

__all__ = [
    "DeadTestEliminationPass",
    "DeadWriteEliminationPass",
    "IncRegMergePass",
]
