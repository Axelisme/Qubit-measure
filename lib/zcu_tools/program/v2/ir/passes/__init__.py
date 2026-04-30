from .branch import BranchCaseNormalizePass
from .dce import LabelDCEPass
from .loop import ConstantLoopUnrollPass
from .optimize import LoopInvariantHoistPass, PeepholePass
from .timeline import TimedInstructionMergePass, ZeroDelayDCEPass
from .timing import TimingSanityPass
from .validation import IRStructureValidationPass, LabelReferenceValidationPass

__all__ = [
    "BranchCaseNormalizePass",
    "ConstantLoopUnrollPass",
    "IRStructureValidationPass",
    "LabelDCEPass",
    "LabelReferenceValidationPass",
    "LoopInvariantHoistPass",
    "PeepholePass",
    "TimedInstructionMergePass",
    "TimingSanityPass",
    "ZeroDelayDCEPass",
]
