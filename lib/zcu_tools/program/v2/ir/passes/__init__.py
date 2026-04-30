from .branch import BranchCaseNormalizePass
from .dce import LabelDCEPass
from .loop import ConstantLoopUnrollPass
from .optimize import LoopInvariantHoistPass, PeepholePass
from .timing import TimingSanityPass
from .timeline import ZeroDelayDCEPass
from .validation import IRStructureValidationPass, LabelReferenceValidationPass

__all__ = [
    "BranchCaseNormalizePass",
    "ConstantLoopUnrollPass",
    "IRStructureValidationPass",
    "LabelDCEPass",
    "LabelReferenceValidationPass",
    "LoopInvariantHoistPass",
    "PeepholePass",
    "TimingSanityPass",
    "ZeroDelayDCEPass",
]
