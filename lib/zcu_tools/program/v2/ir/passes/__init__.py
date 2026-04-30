from .dce import LabelDCEPass
from .validation import IRStructureValidationPass, LabelReferenceValidationPass

__all__ = [
    "IRStructureValidationPass",
    "LabelDCEPass",
    "LabelReferenceValidationPass",
]
