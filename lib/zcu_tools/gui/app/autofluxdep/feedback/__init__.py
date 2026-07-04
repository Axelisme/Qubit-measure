"""Generic scalar feedback capabilities for autofluxdep runs.

The framework owns reusable estimation/control mechanics and policy parsing.
Experiment nodes still own domain composition, gates, bounds, and Patch output.
"""

from .runtime import (
    FeedbackRuntime,
    FeedbackSlotDecl,
    FeedbackView,
    IdwEstimator,
    LastGoodEstimator,
    LogStepController,
    ScalarController,
    ScalarEstimator,
    build_feedback_runtime,
    feedback_generation_fields,
)

__all__ = [
    "FeedbackRuntime",
    "FeedbackSlotDecl",
    "FeedbackView",
    "IdwEstimator",
    "LastGoodEstimator",
    "LogStepController",
    "ScalarController",
    "ScalarEstimator",
    "build_feedback_runtime",
    "feedback_generation_fields",
]
