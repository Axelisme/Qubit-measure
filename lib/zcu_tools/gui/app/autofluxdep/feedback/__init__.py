"""Generic scalar feedback capabilities for autofluxdep runs.

The framework owns reusable estimation/control mechanics and policy parsing.
Experiment nodes still own domain composition, gates, bounds, and Patch output.
"""

from .runtime import (
    FeedbackRuntime,
    FeedbackSample,
    FeedbackSlotDecl,
    FeedbackView,
    IdwEstimator,
    LastGoodEstimator,
    LogStepController,
    ScalarController,
    ScalarEstimator,
    build_feedback_runtime,
)

__all__ = [
    "FeedbackRuntime",
    "FeedbackSample",
    "FeedbackSlotDecl",
    "FeedbackView",
    "IdwEstimator",
    "LastGoodEstimator",
    "LogStepController",
    "ScalarController",
    "ScalarEstimator",
    "build_feedback_runtime",
]
