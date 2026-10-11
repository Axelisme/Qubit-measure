"""Closed set of independent qubit models that front ends can select by name."""

from __future__ import annotations

from typing import Literal, TypeAlias

from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulate.predictor import PredictorInputError
from zcu_tools.simulate.transmon import TransmonParams

ModelName: TypeAlias = Literal["fluxonium", "transmon"]
ModelParams: TypeAlias = FluxoniumParams | TransmonParams

__all__ = ["ModelName", "ModelParams", "params_class"]


def params_class(model: str) -> type[ModelParams]:
    """Return the nullable parameter type for a model name; unknown names raise PredictorInputError."""
    if model == "fluxonium":
        return FluxoniumParams
    if model == "transmon":
        return TransmonParams
    raise PredictorInputError(f"Unknown predictor model: {model}")
