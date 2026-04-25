from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from typing_extensions import Any, Mapping, Optional, TypeVar, Union, cast, get_args

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.program.v2 import SweepCfg

T = TypeVar("T", bound=Union[SweepCfg, list])


def unwrap_model_annotation(annotation: Any) -> Optional[type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation

    for arg in get_args(annotation):
        if (model := unwrap_model_annotation(arg)) is not None:
            return model

    return None


def get_single_sweep_name(cfg_model: type[ExpCfgModel]) -> Optional[str]:
    sweep_field = cfg_model.model_fields.get("sweep")
    if sweep_field is None:
        return None

    sweep_model = unwrap_model_annotation(sweep_field.annotation)
    if sweep_model is None:
        return None

    sweep_names = tuple(sweep_model.model_fields)
    if len(sweep_names) != 1:
        return None

    return sweep_names[0]


def format_sweep1D(sweep: Union[Mapping[str, T], T], name: str) -> dict[str, T]:
    """
    Convert abbreviated single sweep to regular format.

    This function takes a sweep parameter in different formats and converts it
    to a standardized dictionary format with a specified key name.

    Args:
        sweep: A dictionary containing sweep parameters (with 'start' and 'stop' keys)
               or a numpy array of values to sweep through
        name: Expected key name for the sweep in the returned dictionary

    Returns:
        A dictionary in regular format with 'name' as the key
    """

    if isinstance(sweep, np.ndarray) or isinstance(sweep, list):
        return {name: cast(T, np.asarray(sweep))}

    elif isinstance(sweep, dict):
        # conclude by key "start" and "stop"
        if "start" in sweep and "stop" in sweep:
            # it is in abbreviated format
            return {name: cast(T, sweep)}

        # check if only one sweep is provided
        assert len(sweep) == 1, "Only one sweep is allowed"
        assert sweep.get(name) is not None, f"Key {name} is not found in the sweep"

        # it is already in regular format
        return dict(sweep)
    else:
        raise ValueError(sweep)
