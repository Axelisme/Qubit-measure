from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast, get_args

import numpy as np
from pydantic import BaseModel

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.program.v2 import SweepCfg

T = TypeVar("T", bound=SweepCfg | list)


def make_sweep(
    start: int | float,
    stop: int | float | None = None,
    expts: int | None = None,
    step: int | float | None = None,
    *,
    force_int: bool = False,
) -> SweepCfg:
    """Build a regular sweep from endpoints, count, or increment.

    Numeric values use the caller's units. Provide stop and step to infer expts,
    or expts and stop to infer step. A single point (expts=1) may omit both.
    When expts and step are supplied, stop is ignored and recomputed.

    Count inference truncates (stop - start) / step + 1 with int(); the returned
    stop always equals start + step * (expts - 1). Negative steps are allowed.
    With force_int=True, infer missing values first, then truncate start, step,
    and expts toward zero before recomputing stop.

    Raise ValueError for insufficient inputs, nonpositive counts, a nonzero
    single-point step, a zero multi-point step, or SweepCfg validation failure.
    The returned SweepCfg is fresh; this function does not access hardware.
    """
    if expts is None:
        if stop is None or step is None:
            raise ValueError("Not enough information to define a sweep.")
        expts = _infer_sweep_count(start, stop, step)
    elif step is None:
        step = _infer_sweep_step(start, stop, expts)

    if force_int:
        start = int(start)
        step = int(step)
        expts = int(expts)

    if expts <= 0:
        raise ValueError(f"expts must be greater than 0, but got {expts}")
    if expts == 1 and step != 0:
        raise ValueError(f"for expts == 1, step must be 0, but got {step}")
    if expts > 1 and step == 0:
        raise ValueError(f"step must not be zero when expts > 1, but got {step}")

    stop = start + step * (expts - 1)
    return SweepCfg(start=start, stop=stop, expts=expts, step=step)


def _infer_sweep_count(start: int | float, stop: int | float, step: int | float) -> int:
    if step == 0:
        if stop != start:
            raise ValueError(
                f"stop must equal start when step is 0, got start={start}, stop={stop}"
            )
        return 1
    return int((stop - start) / step + 1)


def _infer_sweep_step(
    start: int | float, stop: int | float | None, expts: int
) -> int | float:
    if expts == 1:
        if stop is not None and stop != start:
            raise ValueError(
                f"for expts == 1, stop must equal start, got start={start}, stop={stop}"
            )
        return 0
    if stop is None:
        raise ValueError("Not enough information to define a sweep.")
    return (stop - start) / (expts - 1)


def unwrap_model_annotation(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation

    for arg in get_args(annotation):
        if (model := unwrap_model_annotation(arg)) is not None:
            return model

    return None


def get_single_sweep_name(cfg_model: type[ExpCfgModel]) -> str | None:
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


def format_sweep1D(sweep: Mapping[str, T] | T, name: str) -> dict[str, T]:
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

    elif isinstance(sweep, SweepCfg):
        return {name: cast(T, sweep)}

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
