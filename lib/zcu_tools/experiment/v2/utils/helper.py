from __future__ import annotations

import numpy as np
from typing_extensions import Sequence, TypeVar

from zcu_tools.experiment.v2.runner import Result

T_Result = TypeVar("T_Result", bound=Result)


def merge_result_list(results: Sequence[T_Result]) -> T_Result:
    assert isinstance(results, list) and len(results) > 0
    if isinstance(results[0], dict):
        return {
            name: merge_result_list([r[name] for r in results])  # type: ignore
            for name in results[0]
        }
    return np.asarray(results)  # type: ignore
