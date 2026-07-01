from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

Result: TypeAlias = Sequence["Result"] | Mapping[Any, "Result"] | NDArray[Any]

T_Result = TypeVar("T_Result", bound=Result)


def merge_result_list(results: Sequence[T_Result]) -> T_Result:
    assert isinstance(results, list) and len(results) > 0
    if isinstance(results[0], dict):
        return {
            name: merge_result_list([r[name] for r in results])  # type: ignore
            for name in results[0]
        }
    return np.asarray(results)  # type: ignore
