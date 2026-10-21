from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

Result: TypeAlias = Sequence["Result"] | Mapping[Any, "Result"] | NDArray[Any]


def merge_result_list(results: Sequence[Result]) -> Result:
    """Stack a nonempty list of like-structured results along a new first axis.

    Dict nodes are rebuilt in the first row's key order; leaves and sequence
    nodes are stacked with ``np.asarray``, retaining NumPy's dtype/shape rules.
    The output is a dict tree or ndarray, not the input's concrete container
    type. Rows must share keys and stack-compatible leaf shapes.
    """
    assert isinstance(results, list) and len(results) > 0
    if isinstance(results[0], dict):
        return {
            name: merge_result_list([r[name] for r in results]) for name in results[0]
        }
    return np.asarray(results)
