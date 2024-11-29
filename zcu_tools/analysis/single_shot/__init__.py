from typing import Literal

from .base import fidelity_func
from .center import fit_by_center
from .regression import fit_by_regression


def singleshot_analysis(
    Is, Qs, plot=True, backend: Literal["center", "regression"] = "regression"
):
    if backend == "center":
        return fit_by_center(Is, Qs, plot=plot)
    elif backend == "regression":
        return fit_by_regression(Is, Qs, plot=plot)
    else:
        raise ValueError(f"Unknown backend: {backend}")


__all__ = ["singleshot_analysis", "fidelity_func"]
