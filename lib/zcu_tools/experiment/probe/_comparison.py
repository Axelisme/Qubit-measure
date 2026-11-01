"""Result comparison shared by staged and legacy native roundtrips."""

from dataclasses import fields, is_dataclass
from typing import TypeVar

import numpy as np

from zcu_tools.cfg_model import ConfigBase

from ..axes_spec import AxesSpec, GroupedAxesSpec

_Cfg = TypeVar("_Cfg", bound=ConfigBase)
_Result = TypeVar("_Result")


def _scaled_component_equal(before: np.ndarray, after: np.ndarray) -> bool:
    if not np.array_equal(np.isnan(before), np.isnan(after)):
        return False
    finite = np.isfinite(before)
    if not np.array_equal(finite, np.isfinite(after)):
        return False
    infinite = np.isinf(before)
    if not np.array_equal(before[infinite], after[infinite]):
        return False
    # Widen before subtraction/division so small tolerance products cannot
    # underflow to a rounded absolute allowance for subnormal inputs.
    left = before[finite].astype(np.longdouble)
    right = after[finite].astype(np.longdouble)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        relative = np.abs(left - right) / np.abs(left)
    return bool(np.all((left == right) | (relative <= 2 * np.finfo(before.dtype).eps)))


def _scaled_fields(
    spec: AxesSpec[_Result, _Cfg] | GroupedAxesSpec[_Result, _Cfg],
) -> set[str]:
    """Select Result fields whose declared scale permits relative comparison."""
    declarations = (
        (*spec.axes, spec.z)
        if isinstance(spec, AxesSpec)
        else tuple(
            declaration
            for variable in spec.variables
            for declaration in (*variable.axes, variable.z)
        )
    )
    scales_by_field: dict[str, set[float]] = {}
    for declaration in declarations:
        if declaration.field_name is not None:
            scales_by_field.setdefault(declaration.field_name, set()).add(
                declaration.scale
            )
    if isinstance(spec, GroupedAxesSpec):
        for name, scales in scales_by_field.items():
            if len(scales) > 1:
                raise ValueError(
                    f"Conflicting scales for Result field {name!r}: {sorted(scales)}"
                )
    return {
        declaration.field_name
        for declaration in declarations
        if declaration.field_name is not None
        and np.isfinite(declaration.scale)
        and declaration.scale not in (0, 1)
    }


def compare_roundtrip(
    original: _Result,
    restored: _Result,
    *,
    spec: AxesSpec[_Result, _Cfg] | GroupedAxesSpec[_Result, _Cfg],
) -> bool:
    """Compare in-memory Results before save and after native restore.

    Both inputs must be dataclass instances of spec.result_type with the same
    fields; otherwise raise TypeError. Compare every field, including fields
    not named by the persistence declaration.
    Return False for any differing field shape or dtype. AxesSpec and
    GroupedAxesSpec fields with a finite, nonzero scale other than 1 allow
    relative error <= 2*eps, atol=0, for
    floating values (each real/imaginary component for complex arrays). eps is
    the actual real storage dtype's machine epsilon. NaN component positions
    must match and infinities must match sign. Zero has no absolute allowance.
    Fixed/generated axes without a Result field are excluded.
    Other fields and integer/discrete arrays remain exact,
    with NaNs equal under NumPy array_equal semantics.

    For round-to-nearest arithmetic with the same represented scale s, one
    multiplication and division give fl(fl(x*s)/s) = x*(1+d1)*(1+d2).
    If both operations stay in the normal finite range, |di| <= u = eps/2,
    hence relative error <= 2*u + u**2 < 2*eps. This assumes no additional
    lossy dtype conversion. Underflow, overflow and invalid scales do not have
    this guarantee; they receive no enlarged tolerance or repaired values.
    Conflicting scales for one Result field across grouped declarations raise
    ValueError naming the field and scales, even when both Results are identical.
    Other persistence validation and I/O remain the caller's responsibility.
    """
    if (
        not is_dataclass(original)
        or isinstance(original, type)
        or not isinstance(original, spec.result_type)
        or not is_dataclass(restored)
        or isinstance(restored, type)
        or not isinstance(restored, spec.result_type)
    ):
        raise TypeError("Roundtrip Results must match spec.result_type")
    if {item.name for item in fields(original)} != {
        item.name for item in fields(restored)
    }:
        raise TypeError("Roundtrip Results must have the same fields")
    scaled_fields = _scaled_fields(spec)
    for item in fields(original):
        before = np.asarray(getattr(original, item.name))
        after = np.asarray(getattr(restored, item.name))
        if before.shape != after.shape or before.dtype != after.dtype:
            return False
        if item.name in scaled_fields and before.dtype.kind in "fc":
            if not _scaled_component_equal(np.real(before), np.real(after)):
                return False
            if before.dtype.kind == "c" and not _scaled_component_equal(
                np.imag(before), np.imag(after)
            ):
                return False
        elif not np.array_equal(before, after, equal_nan=before.dtype.kind in "fc"):
            return False
    return True

