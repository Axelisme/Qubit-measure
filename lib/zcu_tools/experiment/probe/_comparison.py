"""Result comparison for native roundtrips."""

import logging
from dataclasses import fields, is_dataclass
from typing import TypeVar

import numpy as np

from zcu_tools.cfg_model import ConfigBase

from ..axes_spec import AxesSpec, GroupedAxesSpec, VariableZSpec

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


def _comparison_difference(
    before: np.ndarray, after: np.ndarray, *, scale: float
) -> str | None:
    """Describe a slice mismatch, or return None under its declared scale."""
    if before.shape != after.shape:
        return f"shape {before.shape} != {after.shape}"
    if before.dtype != after.dtype:
        return f"dtype {before.dtype} != {after.dtype}"
    if np.isfinite(scale) and scale not in (0, 1) and before.dtype.kind in "fc":
        equal = _scaled_component_equal(np.real(before), np.real(after))
        if before.dtype.kind == "c":
            equal &= _scaled_component_equal(np.imag(before), np.imag(after))
    else:
        equal = np.array_equal(before, after, equal_nan=before.dtype.kind in "fc")
    return None if equal else "values differ"


def _compare_field(
    name: str,
    before: np.ndarray,
    after: np.ndarray,
    comparisons: list[tuple[float, int | None, int]],
) -> bool:
    """Check each declared region and the exact, undeclared remainder."""
    logger = logging.getLogger("zcu_tools.experiment.probe")
    if before.shape != after.shape:
        logger.warning(
            "Roundtrip field=%s index_axis=None index=None: shape %s != %s",
            name,
            before.shape,
            after.shape,
        )
        return False
    covered = np.zeros(before.shape, dtype=bool)
    equal = True
    for scale, index, index_axis in comparisons or [(1.0, None, -1)]:
        selection: list[slice | int] = [slice(None)] * before.ndim
        if index is not None:
            selection[index_axis] = index
        region = tuple(selection)
        covered[region] = True
        difference = _comparison_difference(before[region], after[region], scale=scale)
        if difference is not None:
            logger.warning(
                "Roundtrip field=%s index_axis=%s index=%s: %s",
                name,
                index_axis,
                index,
                difference,
            )
            equal = False
    if not np.all(covered):
        difference = _comparison_difference(before[~covered], after[~covered], scale=1)
        if difference is not None:
            logger.warning(
                "Roundtrip field=%s index_axis=None index=None (undeclared): %s",
                name,
                difference,
            )
            equal = False
    return equal


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
    Grouped declarations compare each selected index on its index_axis with
    its own scale. Overlapping regions must satisfy every declaration; unselected
    regions remain exact. Return a bool.
    Log each mismatching slice at WARNING through "zcu_tools.experiment.probe", naming
    field, index_axis, index and the shape, dtype or value difference. Matching
    slices produce no warning. Continue comparing other slices after a mismatch.
    A whole-field shape mismatch is logged with index_axis=None and index=None.
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
    declarations = (
        (*spec.axes, spec.z)
        if isinstance(spec, AxesSpec)
        else tuple(
            declaration
            for variable in spec.variables
            for declaration in (*variable.axes, variable.z)
        )
    )
    equal = True
    for item in fields(original):
        comparisons = [
            (
                declaration.scale,
                declaration.index if isinstance(declaration, VariableZSpec) else None,
                declaration.index_axis
                if isinstance(declaration, VariableZSpec)
                else -1,
            )
            for declaration in declarations
            if declaration.field_name == item.name
        ]
        field_equal = _compare_field(
            item.name,
            np.asarray(getattr(original, item.name)),
            np.asarray(getattr(restored, item.name)),
            comparisons,
        )
        equal &= field_equal
    return equal
