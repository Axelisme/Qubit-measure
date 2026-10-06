"""Labber container validation, independent of experiment Result models."""

import numpy as np
from numpy.typing import ArrayLike

from .models import LabberPayload
from .native_models import VariableSchema


def validate_labber_payload(
    payload: LabberPayload, *, schema: VariableSchema, context: str
) -> None:
    """Check labels/units, numeric containers and reversed inner-first shape.

    schema declares channel names, units and target dtype kinds. Labber stores
    float/complex containers, so exact NumPy width is not required; real targets
    reject nonzero imaginary values. schema.variable is the caller's domain
    identity, checked separately by the grouped reader's required_variables.
    context identifies the source/variable in ValueError. This function neither
    changes arrays nor converts units; integer coordinate semantics belong to
    the typed mapping. Malformed axes/shape/dtype raise ValueError.
    """
    if len(payload.axes) != len(schema.axes):
        raise ValueError(
            f"{context} has {len(payload.axes)} axes; expected {len(schema.axes)}"
        )
    lengths: list[int] = []
    for index, (actual, expected) in enumerate(
        zip(payload.axes, schema.axes, strict=True)
    ):
        label = f"{context} axis {index}"
        if actual.name != expected.name:
            raise ValueError(
                f"{label} label is {actual.name!r}; expected {expected.name!r}"
            )
        if actual.unit != expected.unit:
            raise ValueError(
                f"{label} unit is {actual.unit!r}; expected {expected.unit!r}"
            )
        values = np.asarray(actual.values)
        if values.ndim != 1:
            raise ValueError(f"{label} is {values.ndim}D; expected 1D")
        _validate_dtype(values, expected.dtype, context=label)
        lengths.append(len(values))
    if payload.data.name != schema.signal_name:
        raise ValueError(
            f"{context} z channel label is {payload.data.name!r}; "
            f"expected {schema.signal_name!r}"
        )
    if payload.data.unit != schema.signal_unit:
        raise ValueError(
            f"{context} z channel unit is {payload.data.unit!r}; "
            f"expected {schema.signal_unit!r}"
        )
    values = np.asarray(payload.z)
    shape = tuple(reversed(lengths))
    if values.shape != shape:
        raise ValueError(
            f"{context} z shape {values.shape} != expected {shape}; "
            "axis lengths must match z dimensions"
        )
    _validate_dtype(values, schema.signal_dtype, context=f"{context} z channel")


def cast_labber_values(
    values: ArrayLike, dtype: np.dtype[np.generic], *, context: str
) -> np.ndarray:
    """Return numeric values as dtype, preserving shape and the caller's units.

    values is the loaded numeric container (possibly already unit-converted).
    dtype is a numeric NumPy dtype. Real/integer targets accept a complex
    container only when every imaginary value is zero. Invalid dtype or nonzero
    imaginary values raise ValueError with context, the source/channel label.
    Integer casts follow NumPy conversion; typed mappings own rounding rules.
    This function does not mutate the input arrays.
    """
    array = np.asarray(values)
    _validate_dtype(array, dtype, context=context)
    if dtype.kind != "c" and np.iscomplexobj(array):
        array = np.real(array)
    return array.astype(dtype)


def _validate_dtype(
    values: np.ndarray, dtype: np.dtype[np.generic], *, context: str
) -> None:
    if dtype.kind not in "iufc" or values.dtype.kind not in "iufc":
        raise ValueError(f"{context} requires a numeric dtype")
    if dtype.kind != "c" and np.iscomplexobj(values) and np.any(np.imag(values) != 0.0):
        raise ValueError(
            f"{context} contains non-zero imaginary component; cannot load as {dtype}"
        )
