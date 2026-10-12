"""Stateless complex scalar format shared by GUI and resource boundaries."""

from typing import TypedDict


class ComplexTag(TypedDict):
    """Explicit scalar tag, not a string or an untagged coordinate pair.

    __complex__ contains exactly two Python floats, real then imaginary, in
    the caller's working units. Encoding uses normal Python float rounding.
    Finite-number validation belongs to the consuming schema.
    """

    __complex__: list[float]


def encode_complex(value: complex) -> ComplexTag:
    """Return a fresh tag of Python float components for one native complex.

    Both components retain the scalar's working units and use normal Python
    float rounding. Finite-number validation belongs to the consuming schema.
    This function performs no I/O and does not alter value.
    """
    return {"__complex__": [float(value.real), float(value.imag)]}


def decode_complex(value: object) -> complex:
    """Read exactly {"__complex__": [real, imaginary]} as a native complex.

    The object must be a dict with only that key and a two-item list of int
    or float components, excluding bool. Strings, tuples and extra keys raise
    ValueError. Oversized integers that cannot become complex components raise
    OverflowError. Finite-number restrictions belong to the consuming schema.
    This function performs no I/O and does not alter value.
    """
    if not isinstance(value, dict) or set(value) != {"__complex__"}:
        raise ValueError("Complex tag must contain exactly the __complex__ key")
    components = value["__complex__"]
    if not isinstance(components, list) or len(components) != 2:
        raise ValueError("Complex components must be a two-item list")
    real, imaginary = components
    if (
        isinstance(real, bool)
        or isinstance(imaginary, bool)
        or not isinstance(real, (int, float))
        or not isinstance(imaginary, (int, float))
    ):
        raise ValueError("Complex components must be numeric, excluding bool")
    return complex(real, imaginary)
