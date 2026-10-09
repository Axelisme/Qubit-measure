"""Complex coercion and complete context projection at the wire seam."""

import pytest
from zcu_tools.gui.app.measure.remote.handlers._wire_values import (
    coerce_wire_value,
    context_wire_value,
)


def test_context_projection_and_inbound_coercion_share_scalar_format() -> None:
    value = complex(0.012345678901234567, -0.03456789012345678)
    tag = {"__complex__": [value.real, value.imag]}
    assert context_wire_value({"center": value}) == {"center": tag}
    assert coerce_wire_value(tag) == value


@pytest.mark.parametrize(
    "value",
    [
        {"__complex__": (1, 2)},
        {"__complex__": [True, 2]},
        {"__complex__": [1]},
        {"__complex__": [1, 2], "extra": 3},
    ],
)
def test_inbound_malformed_tag_fails_without_fallback(value: object) -> None:
    with pytest.raises(ValueError, match="Complex"):
        coerce_wire_value(value)


@pytest.mark.parametrize("value", ["(1+2j)", [1, 2], {"other": [1, 2]}])
def test_ordinary_wire_values_are_not_guessed_as_complex(value: object) -> None:
    assert coerce_wire_value(value) == value


def test_context_complex_projection_keeps_finite_number_limit() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        context_wire_value(complex(1, float("inf")))
