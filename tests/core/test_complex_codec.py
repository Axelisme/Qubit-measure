"""Shared scalar format, precision and malformed-input contracts."""

import json

import numpy as np
import pytest
from zcu_tools.core.complex_codec import decode_complex, encode_complex


@pytest.mark.parametrize(
    "value",
    [
        complex(0.012345678901234567, -0.03456789012345678),
        complex(-3.25, 0.0),
        complex(0.0, -2.5),
        pytest.param(
            np.complex128(0.012345678901234567 - 0.03456789012345678j),
            id="numpy-complex128",
        ),
    ],
)
def test_complex_scalar_survives_json_without_rounding(value: complex) -> None:
    tag = encode_complex(value)
    assert [type(component) for component in tag["__complex__"]] == [float, float]
    assert tag == {"__complex__": [value.real, value.imag]}
    assert decode_complex(json.loads(json.dumps(tag))) == value


def test_integer_components_decode_as_complex() -> None:
    result = decode_complex({"__complex__": [3, -2]})
    assert result == 3 - 2j
    assert isinstance(result, complex)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "(1+2j)",
        [1, 2],
        {},
        {"other": [1, 2]},
        {"__complex__": [1, 2], "other": 3},
        {"__complex__": [1]},
        {"__complex__": [1, 2, 3]},
        {"__complex__": (1, 2)},
        {"__complex__": [True, 2]},
        {"__complex__": [1, False]},
        {"__complex__": ["1", 2]},
        {"__complex__": [1, None]},
    ],
)
def test_malformed_complex_scalar_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="Complex"):
        decode_complex(value)
