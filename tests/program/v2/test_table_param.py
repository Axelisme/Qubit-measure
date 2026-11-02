"""Lookup parameter conversion at the QICK scalar and waveform seams."""

import operator
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from qick import QickConfig
from qick.asm_v2 import QickParam
from qick.helpers import to_int
from zcu_tools.program import v2


@pytest.mark.parametrize("quantize,trunc", [(1, False), (4, False), (4, True)])
def test_table_conversion_rounds_each_point_like_qick(
    mock_soccfg: QickConfig, quantize: int, trunc: bool
) -> None:
    values = [0.017, -0.129, 0.013, 0.017]
    scale = mock_soccfg.us2cycles(1.0)
    table = v2.TableParam("point", values)
    raw = to_int(table, scale, quantize=quantize, parname="length", trunc=trunc)
    expected = [
        to_int(value, scale, quantize=quantize, parname="length", trunc=trunc)
        for value in values
    ]
    assert isinstance(raw, v2.TableRawParam)
    assert raw.loop == "point"
    assert list(raw.values) == expected
    assert raw.start == expected[0]
    assert raw.minval() == min(expected)
    assert raw.maxval() == max(expected)
    np.testing.assert_array_equal(table.to_array(), values)
    np.testing.assert_allclose(
        table.get_rounded({"point": len(values)}).to_array(),
        np.asarray(expected) / scale,
    )


@pytest.mark.parametrize(
    "operation", [operator.add, operator.sub, operator.mul, operator.truediv]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_scalar_arithmetic_preserves_loop_and_rejects_sweep_mixing(
    operation: Callable[..., Any], reverse: bool
) -> None:
    values = [2.0, -4.0, 8.0, 2.0]
    table = v2.TableParam("point", values)
    result = operation(3.0, table) if reverse else operation(table, 3.0)
    expected = [operation(3.0, v) if reverse else operation(v, 3.0) for v in values]
    assert result.loop == "point"
    assert result.is_sweep()
    assert result.minval() == min(expected)
    assert result.maxval() == max(expected)
    np.testing.assert_allclose(result.to_array(), expected)
    np.testing.assert_array_equal(table.to_array(), values)
    assert table < 0
    assert table > 0
    assert table <= -4
    assert table >= 8
    assert not table < -4
    assert not table > 8
    with pytest.raises((TypeError, RuntimeError)):
        float(table)
    for other in [v2.TableParam("other", values), QickParam(2.0, {"other": 1.0})]:
        with pytest.raises(TypeError):
            operation(other, table) if reverse else operation(table, other)
