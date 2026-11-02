"""Lookup parameter conversion at the QICK scalar and waveform seams."""

import numpy as np
import pytest
from qick import QickConfig
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
