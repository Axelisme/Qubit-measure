"""Mux collection through compiled timeline emissions and public QICK buffers."""

import numpy as np
import pytest
from zcu_tools.program.v2 import TimelineBuilder

from tests.program.v2._mux_support import mux_soccfg, native_mux


def test_averaged_collection_preserves_read_axes_tones_and_nan_after_module_reuse():
    module = native_mux().build("mux")
    builder = TimelineBuilder(mux_soccfg())
    builder.add_loop("point", 3)
    builder.emit(module)
    builder.emit(module)
    prog = builder.build()
    other = TimelineBuilder(mux_soccfg())
    other.emit(module)
    other.build()
    acquired = [
        np.arange(12, dtype=np.float64).reshape(2, 3, 2),
        100 + np.arange(12, dtype=np.float64).reshape(2, 3, 2),
    ]
    acquired[0][1, 0, 0] = np.nan
    iq = module.collect_iq(prog, acquired, read_index=1)
    np.testing.assert_array_equal(iq[:, 0], acquired[0][1])
    np.testing.assert_array_equal(iq[:, 1], acquired[1][1])
    assert iq.shape == (3, 2, 2)
    with pytest.raises(ValueError, match="mux"):
        native_mux().build("mux").collect_iq(prog, acquired)
