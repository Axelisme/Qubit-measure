"""Migration pin: ticket 10 retires this file when legacy programs are removed."""

import numpy as np
from zcu_tools.program.v2 import ModularProgramV2, TimelineBuilder

from tests.program.v2._mux_support import mux_soccfg, native_mux


def test_legacy_and_timeline_collect_identical_average_and_raw_buffers(monkeypatch):
    old = native_mux().build("mux")
    legacy = ModularProgramV2(mux_soccfg(), [old], reps=3, sweep=[("point", 2)])
    new = native_mux().build("mux")
    builder = TimelineBuilder(mux_soccfg(), reps=3)
    builder.add_loop("point", 2)
    builder.emit(new)
    timeline = builder.build()
    acquired = [
        np.array([[[1.25, -2.5], [3.75, 4.5]]]),
        np.array([[[100.5, -200.25], [np.nan, 400.75]]]),
    ]
    np.testing.assert_array_equal(
        old.collect_iq(legacy, acquired), new.collect_iq(timeline, acquired)
    )
    sentinels = np.arange(12, dtype=np.int64).reshape(3, 2, 1, 2)
    raw = [sentinels * 3 + 1, sentinels * 7 - 9]
    monkeypatch.setattr(legacy, "get_raw", lambda: raw)
    monkeypatch.setattr(timeline, "get_raw", lambda: raw)
    np.testing.assert_array_equal(
        old.collect_raw_iq(legacy), new.collect_raw_iq(timeline)
    )
