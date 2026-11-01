"""Mux collection through compiled timeline emissions and public QICK buffers."""

import numpy as np
import pytest
from zcu_tools.program.v2 import TimelineBuilder, TimelineProgramV2

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


@pytest.mark.parametrize("emitted", [False, True])
def test_collection_requires_identity_emitted_in_nested_control(emitted, monkeypatch):
    module = native_mux().build("mux")
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("choice", 2)
            self.declare(module)

        def _body(self, cfg):
            with self.branch(by="choice", align_timeline=True) as cases:
                with cases.case(0):
                    pass
                with cases.case(1), self.repeat("reads", 2):
                    if emitted:
                        self.emit(module)

    prog = Program(mux_soccfg())
    acquired = [np.array([[[2.0, 3.0], [5.0, 7.0]]])] * 2
    raw = [
        np.full((1, 2, 1, 2), 11 * channel["length"], dtype=np.int64)
        for channel in prog.ro_chs.values()
    ]
    monkeypatch.setattr(prog, "get_raw", lambda: raw)
    if not emitted:
        with pytest.raises(ValueError, match="no compiled readout emission"):
            module.collect_iq(prog, acquired)
        with pytest.raises(ValueError, match="no compiled readout emission"):
            module.collect_raw_iq(prog)
    else:
        np.testing.assert_array_equal(
            module.collect_iq(prog, acquired),
            [[[2, 3], [2, 3]], [[5, 7], [5, 7]]],
        )
        np.testing.assert_array_equal(
            module.collect_raw_iq(prog), np.full((1, 2, 1, 2, 2), 11)
        )


def test_raw_collection_keeps_shots_reads_tones_and_offsets(monkeypatch):
    module = native_mux().build("mux")
    builder = TimelineBuilder(mux_soccfg(), reps=3)
    builder.add_loop("point", 2)
    builder.emit(module)
    builder.emit(module)
    prog = builder.build()
    sentinels = np.arange(24, dtype=np.int64).reshape(3, 2, 2, 2)
    raw = [
        sentinels * prog.ro_chs[4]["length"],
        (sentinels + 1000) * prog.ro_chs[1]["length"],
    ]
    monkeypatch.setattr(prog, "get_raw", lambda: raw)
    iq = module.collect_raw_iq(prog)
    assert iq.shape == (3, 2, 2, 2, 2)
    np.testing.assert_array_equal(iq[..., 0, :], sentinels)
    np.testing.assert_array_equal(iq[..., 1, :], sentinels + 1000)
    np.testing.assert_array_equal(
        module.collect_raw_iq(prog, read_index=1), iq[..., 1, :, :]
    )
    with pytest.raises(ValueError, match="mux"):
        native_mux().build("mux").collect_raw_iq(prog)
