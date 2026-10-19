"""Config-to-Labber snapshot mapping at the public experiment seam."""

import numpy as np
import pytest
from numpy.typing import NDArray
from qick.asm_v2 import QickParam
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.utils import make_labber_cfg_snapshot


class RangeCfg(ConfigBase):
    """Synthetic cfg with shared or per-item ranges and runtime numeric values.

    ranges holds a shared pair or one pair per item, in arbitrary working units.
    samples holds a one-dimensional float array. parameter is a QICK sweep value.
    """

    ranges: tuple[float, float] | list[tuple[float, float]]
    samples: NDArray[np.float64]
    parameter: QickParam


@pytest.mark.parametrize(
    ("ranges", "expected"),
    [
        ((1.0, 110.0), [1.0, 110.0]),
        ([(1.0, 110.0), (2.0, 120.0)], [[1.0, 110.0], [2.0, 120.0]]),
    ],
)
def test_labber_snapshot_maps_ranges_without_mutating_cfg(
    ranges: tuple[float, float] | list[tuple[float, float]],
    expected: list[float] | list[list[float]],
) -> None:
    parameter = QickParam(start=1.0, spans={"sweep": 2.0})
    cfg = RangeCfg(ranges=ranges, samples=np.array([1.0, 2.0]), parameter=parameter)
    original = cfg.model_copy(deep=True)

    snapshot = make_labber_cfg_snapshot(cfg, schema_version="1.0")

    assert snapshot.values == {
        "ranges": expected,
        "samples": [1.0, 2.0],
        "parameter": str(parameter),
    }
    assert snapshot.cfg_type == "RangeCfg"
    assert snapshot.schema_version == "1.0"
    assert cfg.ranges == original.ranges
    np.testing.assert_array_equal(cfg.samples, original.samples)
    assert str(cfg.parameter) == str(original.parameter)
