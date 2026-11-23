"""Native Load retains acquisition data when a cfg asset is unavailable."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pytest
from pydantic import model_validator
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import load
from zcu_tools.experiment import (
    AxesSpec,
    Axis,
    CfgUnavailableWarning,
    RunRecord,
    ZSpec,
    restore_run,
    save_run,
)

from tests._native_support import native_metadata


@dataclass
class Result:
    frequency: np.ndarray
    signal: np.ndarray


def test_restore_retains_measurements_when_cfg_asset_was_moved(tmp_path: Path) -> None:
    class AssetCfg(ConfigBase):
        asset: Path

        @model_validator(mode="after")
        def check_asset(self) -> Self:
            self.asset.read_bytes()
            return self

    spec = AxesSpec(
        axes=(Axis("frequency", "Frequency", "Hz"),),
        z=ZSpec("signal", "Signal", "V"),
        result_type=Result,
        cfg_type=AssetCfg,
        tag="test/cfg_asset",
    )
    asset = tmp_path / "waveform.bin"
    asset.write_bytes(b"asset used by typed configuration")
    path = tmp_path / "data.h5"
    metadata = native_metadata(spec.tag).model_dump(mode="json")
    result = Result(frequency=np.array([4.0]), signal=np.array([2.0 + 3j]))
    save_run(
        RunRecord(cfg=AssetCfg(asset=asset), result=result, metadata=metadata),
        path,
        spec=spec,
    )
    asset.rename(tmp_path / "moved.bin")

    with pytest.warns(CfgUnavailableWarning, match="waveform.bin") as caught:
        restored = restore_run(load(path), spec=spec)

    assert restored.record.cfg is None
    assert restored.cfg_issue == str(caught[0].message)
    assert restored.record.metadata == metadata
    np.testing.assert_array_equal(restored.record.result.frequency, result.frequency)
    np.testing.assert_array_equal(restored.record.result.signal, result.signal)
