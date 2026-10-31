"""Readout calibration survives core/native conversion and data.h5 cfg storage."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.cfg_model import (
    Centroid3Classifier,
    CircleClassifier,
    ConfigBase,
    LinearClassifier,
)
from zcu_tools.cfg_model import (
    ModuleCfgFactory as CoreModuleCfgFactory,
)
from zcu_tools.experiment import AxesSpec, Axis, RunRecord, ZSpec, load_run, save_run
from zcu_tools.program.v2.modules import ModuleCfgFactory, readout

from tests._native_support import native_metadata


class ReadoutCfg(ConfigBase):
    """readout holds native settings and optional normalized ADC calibration."""

    readout: readout.ReadoutCfg


@dataclass
class ReadoutResult:
    """points are dimensionless indices; signal holds dimensionless ADC means."""

    points: np.ndarray
    signal: np.ndarray


SPEC = AxesSpec(
    axes=(Axis("points", "Point", ""),),
    z=ZSpec("signal", "Signal", ""),
    result_type=ReadoutResult,
    cfg_type=ReadoutCfg,
    tag="test/readout-calibration",
)


@pytest.mark.parametrize("family", ["pulse", "mux"])
@pytest.mark.parametrize("mode", ["circle", "linear", "centroid3"])
def test_classifier_survives_native_conversion_and_data_h5(
    tmp_path: Path, family: str, mode: str
):
    classifier = (
        CircleClassifier(
            g_center=1.0 - 2.0j,
            e_center=-3.0 + 4.0j,
            radius=0.5,
            confusion_matrix=[[0.9, 0.1, 0.0], [0.2, 0.8, 0.0], [0.0, 0.0, 1.0]],
        )
        if mode == "circle"
        else Centroid3Classifier(
            g_center=1.0 - 2.0j,
            e_center=-3.0 + 4.0j,
            f_center=5.0 + 6.0j,
            confusion_matrix=[[1.2, -0.2, 0.1], [0.3, 0.6, 0.2], [0.0, 0.1, 0.8]],
        )
        if mode == "centroid3"
        else LinearClassifier(
            angle=-0.75, threshold=0.25, confusion_matrix=[[0.9, 0.1], [0.2, 0.8]]
        )
    )
    common = {"type": f"readout/{family}", "gen_ch": 0, "nqz": 1, "ro_length": 1.5}
    source = CoreModuleCfgFactory.from_raw(
        {
            **common,
            "freq": 700.0,
            "gain": 0.2,
            "waveform": {"style": "const", "length": 1.0},
            "ro_ch": 1,
            "classifier": classifier,
        }
        if family == "pulse"
        else {
            **common,
            "mixer_freq": 6000.0,
            "length": 1.0,
            "tones": [
                {"freq": 6020.0, "gain": 0.1, "ro_ch": 3, "classifier": classifier},
                {"freq": 5980.0, "gain": 0.2, "ro_ch": 1},
            ],
        }
    )
    native = ModuleCfgFactory.from_core(source, context=None)
    assert native.to_dict() == source.to_dict()
    cfg = ReadoutCfg.model_validate({"readout": native})
    record = RunRecord(
        cfg=cfg,
        result=ReadoutResult(np.array([0.0, 1.0]), np.array([2.0, 3.0])),
        metadata=native_metadata(SPEC.tag).model_dump(mode="json"),
    )

    path = tmp_path / "data.h5"
    save_run(record, path, spec=SPEC)
    restored, _ = load_run(path, spec=SPEC)

    assert restored.cfg == cfg
    np.testing.assert_array_equal(restored.result.signal, [2.0, 3.0])
