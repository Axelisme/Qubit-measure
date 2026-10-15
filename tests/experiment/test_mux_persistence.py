"""New mux cfg and helper results use the existing numeric-axis storage seam."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from zcu_tools.datafile import decode_labber_comment, load_labber_data, write_labber
from zcu_tools.experiment import (
    AxesSpec,
    ExpCfgModel,
    RunRecord,
    ZSpec,
    load_run,
    prepare_run_data,
    save_run,
)
from zcu_tools.experiment.axes_spec import Axis
from zcu_tools.program.v2.modules import readout

from tests._native_support import native_metadata
from tests.program.v2._mux_support import RecordingProgram, mux_soccfg, native_mux


class MuxCfg(ExpCfgModel):
    readout: readout.ReadoutCfg


@dataclass
class MuxResult:
    offsets: np.ndarray
    tones: np.ndarray
    signals: np.ndarray


def test_mux_numeric_tones_and_order_survive_native_and_labber(tmp_path: Path):
    cfg = MuxCfg(readout=native_mux())
    module = cfg.readout.build("mux")
    assert isinstance(module, readout.MuxReadout)
    prog = RecordingProgram(mux_soccfg(), [module])
    raw = [
        np.array([[[11.0, -12.0], [21.0, -22.0], [31.0, -32.0]]]),
        np.array([[[41.0, -42.0], [51.0, -52.0], [61.0, -62.0]]]),
    ]
    iq = module.collect_iq(prog, raw)
    result = MuxResult(
        np.array([-1.0, 0.0, 1.0]),
        np.arange(2, dtype=np.int64),
        iq[..., 0] + 1j * iq[..., 1],
    )
    spec = AxesSpec(
        axes=(
            Axis("tones", "Tone", "", dtype=np.int64),
            Axis("offsets", "Frequency offset", "Hz", scale=1e6),
        ),
        z=ZSpec("signals", "Signal", "", dtype=np.complex128),
        result_type=MuxResult,
        cfg_type=MuxCfg,
        tag="test/mux",
    )
    record = RunRecord(cfg=cfg, result=result)
    path = tmp_path / "native.h5"
    save_run(record, path, spec=spec, metadata=native_metadata(spec.tag))
    restored, _ = load_run(path, spec=spec)
    assert restored.cfg is not None
    np.testing.assert_array_equal(
        restored.result.signals,
        [[11 - 12j, 41 - 42j], [21 - 22j, 51 - 52j], [31 - 32j, 61 - 62j]],
    )
    np.testing.assert_array_equal(restored.result.tones, [0, 1])
    np.testing.assert_array_equal(restored.result.offsets, result.offsets)
    assert restored.cfg.to_dict() == cfg.to_dict()
    assert isinstance(restored.cfg.to_dict()["readout"]["tones"], list)
    assert [t["ro_ch"] for t in restored.cfg.to_dict()["readout"]["tones"]] == [4, 1]

    prepared = prepare_run_data(record, spec=spec, comment="")
    labber = tmp_path / "labber.h5"
    write_labber(labber, prepared.payload, cfg=prepared.cfg)
    loaded = load_labber_data(str(labber))
    np.testing.assert_array_equal(loaded.z, result.signals)
    np.testing.assert_array_equal(loaded.axes[0][2], [0, 1])
    np.testing.assert_array_equal(loaded.axes[1][2], result.offsets * 1e6)
    envelope = decode_labber_comment(loaded.metadata.comment)
    assert envelope.cfg == cfg.model_dump(mode="json")
