"""Real T1 integration through the common Notebook record boundary."""

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.datafile import LabberData
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.twotone.time_domain.t1 import (
    T1AnalyzeOptions,
    T1Cfg,
    T1Exp,
    T1Result,
)
from zcu_tools.notebook import NotebookAdapter
from zcu_tools.plotting.plots import NonPresentingHost


@pytest.fixture
def t1_cfg() -> T1Cfg:
    return T1Cfg.model_validate(
        {
            "reps": 2,
            "rounds": 1,
            "uniform": False,
            "modules": {
                "pi_pulse": {
                    "ch": 0,
                    "nqz": 1,
                    "gain": 1.0,
                    "freq": 4000.0,
                    "waveform": {"style": "const", "length": 0.4},
                },
                "readout": {
                    "type": "readout/direct",
                    "ro_ch": 0,
                    "ro_length": 1.0,
                    "ro_freq": 7000.0,
                },
            },
            "sweep": {"length": [0.0, 10.0, 40.0]},
        }
    )


def make_source(cfg: T1Cfg | None, t1: float) -> RunRecord[T1Cfg, T1Result]:
    times = np.linspace(0, 80, 81)
    return RunRecord(
        cfg=cfg,
        result=T1Result(times, np.exp(-times / t1).astype(np.complex128)),
    )


def test_canonical_load_and_explicit_analysis_keep_source_and_native_figure(
    t1_cfg: T1Cfg, tmp_path: Path
) -> None:
    adapter = NotebookAdapter(host=NonPresentingHost())(T1Exp())
    original = make_source(t1_cfg, 20.0)
    path_a = adapter.save(tmp_path / "a.hdf5", source=original)
    source_a = adapter.load(path_a)
    assert source_a.cfg == t1_cfg
    np.testing.assert_allclose(
        source_a.result.times, original.result.times, rtol=1e-12, atol=0.0
    )
    np.testing.assert_array_equal(source_a.result.signals, original.result.signals)
    initial = adapter.analyze(T1AnalyzeOptions(skip=1))
    assert initial.result.t1 == pytest.approx(20.0)

    cfg_b = t1_cfg.model_copy(update={"reps": 3})
    path_b = adapter.save(tmp_path / "b.hdf5", source=make_source(cfg_b, 12.0))
    source_b = adapter.load(path_b)
    assert source_b.cfg == cfg_b
    assert adapter.last_run is source_b
    assert adapter.analysis is None
    initial.figures["fit"].savefig(tmp_path / "retained.png")
    assert (tmp_path / "retained.png").stat().st_size > 0

    older = adapter.analyze(T1AnalyzeOptions(skip=2), source=source_a)
    assert older.source is source_a
    assert older.result.t1 == pytest.approx(20.0)
    assert older.options == T1AnalyzeOptions(skip=2)
    assert adapter.last_run is source_b
    latest = adapter.analyze(T1AnalyzeOptions())
    assert latest.source is source_b
    assert latest.result.t1 == pytest.approx(12.0)

    saved_a = adapter.save(tmp_path / "saved-a.hdf5", source=source_a)
    reloaded_a = adapter.load(saved_a)
    assert reloaded_a.cfg == t1_cfg
    np.testing.assert_array_equal(reloaded_a.result.signals, original.result.signals)
    assert adapter.last_run is reloaded_a
    assert adapter.analysis is None


@pytest.mark.parametrize("cfg_state", ["missing", "invalid"])
def test_loaded_data_without_valid_cfg_remains_analyzable_but_not_saveable(
    t1_cfg: T1Cfg, tmp_path: Path, cfg_state: str
) -> None:
    adapter = NotebookAdapter(host=NonPresentingHost())(T1Exp())
    original = make_source(t1_cfg, 20.0)
    valid_path = adapter.save(tmp_path / "valid.hdf5", source=original)
    adapter.load(valid_path)
    previous = adapter.analyze(T1AnalyzeOptions())

    payload = LabberData.load(str(valid_path))
    metadata = json.loads(payload.comment)
    if cfg_state == "missing":
        metadata.pop("cfg")
    else:
        metadata["cfg"]["reps"] = "invalid"
    payload.comment = json.dumps(metadata)
    boundary_path = Path(payload.save(str(tmp_path / "without-cfg.hdf5")))
    warning = pytest.warns(UserWarning) if cfg_state == "invalid" else nullcontext()
    with warning:
        loaded = adapter.load(boundary_path)

    assert loaded.cfg is None
    np.testing.assert_array_equal(loaded.result.signals, original.result.signals)
    assert adapter.last_run is loaded
    assert adapter.analysis is None
    analysis = adapter.analyze(T1AnalyzeOptions())
    assert analysis.source is loaded
    assert analysis.result.t1 == pytest.approx(20.0)

    destination = tmp_path / "rejected.hdf5"
    with pytest.raises(ValueError, match=r"RunRecord\.cfg is None"):
        adapter.save(destination, source=loaded)
    assert not destination.exists()
    assert adapter.last_run is loaded
    assert adapter.analysis is analysis
    previous.figures["fit"].savefig(tmp_path / "previous.png")
    assert (tmp_path / "previous.png").stat().st_size > 0
