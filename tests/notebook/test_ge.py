"""Real GE integration through the shared Notebook record boundary."""

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.datafile import LabberData
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.singleshot.ge import (
    GE_Cfg,
    GE_Exp,
    GE_Result,
    GEAnalyzeOptions,
    GEModuleCfg,
    GEPostAnalyzeOptions,
)
from zcu_tools.notebook import NotebookAdapter
from zcu_tools.notebook.experiments import GEPostAnalyzer
from zcu_tools.plotting.plots import NonPresentingHost
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


def make_source() -> RunRecord[GE_Cfg, GE_Result]:
    rng = np.random.default_rng(83)
    excited = rng.random((2, 6000)) < np.array([0.1, 0.9])[:, None]
    signals = np.asarray(
        np.where(excited, 1 + 0.4j, -1 - 0.4j)
        + 0.2 * (rng.normal(size=excited.shape) + 1j * rng.normal(size=excited.shape)),
        dtype=np.complex128,
    )
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.2,
        freq=7000.0,
        phase=0.0,
        waveform=ConstWaveformCfg(length=1.0),
    )
    cfg = GE_Cfg(
        reps=1,
        rounds=1,
        shots=6000,
        modules=GEModuleCfg(
            probe_pulse=PulseCfg(
                ch=0,
                nqz=1,
                gain=0.5,
                freq=4000.0,
                phase=0.0,
                waveform=ConstWaveformCfg(length=1.0),
            ),
            readout=PulseReadoutCfg(
                pulse_cfg=pulse,
                ro_cfg=DirectReadoutCfg(
                    ro_ch=0, ro_length=1.0, ro_freq=7000.0, trig_offset=0.0
                ),
            ),
        ),
    )
    return RunRecord(cfg, GE_Result(signals, np.arange(6000), np.array([0, 1])))


def test_post_uses_adopted_fit_source_and_keeps_prior_figures(tmp_path: Path) -> None:
    core, host = GE_Exp(), NonPresentingHost()
    adapter = NotebookAdapter(core, host=host)
    tool = GEPostAnalyzer(core, host=host)
    source = make_source()
    primary = adapter.analyze(
        GEAnalyzeOptions(backend="pca", length_ratio=0.01), source=source
    )
    assert primary.source is source
    assert primary.options == GEAnalyzeOptions(backend="pca", length_ratio=0.01)
    fit_presentation = adapter.analysis_presentation

    other = make_source()
    shifted = GE_Result(
        other.result.signals + 4.0,
        other.result.shot_indices,
        other.result.prepared_states,
    )
    path_b = adapter.save(RunRecord(other.cfg, shifted), tmp_path / "b.hdf5")
    current = adapter.load(path_b)
    assert adapter.last_run is current
    assert adapter.analysis is None

    record = tool.analyze(primary, GEPostAnalyzeOptions())

    assert adapter.last_run is current
    assert record.source is source
    assert record.primary is primary
    assert list(primary.figures) == ["fit"]
    assert list(record.figures) == ["post"]
    np.testing.assert_allclose(record.result.confusion.matrix, np.eye(3), atol=0.06)
    assert fit_presentation is not None
    post_presentation = tool.analysis_plots
    assert post_presentation is not None
    fit_presentation.release()
    post_presentation.release()
    primary.figures["fit"].savefig(tmp_path / "fit.png")
    record.figures["post"].savefig(tmp_path / "post.png")
    assert (tmp_path / "fit.png").stat().st_size > 0
    assert (tmp_path / "post.png").stat().st_size > 0


def test_canonical_save_load_replaces_latest_and_preserves_retained_figures(
    tmp_path: Path,
) -> None:
    core, host = GE_Exp(), NonPresentingHost()
    adapter = NotebookAdapter(core, host=host)
    tool = GEPostAnalyzer(core, host=host)
    path = adapter.save(make_source(), tmp_path / "source.hdf5")
    loaded = adapter.load(path)
    assert adapter.last_run is loaded
    old_fit = adapter.analyze(GEAnalyzeOptions(length_ratio=0.01))
    old_post = tool.analyze(old_fit, GEPostAnalyzeOptions())

    destination = adapter.save(loaded, tmp_path / "saved.hdf5", comment="GE notebook")
    replaced = adapter.load(destination)
    assert adapter.last_run is replaced
    assert adapter.analysis is None
    assert adapter.analysis_presentation is None
    assert tool.analysis is old_post
    np.testing.assert_array_equal(replaced.result.signals, loaded.result.signals)
    np.testing.assert_array_equal(replaced.result.prepared_states, [0, 1])
    assert replaced.cfg == loaded.cfg
    with pytest.raises(FileNotFoundError):
        adapter.load(tmp_path / "missing.hdf5")
    assert adapter.last_run is replaced
    old_fit.figures["fit"].savefig(tmp_path / "old-fit.png")
    old_post.figures["post"].savefig(tmp_path / "old-post.png")
    assert (tmp_path / "old-fit.png").stat().st_size > 0
    assert (tmp_path / "old-post.png").stat().st_size > 0


@pytest.mark.parametrize("cfg_state", ["missing", "invalid"])
def test_loaded_data_without_valid_cfg_supports_fit_and_post_but_not_save(
    tmp_path: Path, cfg_state: str
) -> None:
    core, host = GE_Exp(), NonPresentingHost()
    adapter = NotebookAdapter(core, host=host)
    original = make_source()
    valid_path = adapter.save(original, tmp_path / "valid.hdf5")
    payload = LabberData.load(str(valid_path))
    metadata = json.loads(payload.comment)
    if cfg_state == "missing":
        metadata.pop("cfg")
    else:
        metadata["cfg"]["shots"] = "invalid"
    payload.comment = json.dumps(metadata)
    path = Path(payload.save(str(tmp_path / "without-cfg.hdf5")))

    warning = pytest.warns(UserWarning) if cfg_state == "invalid" else nullcontext()
    with warning:
        loaded = adapter.load(path)
    assert loaded.cfg is None
    np.testing.assert_array_equal(loaded.result.signals, original.result.signals)
    primary = adapter.analyze(GEAnalyzeOptions(length_ratio=0.01))
    post = GEPostAnalyzer(core, host=host).analyze(primary, GEPostAnalyzeOptions())
    assert primary.source is loaded
    assert post.source is loaded
    np.testing.assert_allclose(post.result.confusion.matrix, np.eye(3), atol=0.06)

    destination = tmp_path / "rejected.hdf5"
    with pytest.raises(ValueError, match=r"RunRecord\.cfg is None"):
        adapter.save(loaded, destination)
    assert not destination.exists()
    assert adapter.last_run is loaded
    assert adapter.analysis is primary
