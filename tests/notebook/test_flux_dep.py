"""FluxDep Notebook analysis retains explicit sources across canonical loads."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.datafile import LabberData
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.onetone.flux_dep import FluxDepExp, FluxDepResult
from zcu_tools.notebook import NotebookAdapter
from zcu_tools.notebook.experiments import FluxDepAnalyzer, FluxDepPickerOptions

from tests.experiment.v2.onetone.flux_dep_support import make_cfg, make_result


@pytest.fixture(autouse=True)
def suppress_display(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("IPython.display.display", lambda _widget: None)


def test_canonical_load_keeps_pending_pick_source_and_retains_old_figure(
    tmp_path: Path,
) -> None:
    cfg_a = make_cfg()
    source_a = RunRecord(cfg=cfg_a, result=make_result())
    source_b = RunRecord(
        cfg=cfg_a.model_copy(update={"reps": 3}, deep=True),
        result=FluxDepResult(
            source_a.result.values,
            source_a.result.freqs + 1.0,
            np.asarray(source_a.result.signals * (2 + 1j), dtype=np.complex128),
        ),
    )
    core = FluxDepExp()
    path_a, path_b = tmp_path / "source-a.hdf5", tmp_path / "source-b.hdf5"
    core.save(source_a, path_a)
    core.save(source_b, path_b)
    data = LabberData.from_file(path_a)
    np.testing.assert_array_equal(
        data.get_x("Frequency", "Hz"), source_a.result.freqs * 1e6
    )
    np.testing.assert_array_equal(
        data.get_x("Flux device value", "a.u."), source_a.result.values
    )
    np.testing.assert_array_equal(data.get_z("Signal", "a.u."), source_a.result.signals)
    assert data.axis_order == ("Frequency", "Flux device value")

    adapter = NotebookAdapter(core)
    loaded_a = adapter.load(path_a)
    assert loaded_a.cfg is not None
    assert loaded_a.cfg.model_dump() == cfg_a.model_dump()
    np.testing.assert_array_equal(loaded_a.result.signals, source_a.result.signals)
    tool = FluxDepAnalyzer()
    previous = tool.start(loaded_a, FluxDepPickerOptions(-0.2, 0.3)).done()
    old_plots = tool.analysis_plots
    assert old_plots is not None
    pending = tool.start(loaded_a, FluxDepPickerOptions(-0.2, 0.3))
    pending.set_positions(-0.1, 0.4)
    loaded_b = adapter.load(path_b)
    assert adapter.last_run is loaded_b
    assert loaded_b.cfg is not None and loaded_b.cfg.reps == 3
    np.testing.assert_array_equal(loaded_b.result.signals, source_b.result.signals)
    assert tool.analysis is previous and tool.analysis_plots is old_plots
    with pytest.raises(FileNotFoundError):
        adapter.load(tmp_path / "missing.hdf5")
    assert adapter.last_run is loaded_b
    assert tool.analysis is previous and tool.analysis_plots is old_plots

    completed = pending.done()
    assert completed.source is loaded_a
    assert completed.options.flux_half == pytest.approx(-0.1)
    assert completed.options.flux_int == pytest.approx(0.4)
    assert completed.analysis.flux_period == pytest.approx(1.0)
    assert adapter.last_run is loaded_b
    saved = adapter.save(loaded_a, tmp_path / "saved-a.hdf5", unique=False)
    round_trip = core.load(saved)
    assert adapter.last_run is loaded_b
    assert round_trip.cfg is not None
    assert round_trip.cfg.model_dump() == cfg_a.model_dump()
    np.testing.assert_array_equal(round_trip.result.values, loaded_a.result.values)
    np.testing.assert_array_equal(round_trip.result.freqs, loaded_a.result.freqs)
    np.testing.assert_array_equal(round_trip.result.signals, loaded_a.result.signals)
    old_plots.release()
    previous.figures["pick"].savefig(tmp_path / "old-pick.png")
    assert (tmp_path / "old-pick.png").stat().st_size > 0
    assert previous.source is loaded_a
    current_plots = tool.analysis_plots
    assert current_plots is not None
    current_plots.release()


@pytest.mark.parametrize("cfg_kind", ["missing", "invalid"])
def test_loaded_data_without_valid_cfg_can_be_picked_but_not_saved(
    cfg_kind: str, tmp_path: Path
) -> None:
    source = RunRecord(cfg=make_cfg(), result=make_result())
    core = FluxDepExp()
    original = tmp_path / "original.hdf5"
    core.save(source, original)
    data = LabberData.from_file(original)
    metadata = json.loads(data.comment)
    if cfg_kind == "missing":
        del metadata["cfg"]
    else:
        metadata["cfg"]["reps"] = "not-an-integer"
    data.comment = json.dumps(metadata)
    altered = tmp_path / "without-valid-cfg.hdf5"
    data.write(altered)
    adapter = NotebookAdapter(core)
    expected_warning = (
        pytest.warns(UserWarning, match="Config validation failed")
        if cfg_kind == "invalid"
        else nullcontext()
    )
    with expected_warning:
        loaded = adapter.load(altered)
    assert loaded.cfg is None and adapter.last_run is loaded
    np.testing.assert_array_equal(loaded.result.values, source.result.values)
    np.testing.assert_array_equal(loaded.result.freqs, source.result.freqs)
    np.testing.assert_array_equal(loaded.result.signals, source.result.signals)
    tool = FluxDepAnalyzer()
    completed = tool.start(loaded, FluxDepPickerOptions(-0.2, 0.3)).done()
    assert completed.source is loaded and completed.cfg is None
    assert completed.analysis.flux_half == pytest.approx(-0.2)
    assert completed.analysis.flux_int == pytest.approx(0.3)
    assert completed.analysis.flux_period == pytest.approx(1.0)
    with pytest.raises(ValueError, match="RunRecord.cfg is None"):
        adapter.save(loaded, tmp_path / "rejected.hdf5", unique=False)
    assert not (tmp_path / "rejected.hdf5").exists()
    assert tool.analysis is completed
    plots = tool.analysis_plots
    assert plots is not None
    plots.release()
    completed.figures["pick"].savefig(tmp_path / "nullable-pick.png")
    assert (tmp_path / "nullable-pick.png").stat().st_size > 0
