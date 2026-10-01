"""The fixed GUI harness preserves numeric behavior through shared records."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from zcu_tools.device import FakeDeviceInfo
from zcu_tools.experiment.context import QickContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2_gui.measure.adapters.fake.stub import (
    FakeAdapter,
    FakeAnalysis,
    FakeAnalyzeOptions,
    FakeAnalyzeParams,
    FakeExp,
    FakeExpCfg,
    FakeResult,
    FakeRunResult,
)
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    MetaDictWriteback,
    RunRequest,
    SaveDataRequest,
    SessionEnv,
    WritebackRequest,
)
from zcu_tools.notebook import NotebookAdapter
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2 import SweepCfg


def make_cfg(noise_scale: float = 0.1) -> FakeExpCfg:
    return FakeExpCfg(
        noise_scale=noise_scale,
        sweep=SweepCfg(start=5.0, stop=5.0, expts=1, step=0.0),
    )


@pytest.mark.parametrize("noise_scale", [0.0, 0.05, 0.2])
def test_core_run_keeps_seeded_eleven_samples_independent_of_sweep(
    noise_scale: float,
) -> None:
    cfg = make_cfg(noise_scale)
    before = cfg.model_copy(deep=True)
    plots = Plots(NonPresentingHost())
    result = FakeExp().run(cfg, context=QickContext(None, None, plots))
    expected = np.random.default_rng(seed=42).normal(0.0, noise_scale, size=11)
    np.testing.assert_array_equal(result.data, expected)
    assert result.data.dtype == np.float64 and cfg == before
    assert tuple(plots.finish()) == ()
    plots.release()


@pytest.mark.parametrize(
    "threshold,marked", [(0.49, True), (0.5, False), (0.51, False)]
)
def test_nullable_source_peak_and_strict_threshold_markers(
    threshold: float, marked: bool
) -> None:
    source = FakeRunResult(cfg=None, result=FakeResult(np.array([0.2, -0.5, 0.1])))
    plots = Plots(NonPresentingHost())
    result = FakeExp().analyze(source, FakeAnalyzeOptions(threshold), plots=plots)
    assert result == FakeAnalysis(peak=0.5)
    figures = plots.finish()
    assert tuple(figures) == ("fit",)
    lines = figures["fit"].axes[0].lines
    np.testing.assert_array_equal(lines[0].get_ydata(), source.result.data)
    assert len(lines) == (3 if marked else 2)
    if marked:
        np.testing.assert_array_equal(lines[1].get_xdata(), [1, 1])
    np.testing.assert_array_equal(lines[-1].get_ydata(), [threshold, threshold])
    plots.release()


def test_adapter_run_keeps_matching_cfg_and_detached_device_after_b() -> None:
    adapter = FakeAdapter()
    raw = make_cfg(0.05).to_dict()
    device = FakeDeviceInfo(address="none", value=0.25)
    request = RunRequest(soc=None, soccfg=None, device_snapshot={"bias": device})
    plots = Plots(NonPresentingHost())
    a = adapter.run(request, raw, plots=plots)
    raw["noise_scale"] = 0.2
    device.value = 0.75
    b = adapter.run(request, raw, plots=plots)
    assert isinstance(a, RunRecord)
    assert a.cfg is not None and b.cfg is not None and a.cfg.dev is not None
    assert a.cfg.noise_scale == 0.05 and b.cfg.noise_scale == 0.2
    recorded_device = a.cfg.dev["bias"]
    assert isinstance(recorded_device, FakeDeviceInfo)
    assert recorded_device.value == 0.25
    np.testing.assert_array_equal(
        a.result.data, np.random.default_rng(seed=42).normal(0.0, 0.05, 11)
    )
    plots.finish()
    plots.release()


def test_adapter_projects_explicit_source_threshold_and_peak_writeback() -> None:
    adapter = FakeAdapter()
    source = FakeRunResult(cfg=None, result=FakeResult(np.array([0.2, -0.5, 0.1])))
    plots = Plots(NonPresentingHost())
    answer = adapter.analyze(
        AnalyzeRequest(
            source,
            FakeAnalyzeParams(threshold=0.49),
            md=Mock(),
            ml=Mock(),
            predictor=None,
        ),
        plots=plots,
    )
    assert answer.peak == 0.5 and answer.to_summary_dict() == {"peak": 0.5}
    assert len(plots.finish()["fit"].axes[0].lines) == 3
    items = adapter.get_writeback_items(
        WritebackRequest(source, answer, Mock(spec=SessionEnv))
    )
    assert len(items) == 1
    item = items[0]
    assert isinstance(item, MetaDictWriteback)
    assert item.target_name == "fake_peak" and item.proposed_value == 0.5
    plots.release()


@pytest.mark.parametrize(
    "field,value", [("reps", "100"), ("gain", "0.1"), ("noise_scale", None)]
)
def test_adapter_rejects_bad_raw_controls_before_acquisition(
    field: str, value: object
) -> None:
    raw = make_cfg().to_dict()
    raw[field] = value
    plots = Plots(NonPresentingHost())
    with pytest.raises(RuntimeError, match=field):
        FakeAdapter().run(RunRequest(None, None, {}), raw, plots=plots)
    assert tuple(plots.finish()) == ()
    plots.release()


def test_save_stays_inert_and_load_is_explicitly_unsupported(tmp_path: Path) -> None:
    source = FakeRunResult(cfg=None, result=FakeResult(np.array([0.1])))
    path = tmp_path / "stub.hdf5"
    FakeExp().save(source, path, comment="source A", tag="test")
    FakeAdapter().save(
        SaveDataRequest(
            run_result=source,
            data_path=str(path),
            comment="source A",
            md=Mock(),
            ml=Mock(),
            chip_name="chip",
            qub_name="qubit",
            res_name="resonator",
            active_label="zero",
        )
    )
    assert not path.exists()
    with pytest.raises(NotImplementedError, match="inert.*no data file format"):
        FakeExp().load(path)


def test_notebook_b_keeps_explicit_a_native_plot_and_last_success_on_failure(
    tmp_path: Path,
) -> None:
    adapter = NotebookAdapter(
        FakeExp(), soc=object(), soccfg=object(), host=NonPresentingHost()
    )
    a = adapter.run(make_cfg(0.05))
    b = adapter.run(make_cfg(0.2))
    answer = adapter.analyze(FakeAnalyzeOptions(threshold=0.0), source=a)
    assert answer.source is a and adapter.last_run is b
    assert answer.result.peak == pytest.approx(float(np.max(np.abs(a.result.data))))
    np.testing.assert_array_equal(
        answer.figures["fit"].axes[0].lines[0].get_ydata(), a.result.data
    )
    handle = adapter.analysis_presentation
    assert handle is not None
    handle.release()
    answer.figures["fit"].axes[0].set_title("User annotation")
    answer.figures["fit"].savefig(tmp_path / "native.png")
    assert (tmp_path / "native.png").stat().st_size > 0
    adapter.save(a, tmp_path / "no-file.hdf5")
    assert not (tmp_path / "no-file.hdf5").exists()
    bad = FakeRunResult(cfg=None, result=FakeResult(np.empty(0)))
    with pytest.raises(ValueError, match="zero-size array"):
        adapter.analyze(FakeAnalyzeOptions(), source=bad)
    assert adapter.analysis is answer and adapter.last_run is b
    with pytest.raises(NotImplementedError, match="inert"):
        adapter.load(tmp_path / "unsupported.hdf5")
    assert adapter.analysis is answer and adapter.last_run is b
