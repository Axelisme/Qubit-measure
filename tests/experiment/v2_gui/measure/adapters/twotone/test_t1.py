"""GUI T1 adapter hands one typed run and analysis to the shared core."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.v2.twotone.time_domain.t1 import (
    T1Analysis,
    T1Cfg,
    T1Exp,
    T1Result,
)
from zcu_tools.experiment.v2_gui.measure.adapters.twotone.time_domain.t1 import (
    T1Adapter,
    T1AnalyzeParams,
)
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    LoadDataRequest,
    RunRequest,
    SaveDataRequest,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots


def _result() -> T1Result:
    times = np.linspace(0, 100, 101)
    signals = (0.2 + 0.8 * np.exp(-times / 20)).astype(np.complex128)
    return T1Result(times, signals)


def test_t1_gui_analysis_preserves_core_numbers_and_named_fit() -> None:
    result = _result()
    source = RunRecord[T1Cfg, T1Result](cfg=None, result=result)
    plots = Plots(NonPresentingHost())
    answer = T1Adapter().analyze(
        AnalyzeRequest(
            source, T1AnalyzeParams(skip=3), md=Mock(), ml=Mock(), predictor=None
        ),
        plots=plots,
    )
    plots.finish()
    assert isinstance(answer.analysis, T1Analysis)
    assert answer.t1 == pytest.approx(20, rel=0.01)
    assert answer.t1_err == answer.analysis.t1_err
    assert answer.to_summary_dict()["t1"] == answer.t1
    assert tuple(plots) == ("fit",)
    np.testing.assert_array_equal(
        plots["fit"].axes[0].lines[0].get_xdata(), result.times[3:]
    )
    plots.release()


@pytest.mark.parametrize("dual_exp", [False, True])
def test_t1_summary_quality_uses_only_samples_after_skip(dual_exp: bool) -> None:
    times = np.linspace(0, 100, 101)
    values = 0.2 + 0.8 * np.exp(-times / 20)
    if dual_exp:
        values += 0.4 * np.exp(-times / 4)
    values += np.where(np.arange(times.size) % 2 == 0, 0.01, -0.01)
    values[:4] = 1000
    source = RunRecord[T1Cfg, T1Result](
        cfg=None, result=T1Result(times, values.astype(np.complex128))
    )
    plots = Plots(NonPresentingHost())
    try:
        answer = T1Adapter().analyze(
            AnalyzeRequest(
                source,
                T1AnalyzeParams(skip=4, dual_exp=dual_exp),
                md=Mock(),
                ml=Mock(),
                predictor=None,
            ),
            plots=plots,
        )
        summary = json.loads(json.dumps(answer.to_summary_dict(), allow_nan=False))
        quality = summary["fit_quality"]["fit"]
        assert 0.99 < quality["r2"] < 1.0
        assert 0.01 < quality["normalized_residual_rms"] < 0.03
        assert len(quality["relative_parameter_errors"]) == (5 if dual_exp else 3)
        json.dumps(summary, allow_nan=False)
    finally:
        plots.finish()
        plots.release()


def test_t1_gui_run_passes_captured_context_and_formal_cfg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = T1Adapter()
    cfg = Mock(uniform=False)
    monkeypatch.setattr(adapter, "build_exp_cfg", lambda _raw, _req: cfg)
    result = _result()
    observed: list[tuple[object, RunContext]] = []

    def run(_exp: T1Exp, config: object, *, context: RunContext) -> T1Result:
        observed.append((config, context))
        return result

    monkeypatch.setattr(T1Exp, "run", run)
    plots = Plots(NonPresentingHost())
    req = RunRequest(soc=Mock(), soccfg=Mock(), device_snapshot={})
    source = adapter.run(
        req,
        {"uniform": False},
        context=RunContext(
            req.soc, req.soccfg, plots, devices={}, cancel_signal=StopSignal()
        ),
    )
    assert source.result is result
    assert source.cfg is not None and source.cfg is not cfg
    assert source.cfg.uniform is False
    cfg.uniform = True
    assert source.cfg.uniform is False
    assert observed[0][0] is cfg
    assert observed[0][1].soc is req.soc
    assert observed[0][1].soccfg is req.soccfg
    assert observed[0][1].plots is plots
    plots.finish(present=False)
    plots.release()


def test_t1_gui_save_and_load_delegate_canonical_result_first_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = RunRecord[T1Cfg, T1Result](cfg=None, result=_result())
    observed: list[tuple[RunRecord[T1Cfg, T1Result], Path, str | None]] = []

    def save(
        _exp: T1Exp,
        data: RunRecord[T1Cfg, T1Result],
        path: Path,
        *,
        comment: str | None = None,
    ) -> None:
        observed.append((data, path, comment))

    def load(_exp: T1Exp, path: Path) -> RunRecord[T1Cfg, T1Result]:
        assert path == tmp_path / "data.hdf5"
        return source

    monkeypatch.setattr(T1Exp, "save", save)
    monkeypatch.setattr(T1Exp, "load", load)
    adapter = T1Adapter()
    path = str(tmp_path / "data.hdf5")
    adapter.save(
        SaveDataRequest(
            source,
            path,
            md=Mock(),
            ml=Mock(),
            chip_name="c",
            qub_name="q",
            res_name="r",
            active_label="a",
            comment="note",
        )
    )
    assert observed == [(source, Path(path), "note")]
    assert adapter.load(LoadDataRequest(path, md=Mock(), ml=Mock())) is source
