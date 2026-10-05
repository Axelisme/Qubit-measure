from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    MetaDictWriteback,
    RunRequest,
    SessionEnv,
    WritebackRequest,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2.mocksoc import make_mock_soc

from tests.zcu_lab.v2.lookback._lookback_support import make_lookback_cfg
from zcu_lab.v2.lookback.core import LookbackCfg, LookbackExp, LookbackResult
from zcu_lab.v2.lookback.gui import LookbackAdapter, LookbackAnalyzeParams


def test_analysis_preserves_gui_smoothing_and_timefly_writeback() -> None:
    result = LookbackResult(
        np.arange(9, dtype=np.float64),
        np.array([0, 0, 0, 8j, -8j, 0, 0, 0, 0], dtype=np.complex128),
    )
    source = RunRecord[LookbackCfg, LookbackResult](cfg=None, result=result)
    adapter = LookbackAdapter()
    plots = Plots(NonPresentingHost())
    answer = adapter.analyze(
        AnalyzeRequest(
            source,
            adapter.get_analyze_params(source, Mock(spec=SessionEnv)),
            md=Mock(),
            ml=Mock(),
            predictor=None,
        ),
        plots=plots,
    )
    figures = plots.finish()

    assert answer.predict_offset == pytest.approx(0.0)
    assert answer.to_summary_dict() == {"predict_offset": answer.predict_offset}
    items = adapter.get_writeback_items(
        WritebackRequest(source, answer, Mock(spec=SessionEnv))
    )
    assert len(items) == 1
    item = items[0]
    assert isinstance(item, MetaDictWriteback)
    assert item.target_name == "timeFly"
    assert item.proposed_value == answer.predict_offset
    assert tuple(figures) == ("fit",)
    q_values = np.asarray(figures["fit"].axes[0].lines[1].get_ydata(), dtype=np.float64)
    assert np.any(q_values > 0) and np.any(q_values < 0)
    plots.release()


def test_run_captures_formal_cfg_and_the_same_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_lookback_cfg()
    raw_cfg = cfg.to_dict()
    soc, soccfg = make_mock_soc()
    req = RunRequest(soc=soc, soccfg=soccfg, device_snapshot={})
    plots = Plots(NonPresentingHost())
    observed: list[tuple[LookbackCfg, RunContext]] = []
    data = LookbackResult(np.array([0.4, 0.5]), np.array([1j, 2j]))

    def run(
        self: LookbackExp, config: LookbackCfg, *, context: RunContext
    ) -> LookbackResult:
        observed.append((config, context))
        return data

    monkeypatch.setattr(LookbackExp, "run", run)
    source = LookbackAdapter().run(
        req,
        raw_cfg,
        context=RunContext(
            req.soc, req.soccfg, plots, devices={}, cancel_signal=StopSignal()
        ),
    )

    config, context = observed[0]
    assert context.soc is req.soc
    assert context.soccfg is req.soccfg
    assert context.plots is plots
    assert source.result is data
    assert config.reps == cfg.reps
    assert config.modules == cfg.modules
    assert source.cfg == config
    config.reps = 11
    raw_cfg["reps"] = 9
    assert source.cfg is not None and source.cfg.reps == 1
    plots.finish()
    plots.release()


def test_plot_fit_false_keeps_the_data_figure() -> None:
    source = RunRecord[LookbackCfg, LookbackResult](
        cfg=None,
        result=LookbackResult(np.arange(3.0), np.array([0j, 1j, 10j])),
    )
    plots = Plots(NonPresentingHost())
    answer = LookbackAdapter().analyze(
        AnalyzeRequest(
            source,
            LookbackAnalyzeParams(plot_fit=False),
            md=Mock(),
            ml=Mock(),
            predictor=None,
        ),
        plots=plots,
    )
    figures = plots.finish()
    assert answer.predict_offset == pytest.approx(0.0)
    assert len(figures["fit"].axes[0].lines) == 3
    plots.release()
