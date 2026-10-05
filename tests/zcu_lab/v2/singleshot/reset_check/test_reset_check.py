from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import AnalyzeRequest, NoAnalyzeParams
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from zcu_lab.definitions import ADAPTERS
from zcu_lab.v2.singleshot.reset_check.core import ResetCheckResult
from zcu_lab.v2.singleshot.reset_check.gui import SsResetCheckAdapter


def test_adapter_analyzes_populations_with_external_correction() -> None:
    matrix = np.array([[0.9, 0.1, 0], [0.1, 0.9, 0], [0, 0, 1]])
    populations = np.tile([0.85, 0.1, 0.05], (4, 3, 1))
    measured = populations @ matrix
    req = AnalyzeRequest(
        run_result=RunRecord(
            cfg=None,
            result=ResetCheckResult(
                np.arange(4, dtype=float), np.arange(3), measured[..., :2]
            ),
        ),
        analyze_params=NoAnalyzeParams(),
        md=cast(Any, {"confusion_matrix": matrix}),
        ml=cast(Any, None),
        predictor=None,
    )
    assert ADAPTERS["singleshot/reset_check"] is SsResetCheckAdapter
    plots = Plots(NonPresentingHost())
    try:
        out = SsResetCheckAdapter().analyze(req, plots=plots)
        figure = plots.finish(present=False)["populations"]
        assert len(figure.axes[0].lines) == 9
        summary = out.to_summary_dict()
        assert summary["reset_mean_excited_population"] == pytest.approx(0.1)
        assert summary["reset_max_other_population"] == pytest.approx(0.05)
        assert summary["analyzed_reset_points"] == 4
        assert "readout_condition_number" not in summary
    finally:
        plots.finish(present=False)
        plots.release()
