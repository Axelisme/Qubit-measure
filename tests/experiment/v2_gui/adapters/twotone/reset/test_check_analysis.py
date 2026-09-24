from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from zcu_tools.experiment.v2.twotone.reset.rabi_check import (
    RabiCheckExp,
    RabiCheckResult,
)
from zcu_tools.experiment.v2_gui.adapters.twotone.reset.check import (
    RabiCheckAdapter,
    RabiCheckAnalyzeResult,
)
from zcu_tools.gui.app.main.adapter import (
    AnalysisMode,
    AnalyzeRequest,
    NoAnalyzeParams,
)
from zcu_tools.gui.cfg import CfgSectionSpec


def test_reset_check_adapter_exposes_figure_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert RabiCheckAdapter.capabilities.analysis is AnalysisMode.FIT
    assert RabiCheckAdapter.analyze_params_cls() is NoAnalyzeParams

    figure = Figure()
    result = cast(RabiCheckResult, object())
    seen: list[RabiCheckResult] = []

    def fake_analyze(self: RabiCheckExp, run_result: RabiCheckResult) -> Figure:
        seen.append(run_result)
        return figure

    monkeypatch.setattr(RabiCheckExp, "analyze", fake_analyze)
    req = AnalyzeRequest(
        run_result=result,
        analyze_params=NoAnalyzeParams(),
        md=cast(Any, None),
        ml=cast(Any, None),
        predictor=None,
    )
    out = RabiCheckAdapter().analyze(req)

    assert isinstance(out, RabiCheckAnalyzeResult)
    assert out.figure is figure
    assert seen == [result]
    plt.close(figure)


def test_reset_check_cfg_has_one_rabi_pulse_field() -> None:
    modules = RabiCheckAdapter.cfg_definition().spec.fields["modules"]
    assert isinstance(modules, CfgSectionSpec)
    assert "rabi_pulse" in modules.fields
    assert "pi_pulse" not in modules.fields
