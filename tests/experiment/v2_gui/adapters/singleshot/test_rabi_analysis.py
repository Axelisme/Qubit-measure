from __future__ import annotations

from typing import Any, cast

import pytest
from matplotlib.figure import Figure
from zcu_tools.experiment.v2.singleshot.len_rabi import LenRabiExp, LenRabiResult
from zcu_tools.experiment.v2.singleshot.rabi_fit import RabiJointFitResult
from zcu_tools.experiment.v2_gui.adapters.singleshot.amp_rabi import SsAmpRabiAdapter
from zcu_tools.experiment.v2_gui.adapters.singleshot.len_rabi import (
    SsLenRabiAdapter,
    SsLenRabiAnalyzeParams,
)
from zcu_tools.gui.app.main.adapter import AnalyzeRequest, NoAnalyzeParams


def test_amp_analysis_has_no_decay_option() -> None:
    assert SsAmpRabiAdapter.analyze_params_cls() is NoAnalyzeParams


@pytest.mark.parametrize("decay", [False, True])
def test_len_analysis_exposes_and_forwards_decay(
    monkeypatch: pytest.MonkeyPatch, decay: bool
) -> None:
    assert SsLenRabiAdapter.analyze_params_cls() is SsLenRabiAnalyzeParams
    assert (
        SsLenRabiAdapter()
        .get_analyze_params(cast(LenRabiResult, object()), cast(Any, object()))
        .decay
        is True
    )

    calls: list[bool] = []
    fit = cast(RabiJointFitResult, object())
    figure = Figure()

    def fake_analyze(
        self: LenRabiExp, result: LenRabiResult, *, decay: bool
    ) -> tuple[RabiJointFitResult, Figure]:
        calls.append(decay)
        return fit, figure

    monkeypatch.setattr(LenRabiExp, "analyze", fake_analyze)
    req = AnalyzeRequest(
        run_result=cast(LenRabiResult, object()),
        analyze_params=SsLenRabiAnalyzeParams(decay=decay),
        md=cast(Any, None),
        ml=cast(Any, None),
        predictor=None,
    )
    out = SsLenRabiAdapter().analyze(req)

    assert calls == [decay]
    assert out.fit_result is fit
    assert out.figure is figure
