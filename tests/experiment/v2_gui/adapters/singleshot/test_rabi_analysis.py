from __future__ import annotations

from dataclasses import fields
from typing import Any, Literal, cast

import pytest
from matplotlib.figure import Figure
from zcu_tools.experiment.v2.singleshot.len_rabi import LenRabiExp, LenRabiResult
from zcu_tools.experiment.v2.singleshot.rabi_fit import RabiJointFitResult
from zcu_tools.experiment.v2_gui.adapters.singleshot.amp_rabi import (
    SsAmpRabiAdapter,
    SsAmpRabiAnalyzeParams,
)
from zcu_tools.experiment.v2_gui.adapters.singleshot.len_rabi import (
    SsLenRabiAdapter,
    SsLenRabiAnalyzeParams,
)
from zcu_tools.gui.app.main.adapter import AnalyzeRequest


def test_amp_analysis_has_no_decay_option() -> None:
    assert SsAmpRabiAdapter.analyze_params_cls() is SsAmpRabiAnalyzeParams
    assert "decay" not in {field.name for field in fields(SsAmpRabiAnalyzeParams)}
    assert SsAmpRabiAnalyzeParams().initial_state == "ground"


@pytest.mark.parametrize("initial_state", ["ground", "excited"])
@pytest.mark.parametrize("decay", [False, True])
def test_len_analysis_exposes_and_forwards_decay(
    monkeypatch: pytest.MonkeyPatch,
    decay: bool,
    initial_state: Literal["ground", "excited"],
) -> None:
    assert SsLenRabiAdapter.analyze_params_cls() is SsLenRabiAnalyzeParams
    assert (
        SsLenRabiAdapter()
        .get_analyze_params(cast(LenRabiResult, object()), cast(Any, object()))
        .decay
        is True
    )

    calls: list[tuple[bool, str]] = []
    fit = cast(RabiJointFitResult, object())
    figure = Figure()

    def fake_analyze(
        self: LenRabiExp,
        result: LenRabiResult,
        *,
        decay: bool,
        initial_state: Literal["ground", "excited"],
    ) -> tuple[RabiJointFitResult, Figure]:
        calls.append((decay, initial_state))
        return fit, figure

    monkeypatch.setattr(LenRabiExp, "analyze", fake_analyze)
    req = AnalyzeRequest(
        run_result=cast(LenRabiResult, object()),
        analyze_params=SsLenRabiAnalyzeParams(decay=decay, initial_state=initial_state),
        md=cast(Any, None),
        ml=cast(Any, None),
        predictor=None,
    )
    out = SsLenRabiAdapter().analyze(req)

    assert calls == [(decay, initial_state)]
    assert out.fit_result is fit
    assert out.figure is figure
