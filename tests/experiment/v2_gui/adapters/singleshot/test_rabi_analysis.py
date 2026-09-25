from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from zcu_tools.experiment.v2.singleshot.amp_rabi import AmpRabiResult
from zcu_tools.experiment.v2.singleshot.len_rabi import LenRabiExp, LenRabiResult
from zcu_tools.experiment.v2.singleshot.rabi_fit import RabiJointFitResult
from zcu_tools.experiment.v2_gui.adapters.singleshot.amp_rabi import (
    SsAmpRabiAdapter,
    SsAmpRabiAnalyzeParams,
)
from zcu_tools.experiment.v2_gui.adapters.singleshot.len_rabi import (
    SsLenRabiAdapter,
    SsLenRabiAnalyzeParams,
    SsLenRabiAnalyzeResult,
)
from zcu_tools.gui.app.main.adapter import (
    AnalyzeRequest,
    ExpContext,
    WritebackRequest,
)
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def test_amp_analysis_has_no_decay_option() -> None:
    assert SsAmpRabiAdapter.analyze_params_cls() is SsAmpRabiAnalyzeParams
    assert SsAmpRabiAnalyzeParams().initial_state == "ground"
    assert not hasattr(SsAmpRabiAnalyzeParams(), "decay")
    assert not hasattr(SsAmpRabiAnalyzeParams(), "fit_phase")


@pytest.mark.parametrize("initial_state", ["ground", "excited"])
@pytest.mark.parametrize("decay", [False, True])
@pytest.mark.parametrize("fit_phase", [False, True])
def test_len_analysis_exposes_and_forwards_decay(
    monkeypatch: pytest.MonkeyPatch,
    decay: bool,
    fit_phase: bool,
    initial_state: Literal["ground", "excited"],
) -> None:
    assert SsLenRabiAdapter.analyze_params_cls() is SsLenRabiAnalyzeParams
    assert (
        SsLenRabiAdapter()
        .get_analyze_params(cast(LenRabiResult, object()), cast(Any, object()))
        .decay
        is True
    )

    assert SsLenRabiAnalyzeParams().fit_phase is False
    calls: list[tuple[bool, bool, str]] = []
    fit = cast(RabiJointFitResult, object())
    figure = Figure()

    def fake_analyze(
        self: LenRabiExp,
        result: LenRabiResult,
        *,
        decay: bool,
        fit_phase: bool,
        initial_state: Literal["ground", "excited"],
    ) -> tuple[RabiJointFitResult, Figure]:
        calls.append((decay, fit_phase, initial_state))
        return fit, figure

    monkeypatch.setattr(LenRabiExp, "analyze", fake_analyze)
    req = AnalyzeRequest(
        run_result=cast(LenRabiResult, object()),
        analyze_params=SsLenRabiAnalyzeParams(
            decay=decay, fit_phase=fit_phase, initial_state=initial_state
        ),
        md=cast(Any, None),
        ml=cast(Any, None),
        predictor=None,
    )
    out = SsLenRabiAdapter().analyze(req)

    assert calls == [(decay, fit_phase, initial_state)]
    assert out.fit_result is fit
    assert out.figure is figure


def test_phase_fit_figure_and_calibration_writeback() -> None:
    ctx = ExpContext(MetaDict(), ModuleLibrary(), None, None)
    rng = np.random.default_rng(290)
    lengths = np.linspace(0.2, 1.7, 31)
    p_e = 0.5 - 0.42 * np.cos(4 * np.pi * lengths + 0.6)
    excited = rng.random((lengths.size, 1000)) < p_e[:, None]
    signals = rng.normal(np.where(excited, 1.0, -1.0), 0.18).astype(np.complex128)
    run = LenRabiResult(lengths, np.arange(1000), signals)
    adapter = SsLenRabiAdapter()
    out = adapter.analyze(
        AnalyzeRequest(
            run,
            SsLenRabiAnalyzeParams(decay=False, fit_phase=True),
            ctx.md,
            ctx.ml,
            None,
        )
    )
    try:
        assert out.fit_result.backend.valid
        assert out.fit_result.phase == pytest.approx(0.6, abs=0.12)
        assert "phase=" in out.figure.axes[0].get_title()
        items = adapter.get_writeback_items(WritebackRequest(run, out, ctx))
        assert {item.target_name for item in items} == {
            "g_center",
            "e_center",
            "ge_radius",
            "confusion_matrix",
        }
        failed = replace(
            out.fit_result, backend=replace(out.fit_result.backend, valid=False)
        )
        invalid = SsLenRabiAnalyzeResult(failed, out.figure)
        assert adapter.get_writeback_items(WritebackRequest(run, invalid, ctx)) == []
    finally:
        plt.close(out.figure)


@pytest.mark.parametrize("initial_state", ["ground", "excited"])
def test_amp_and_len_share_joint_analysis_and_writeback(
    initial_state: Literal["ground", "excited"],
) -> None:
    ctx = ExpContext(MetaDict(), ModuleLibrary(), None, None)
    rng = np.random.default_rng(238)
    xs = np.linspace(-0.25, 1.25, 25)
    p_e0 = 0.1 if initial_state == "ground" else 0.9
    p_e = 0.5 + (p_e0 - 0.5) * np.cos(4 * np.pi * xs)
    excited = rng.random((25, 600)) < p_e[:, None]
    raw = rng.normal(np.where(excited, 1.0, -1.0), 0.18).astype(np.complex128)
    amp_run = AmpRabiResult(xs, np.arange(600), raw)
    len_run = LenRabiResult(xs, np.arange(600), raw)
    amp_adapter, len_adapter = SsAmpRabiAdapter(), SsLenRabiAdapter()
    amp_out = amp_adapter.analyze(
        AnalyzeRequest(
            amp_run, SsAmpRabiAnalyzeParams(initial_state), ctx.md, ctx.ml, None
        )
    )
    len_out = len_adapter.analyze(
        AnalyzeRequest(
            len_run,
            SsLenRabiAnalyzeParams(initial_state=initial_state, decay=False),
            ctx.md,
            ctx.ml,
            None,
        )
    )
    try:
        assert amp_out.fit_result.joint_fit.backend.valid
        np.testing.assert_array_equal(
            amp_out.fit_result.joint_fit.backend.values,
            len_out.fit_result.backend.values,
        )
        np.testing.assert_array_equal(
            amp_out.fit_result.joint_fit.fitted_populations,
            len_out.fit_result.fitted_populations,
        )
        amp_items = amp_adapter.get_writeback_items(
            WritebackRequest(amp_run, amp_out, ctx)
        )
        len_items = len_adapter.get_writeback_items(
            WritebackRequest(len_run, len_out, ctx)
        )
        assert amp_items == len_items
        assert len(amp_items) == 4
        failed_joint = replace(
            amp_out.fit_result.joint_fit,
            backend=replace(amp_out.fit_result.joint_fit.backend, valid=False),
        )
        failed = replace(
            amp_out, fit_result=replace(amp_out.fit_result, joint_fit=failed_joint)
        )
        assert (
            amp_adapter.get_writeback_items(WritebackRequest(amp_run, failed, ctx))
            == []
        )
    finally:
        plt.close(amp_out.figure)
        plt.close(len_out.figure)
