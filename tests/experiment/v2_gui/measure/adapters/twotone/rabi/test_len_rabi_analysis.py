from __future__ import annotations

import json

import numpy as np
import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.twotone.rabi.len_rabi import LenRabiCfg, LenRabiResult
from zcu_tools.experiment.v2_gui.measure.adapters.twotone.rabi.len_rabi import (
    LenRabiAdapter,
    LenRabiAnalyzeParams,
)
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    SessionEnv,
    WritebackRequest,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.context import MetaDict, ModuleLibrary


@pytest.mark.parametrize("decay", [False, True])
def test_len_rabi_quality_uses_only_unmasked_samples(decay: bool) -> None:
    lengths = np.linspace(0.15, 4.15, 101)
    envelope = np.exp(-lengths / 20) if decay else np.ones_like(lengths)
    values = 0.2 + envelope * np.cos(np.pi * lengths + 0.6)
    values += np.where(np.arange(lengths.size) % 2 == 0, 0.01, -0.01)
    values[[2, 9]] = np.nan
    source = RunRecord[LenRabiCfg, LenRabiResult](
        cfg=None, result=LenRabiResult(lengths, values.astype(np.complex128))
    )
    plots = Plots(NonPresentingHost())
    try:
        answer = LenRabiAdapter().analyze(
            AnalyzeRequest(
                source,
                LenRabiAnalyzeParams(decay=decay, fit_phase=True),
                MetaDict(),
                ModuleLibrary(),
                None,
            ),
            plots=plots,
        )
        summary = json.loads(json.dumps(answer.to_summary_dict(), allow_nan=False))
        quality = summary["fit_quality"]["fit"]
        assert 0.99 < quality["r2"] < 1.0
        assert 0.004 < quality["normalized_residual_rms"] < 0.02
        assert set(quality["relative_parameter_errors"]) == (
            {"y0", "yscale", "freq", "phase", "decay_time"}
            if decay
            else {"y0", "yscale", "freq", "phase"}
        )
    finally:
        plots.finish()
        plots.release()


@pytest.mark.parametrize("decay", [False, True])
@pytest.mark.parametrize("phase", [-30.0, 0.0, 30.0, 150.0, 180.0, 210.0])
def test_len_rabi_phase_controls_calibrated_lengths(decay: bool, phase: float) -> None:
    ctx = SessionEnv(MetaDict(), ModuleLibrary(), None, None)
    lengths = np.linspace(0.15, 4.15, 301)
    freq = 0.5
    envelope = np.exp(-lengths / 20) if decay else np.ones_like(lengths)
    signals = 0.2 + envelope * np.cos(2 * np.pi * freq * lengths + np.radians(phase))
    run = RunRecord[LenRabiCfg, LenRabiResult](
        cfg=None, result=LenRabiResult(lengths, signals.astype(np.complex128))
    )
    adapter = LenRabiAdapter()
    # The fixed model must handle either IQ polarity even with no zero sample.
    params = LenRabiAnalyzeParams(decay=decay, fit_phase=phase not in (0.0, 180.0))
    plots = Plots(NonPresentingHost())
    result = adapter.analyze(
        AnalyzeRequest(run, params, ctx.md, ctx.ml, None), plots=plots
    )
    try:
        offset = (phase if phase < 90 else phase - 180) / 360
        assert result.rabi_f == pytest.approx(freq, rel=1e-4)
        assert result.pi_len == pytest.approx((0.5 - offset) / freq, abs=1e-4)
        assert result.pi2_len == pytest.approx((0.25 - offset) / freq, abs=1e-4)
        assert result.pi_len_err >= 0
        assert result.pi2_len_err >= 0
        items = adapter.get_writeback_items(WritebackRequest(run, result, ctx))
        assert {item.target_name for item in items} == {"pi_len", "pi2_len", "rabi_f"}
    finally:
        plots.finish()
        plots.release()


@pytest.mark.parametrize("decay", [False, True])
def test_default_len_rabi_fit_constrains_phase(decay: bool) -> None:
    ctx = SessionEnv(MetaDict(), ModuleLibrary(), None, None)
    lengths = np.linspace(0.1, 4.1, 301)
    envelope = np.exp(-lengths / 8) if decay else np.ones_like(lengths)
    signals = envelope * np.cos(np.pi * lengths + 0.6)
    run = RunRecord[LenRabiCfg, LenRabiResult](
        cfg=None, result=LenRabiResult(lengths, signals.astype(np.complex128))
    )
    params = LenRabiAnalyzeParams(decay=decay)
    assert params.fit_phase is False
    plots = Plots(NonPresentingHost())
    result = LenRabiAdapter().analyze(
        AnalyzeRequest(run, params, ctx.md, ctx.ml, None), plots=plots
    )
    try:
        assert result.pi_len == pytest.approx(0.5 / result.rabi_f)
        assert result.pi2_len == pytest.approx(0.25 / result.rabi_f)
    finally:
        plots.finish()
        plots.release()
