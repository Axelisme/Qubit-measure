from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import AnalyzeRequest
from zcu_tools.gui.app.measure.adapter.analyze_params import (
    describe_analyze_params,
    reconstruct_params,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoCfg, T2EchoResult
from zcu_lab.v2.twotone.time_domain.t2echo.gui import T2EchoAdapter, T2EchoAnalyzeParams
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import T2RamseyCfg, T2RamseyResult
from zcu_lab.v2.twotone.time_domain.t2ramsey.gui import (
    T2RamseyAdapter,
    T2RamseyAnalyzeParams,
)


def _analyze(
    times: NDArray[np.float64],
    signals: NDArray[np.complex128],
    *,
    echo: bool,
    fit_phase: bool,
    fringe: bool = True,
) -> tuple[float, Plots]:
    md, ml = MetaDict(), ModuleLibrary()
    plots = Plots(NonPresentingHost())
    if echo:
        params = T2EchoAnalyzeParams(
            fit_method="fringe" if fringe else "decay", fit_phase=fit_phase
        )
        source = RunRecord[T2EchoCfg, T2EchoResult](
            cfg=None, result=T2EchoResult(times, signals)
        )
        result = T2EchoAdapter().analyze(
            AnalyzeRequest(source, params, md, ml, None), plots=plots
        )
        return result.t2e, plots
    params_r = T2RamseyAnalyzeParams(fit_fringe=fringe, fit_phase=fit_phase)
    source_r = RunRecord[T2RamseyCfg, T2RamseyResult](
        cfg=None, result=T2RamseyResult(times, signals, 0.0)
    )
    result_r = T2RamseyAdapter().analyze(
        AnalyzeRequest(source_r, params_r, md, ml, None), plots=plots
    )
    return result_r.t2r, plots


@pytest.mark.parametrize("echo", [False, True])
@pytest.mark.parametrize("fringe", [False, True])
def test_t2_summary_quality_describes_the_selected_model(
    echo: bool, fringe: bool
) -> None:
    times = np.linspace(0, 12, 101)
    values = 0.2 + 0.8 * np.exp(-times / 5)
    if fringe:
        values = 0.2 + 0.8 * np.exp(-times / 25) * np.cos(2 * np.pi * 0.4 * times + 0.6)
    values += np.where(np.arange(times.size) % 2 == 0, 0.01, -0.01)
    if echo:
        values[0] = 1000  # Echo's existing optimizer excludes the first sample.
    signals = values.astype(np.complex128)
    plots = Plots(NonPresentingHost())
    md, ml = MetaDict(), ModuleLibrary()
    try:
        if echo:
            source_e = RunRecord[T2EchoCfg, T2EchoResult](
                cfg=None, result=T2EchoResult(times, signals)
            )
            result_e = T2EchoAdapter().analyze(
                AnalyzeRequest(
                    source_e,
                    T2EchoAnalyzeParams(
                        fit_method="fringe" if fringe else "decay", fit_phase=True
                    ),
                    md,
                    ml,
                    None,
                ),
                plots=plots,
            )
            summary = result_e.to_summary_dict()
        else:
            source_r = RunRecord[T2RamseyCfg, T2RamseyResult](
                cfg=None, result=T2RamseyResult(times, signals, 0.0)
            )
            result_r = T2RamseyAdapter().analyze(
                AnalyzeRequest(
                    source_r,
                    T2RamseyAnalyzeParams(fit_fringe=fringe, fit_phase=True),
                    md,
                    ml,
                    None,
                ),
                plots=plots,
            )
            summary = result_r.to_summary_dict()
        quality = json.loads(json.dumps(summary, allow_nan=False))["fit_quality"]["fit"]
        assert 0.99 < quality["r2"] < 1.0
        assert 0.005 < quality["normalized_residual_rms"] < 0.03
        assert set(quality["relative_parameter_errors"]) == (
            {"y0", "yscale", "freq", "phase", "decay_time"}
            if fringe
            else {"y0", "yscale", "decay_time"}
        )
    finally:
        plots.finish()
        plots.release()


@pytest.mark.parametrize("echo", [False, True])
@pytest.mark.parametrize("phase", [0.0, 180.0, 35.0, -35.0, 215.0])
def test_t2_phase_recovers_fringe_and_coherence(echo: bool, phase: float) -> None:
    times = np.linspace(0, 12, 301)
    signals = (
        0.2
        + 0.8
        * np.exp(-times / 25)
        * np.cos(2 * np.pi * 0.4 * times + np.radians(phase))
    ).astype(np.complex128)
    t2, plots = _analyze(times, signals, echo=echo, fit_phase=phase != 0.0)
    try:
        assert t2 == pytest.approx(25, rel=0.001)
        observed, fitted = plots["fit"].axes[0].lines[:2]
        np.testing.assert_allclose(
            np.asarray(fitted.get_ydata()), np.asarray(observed.get_ydata()), atol=1e-5
        )
    finally:
        plots.finish()
        plots.release()


@pytest.mark.parametrize("echo", [False, True])
def test_t2_phase_toggle_changes_shifted_fringe_fit(echo: bool) -> None:
    times = np.linspace(0, 12, 301)
    signals = (
        0.2 + 0.8 * np.exp(-times / 25) * np.cos(2 * np.pi * 0.4 * times + 0.6)
    ).astype(np.complex128)
    residuals = []
    for fit_phase in (False, True):
        _, plots = _analyze(times, signals, echo=echo, fit_phase=fit_phase)
        try:
            observed, fitted = plots["fit"].axes[0].lines[:2]
            residuals.append(
                float(
                    np.mean(
                        (
                            np.asarray(observed.get_ydata())
                            - np.asarray(fitted.get_ydata())
                        )
                        ** 2
                    )
                )
            )
        finally:
            plots.finish()
            plots.release()
    assert residuals[0] > 1e-3
    assert residuals[1] < 1e-10


@pytest.mark.parametrize("echo", [False, True])
def test_t2_decay_ignores_phase_option(echo: bool) -> None:
    times = np.linspace(0, 12, 301)
    signals = (0.2 + 0.8 * np.exp(-times / 5)).astype(np.complex128)
    for fit_phase in (False, True):
        t2, plots = _analyze(
            times, signals, echo=echo, fit_phase=fit_phase, fringe=False
        )
        try:
            assert t2 == pytest.approx(5, rel=1e-5)
        finally:
            plots.finish()
            plots.release()


@pytest.mark.parametrize("echo", [False, True])
def test_t2_phase_form_and_wire_contract(echo: bool) -> None:
    cls = T2EchoAnalyzeParams if echo else T2RamseyAnalyzeParams
    spec = next(
        field for field in describe_analyze_params(cls) if field["name"] == "fit_phase"
    )
    assert spec == {
        "name": "fit_phase",
        "type": "bool",
        "label": "Fit phase offset (fringe only)",
        "default": False,
    }
    values = asdict(cls())
    assert reconstruct_params(cls, values).fit_phase is False
    values["fit_phase"] = True
    assert reconstruct_params(cls, values).fit_phase is True
    values["fit_phase"] = "true"
    with pytest.raises(RuntimeError, match="expects bool"):
        reconstruct_params(cls, values)
