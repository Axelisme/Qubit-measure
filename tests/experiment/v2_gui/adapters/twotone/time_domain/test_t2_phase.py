from __future__ import annotations

from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from numpy.typing import NDArray
from zcu_tools.experiment.v2.twotone.time_domain.t2echo import T2EchoResult
from zcu_tools.experiment.v2.twotone.time_domain.t2ramsey import T2RamseyResult
from zcu_tools.experiment.v2_gui.adapters.twotone.time_domain.t2echo import (
    T2EchoAdapter,
    T2EchoAnalyzeParams,
)
from zcu_tools.experiment.v2_gui.adapters.twotone.time_domain.t2ramsey import (
    T2RamseyAdapter,
    T2RamseyAnalyzeParams,
)
from zcu_tools.gui.app.main.adapter import AnalyzeRequest
from zcu_tools.gui.app.main.adapter.analyze_params import (
    describe_analyze_params,
    reconstruct_params,
)
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _analyze(
    times: NDArray[np.float64],
    signals: NDArray[np.complex128],
    *,
    echo: bool,
    fit_phase: bool,
    fringe: bool = True,
) -> tuple[float, Figure]:
    md, ml = MetaDict(), ModuleLibrary()
    if echo:
        params = T2EchoAnalyzeParams(
            fit_method="fringe" if fringe else "decay", fit_phase=fit_phase
        )
        result = T2EchoAdapter().analyze(
            AnalyzeRequest(T2EchoResult(times, signals), params, md, ml, None)
        )
        return result.t2e, result.figure
    params_r = T2RamseyAnalyzeParams(fit_fringe=fringe, fit_phase=fit_phase)
    result_r = T2RamseyAdapter().analyze(
        AnalyzeRequest(T2RamseyResult(times, signals, 0.0), params_r, md, ml, None)
    )
    return result_r.t2r, result_r.figure


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
    t2, figure = _analyze(times, signals, echo=echo, fit_phase=phase != 0.0)
    try:
        assert t2 == pytest.approx(25, rel=0.001)
        observed, fitted = figure.axes[0].lines[:2]
        np.testing.assert_allclose(
            np.asarray(fitted.get_ydata()), np.asarray(observed.get_ydata()), atol=1e-5
        )
    finally:
        plt.close(figure)


@pytest.mark.parametrize("echo", [False, True])
def test_t2_phase_toggle_changes_shifted_fringe_fit(echo: bool) -> None:
    times = np.linspace(0, 12, 301)
    signals = (
        0.2 + 0.8 * np.exp(-times / 25) * np.cos(2 * np.pi * 0.4 * times + 0.6)
    ).astype(np.complex128)
    residuals = []
    for fit_phase in (False, True):
        _, figure = _analyze(times, signals, echo=echo, fit_phase=fit_phase)
        try:
            observed, fitted = figure.axes[0].lines[:2]
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
            plt.close(figure)
    assert residuals[0] > 1e-3
    assert residuals[1] < 1e-10


@pytest.mark.parametrize("echo", [False, True])
def test_t2_decay_ignores_phase_option(echo: bool) -> None:
    times = np.linspace(0, 12, 301)
    signals = (0.2 + 0.8 * np.exp(-times / 5)).astype(np.complex128)
    for fit_phase in (False, True):
        t2, figure = _analyze(
            times, signals, echo=echo, fit_phase=fit_phase, fringe=False
        )
        try:
            assert t2 == pytest.approx(5, rel=1e-5)
        finally:
            plt.close(figure)


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
