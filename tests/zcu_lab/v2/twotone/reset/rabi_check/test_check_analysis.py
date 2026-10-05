from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalysisMode,
    AnalyzeRequest,
    NoAnalyzeParams,
)
from zcu_tools.gui.cfg import CfgSectionSpec
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from zcu_lab.v2.twotone.reset.rabi_check.core import RabiCheckCfg, RabiCheckResult
from zcu_lab.v2.twotone.reset.rabi_check.gui import (
    RabiCheckAdapter,
    RabiCheckAnalyzeResult,
)


def test_reset_check_adapter_exposes_fit_summary() -> None:
    assert RabiCheckAdapter.capabilities.analysis is AnalysisMode.FIT
    assert RabiCheckAdapter.analyze_params_cls() is NoAnalyzeParams

    gains = np.linspace(0, 1, 101)
    angle = 2 * np.pi * 2 * gains
    result = RabiCheckResult(
        gains,
        np.array(
            [
                np.cos(angle),
                0.1 * np.cos(angle),
                0.8 * np.cos(angle + 0.2),
            ],
            dtype=np.complex128,
        ),
    )
    req = AnalyzeRequest(
        run_result=RunRecord[RabiCheckCfg, RabiCheckResult](cfg=None, result=result),
        analyze_params=NoAnalyzeParams(),
        md=cast(Any, None),
        ml=cast(Any, None),
        predictor=None,
    )
    plots = Plots(NonPresentingHost())
    try:
        out = RabiCheckAdapter().analyze(req, plots=plots)
        assert isinstance(out, RabiCheckAnalyzeResult)
        summary = out.to_summary_dict()
        assert summary["before_amplitude"] == pytest.approx(1.0)
        assert summary["after_amplitude"] == pytest.approx(0.8)
        assert summary["relative_contrast"] == pytest.approx(0.8)
        assert summary["reset_residual_amplitude"] == pytest.approx(0.1)
        assert summary["phase_difference_deg"] == pytest.approx(
            np.degrees(0.2), abs=1e-4
        )
        assert summary["frequency_cycles_per_gain"] == pytest.approx(2.0)
        assert "figure" not in summary
        assert "fidelity" not in summary
        assert len(plots["fit"].axes) == 2
    finally:
        plots.finish(present=False)
        plots.release()


def test_reset_check_cfg_has_one_rabi_pulse_field() -> None:
    modules = RabiCheckAdapter.cfg_definition().spec.fields["modules"]
    assert isinstance(modules, CfgSectionSpec)
    assert "rabi_pulse" in modules.fields
    assert "pi_pulse" not in modules.fields
