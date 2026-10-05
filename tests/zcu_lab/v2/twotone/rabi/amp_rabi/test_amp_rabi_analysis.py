"""Amplitude Rabi summary carries the actual skipped-data fit quality."""

import json

import numpy as np
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import AnalyzeRequest
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from zcu_lab.v2.twotone.rabi.amp_rabi.core import AmpRabiCfg, AmpRabiResult
from zcu_lab.v2.twotone.rabi.amp_rabi.gui import AmpRabiAdapter, AmpRabiAnalyzeParams


def test_amp_rabi_quality_uses_only_samples_after_skip() -> None:
    gains = np.linspace(0, 2, 101)
    values = 0.2 + np.cos(2 * np.pi * gains)
    values += np.where(np.arange(gains.size) % 2 == 0, 0.01, -0.01)
    values[:5] = 1000
    source = RunRecord[AmpRabiCfg, AmpRabiResult](
        cfg=None, result=AmpRabiResult(gains, values.astype(np.complex128))
    )
    plots = Plots(NonPresentingHost())
    try:
        answer = AmpRabiAdapter().analyze(
            AnalyzeRequest(
                source, AmpRabiAnalyzeParams(skip=5), MetaDict(), ModuleLibrary(), None
            ),
            plots=plots,
        )
        summary = json.loads(json.dumps(answer.to_summary_dict(), allow_nan=False))
        quality = summary["fit_quality"]["fit"]
        assert 0.99 < quality["r2"] < 1.0
        assert 0.003 < quality["normalized_residual_rms"] < 0.02
        assert set(quality["relative_parameter_errors"]) == {
            "y0",
            "yscale",
            "freq",
            "phase",
        }
    finally:
        plots.finish()
        plots.release()
