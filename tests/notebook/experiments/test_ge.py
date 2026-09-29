"""Notebook GE FIT/post publication and retained native figures."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from zcu_tools.experiment.context import QickContext
from zcu_tools.experiment.v2.singleshot.ge import GE_Exp as GECore
from zcu_tools.experiment.v2.singleshot.ge import (
    GE_Result,
    GEAnalysis,
    GEAnalyzeOptions,
)
from zcu_tools.notebook.experiments import GEExp
from zcu_tools.plotting.plots import Plots


def make_result() -> GE_Result:
    rng = np.random.default_rng(83)
    excited = rng.random((2, 6000)) < np.array([0.1, 0.9])[:, None]
    signals = np.asarray(
        np.where(excited, 1 + 0.4j, -1 - 0.4j)
        + 0.2 * (rng.normal(size=excited.shape) + 1j * rng.normal(size=excited.shape)),
        dtype=np.complex128,
    )
    return GE_Result(signals, np.arange(6000), np.array([0, 1]))


def test_post_uses_adopted_fit_source_and_keeps_prior_figures(tmp_path: Path) -> None:
    exp = GEExp(present=False)
    source = make_result()
    primary = exp.analyze(source, backend="pca", length_ratio=0.01)
    fit_record = exp.analysis
    assert fit_record is not None
    assert fit_record.source is source
    assert fit_record.result is primary
    assert fit_record.options == GEAnalyzeOptions(backend="pca", length_ratio=0.01)
    fit_plots = exp.analysis_plots
    post = exp.post_analyze()
    record = exp.post_analysis
    assert record is not None
    assert record.source is source
    assert record.primary is primary
    assert record.primary_options == fit_record.options
    assert record.result is post
    assert record.plots is exp.post_analysis_plots
    assert list(fit_plots) == ["fit"]
    assert list(exp.post_analysis_plots) == ["post"]
    assert np.isfinite(post.confusion.matrix).all()
    fit_plots["fit"].savefig(tmp_path / "fit.png")
    exp.post_analysis_plots["post"].savefig(tmp_path / "post.png")
    assert (tmp_path / "fit.png").stat().st_size > 0
    assert (tmp_path / "post.png").stat().st_size > 0


def test_failed_analysis_preserves_records_and_successful_run_clears_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    exp = GEExp(present=False)
    old = make_result()
    exp.analyze(old, length_ratio=0.01)
    exp.post_analyze()
    primary_record, post_record = exp.analysis, exp.post_analysis
    previous_fit, previous_post = exp.analysis_plots, exp.post_analysis_plots

    def fail_fit(
        self: GECore, result: GE_Result, options: GEAnalyzeOptions, *, plots: Plots
    ) -> GEAnalysis:
        plots.subplots("failed-diagnostic")
        raise ValueError("bad GE calibration")

    monkeypatch.setattr(GECore, "analyze", fail_fit)
    with pytest.raises(ValueError, match="bad GE calibration"):
        exp.analyze(make_result())
    assert exp.analysis is primary_record
    assert exp.post_analysis is post_record

    newer = make_result()

    def run(self: GECore, config: Any, *, context: QickContext) -> GE_Result:
        assert context.soc == "sim-soc"
        context.plots.subplots("measurement")
        return newer

    monkeypatch.setattr(GECore, "run", run)
    assert exp.run("sim-soc", "sim-cfg", cast(Any, None)) is newer
    assert exp.last_result is newer
    assert exp.analysis is None
    assert exp.post_analysis is None
    previous_fit["fit"].savefig(tmp_path / "old-fit.png")
    previous_post["post"].savefig(tmp_path / "old-post.png")
    assert (tmp_path / "old-fit.png").stat().st_size > 0
    assert (tmp_path / "old-post.png").stat().st_size > 0
