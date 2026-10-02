"""Observable contracts of the stateless ordinary T1 analysis."""

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.twotone.time_domain.t1 import (
    T1AnalyzeOptions,
    T1Cfg,
    T1Exp,
    T1Result,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots


@pytest.mark.parametrize("dual_exp", [False, True])
def test_analysis_fits_decay_and_keeps_native_saveable_figure(
    tmp_path: Path, dual_exp: bool
) -> None:
    times = np.linspace(0, 100, 101)
    signal = 0.2 + 0.8 * np.exp(-times / 20)
    if dual_exp:
        signal += 0.4 * np.exp(-times / 5)
    source: RunRecord[T1Cfg, T1Result] = RunRecord(
        cfg=None, result=T1Result(times, signal.astype(np.complex128))
    )
    plots = Plots(NonPresentingHost())
    analysis = T1Exp().analyze(
        source, T1AnalyzeOptions(dual_exp=dual_exp, skip=2), plots=plots
    )
    plots.finish()

    if dual_exp:
        assert analysis.t1b is not None
        assert sorted([analysis.t1, analysis.t1b]) == pytest.approx([5, 20], rel=0.02)
    else:
        assert analysis.t1 == pytest.approx(20, rel=0.01)
        assert analysis.t1b is None
    assert np.isfinite(analysis.t1_err)
    figure = plots["fit"]
    np.testing.assert_array_equal(figure.axes[0].lines[0].get_xdata(), times[2:])
    np.testing.assert_allclose(
        np.asarray(figure.axes[0].lines[1].get_ydata()), signal[2:]
    )
    plots.release()
    path = tmp_path / "fit.png"
    figure.savefig(path)
    assert path.stat().st_size > 0


def test_reusing_core_uses_only_explicit_analysis_result() -> None:
    times = np.linspace(0, 80, 81)
    core = T1Exp()
    for expected in [12.0, 24.0, 12.0]:
        source: RunRecord[T1Cfg, T1Result] = RunRecord(
            cfg=None,
            result=T1Result(times, np.exp(-times / expected).astype(np.complex128)),
        )
        plots = Plots(NonPresentingHost())
        fitted = core.analyze(source, T1AnalyzeOptions(), plots=plots)
        plots.finish()
        assert fitted.t1 == pytest.approx(expected, rel=0.01)
        plots.release()
