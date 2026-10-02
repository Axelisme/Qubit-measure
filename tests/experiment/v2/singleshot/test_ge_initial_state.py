"""GE FIT/post keeps acquisition order separate from prepared-state labels."""

from typing import Any, Literal, cast

import numpy as np
import pytest
from zcu_tools.analysis.fitting.singleshot import calc_population_pdf
from zcu_tools.experiment import RunRecord
from zcu_tools.experiment.v2.singleshot.ge import (
    GE_Cfg,
    GE_Exp,
    GE_Result,
    GEAnalyzeOptions,
    GEPostAnalyzeOptions,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots


@pytest.mark.parametrize("backend", ["pca", "center"])
def test_ge_initial_state_keeps_fit_and_post_calibration_consistent(
    backend: Literal["pca", "center"],
) -> None:
    rng = np.random.default_rng(83)
    excited = rng.random((2, 6000)) < np.array([0.1, 0.9])[:, None]
    signals = np.asarray(
        np.where(excited, 1 + 0.4j, -1 - 0.4j)
        + 0.2 * (rng.normal(size=excited.shape) + 1j * rng.normal(size=excited.shape)),
        dtype=np.complex128,
    )
    primary_results = []
    post_results = []
    post_titles = []
    for initial_state in ("ground", "excited"):
        state = initial_state
        raw = signals.copy() if state == "ground" else signals[::-1].copy()
        before = raw.copy()
        source = RunRecord[GE_Cfg, GE_Result](
            None, GE_Result(raw, np.arange(6000), np.array([0, 1]))
        )
        fit_plots = Plots(NonPresentingHost())
        primary = GE_Exp().analyze(
            source,
            GEAnalyzeOptions(initial_state=state, backend=backend, length_ratio=0.01),
            plots=fit_plots,
        )
        fit_plots.finish()
        post_plots = Plots(NonPresentingHost())
        post = GE_Exp().post_analyze(
            source, primary, GEPostAnalyzeOptions(), plots=post_plots
        )
        post_plots.finish()
        primary_results.append(primary)
        post_results.append(post.confusion)
        post_titles.append([ax.get_title() for ax in post_plots["post"].axes])
        assert list(fit_plots) == ["fit"]
        assert list(post_plots) == ["post"]
        assert primary.initial_state == state
        assert abs(primary.g_center - (-1 - 0.4j)) < 0.1
        assert abs(primary.e_center - (1 + 0.4j)) < 0.1
        assert primary.init_pops[0, 0] > 0.8
        assert primary.init_pops[1, 1] > 0.8
        np.testing.assert_allclose(post.confusion.matrix, np.eye(3), atol=0.06)
        np.testing.assert_array_equal(raw, before)

    ground, excited_result = primary_results
    assert ground.g_center == pytest.approx(excited_result.g_center)
    assert ground.e_center == pytest.approx(excited_result.e_center)
    assert ground.fidelity == pytest.approx(excited_result.fidelity)
    assert post_results[0].radius == pytest.approx(post_results[1].radius)
    np.testing.assert_allclose(post_results[0].matrix, post_results[1].matrix)
    assert post_titles[0] == post_titles[1]


@pytest.mark.parametrize("backend", ["pca", "center"])
def test_strong_transition_data_relabels_both_populations_and_confusion(
    backend: Literal["pca", "center"],
) -> None:
    rng = np.random.default_rng(832)
    xs = np.linspace(-4, 4, 401)
    pdfs = [
        calc_population_pdf(xs, -1, 1, 0.3, pg, pe, 0.15, 1.2)
        for pg, pe in [(0.9, 0.1), (0.1, 0.9)]
    ]
    shots = np.stack([rng.choice(xs, size=15000, p=pdf / pdf.sum()) for pdf in pdfs])
    signals = (shots + 0.3j * rng.normal(size=shots.shape)) * np.exp(0.4j)
    source = RunRecord[GE_Cfg, GE_Result](
        None, GE_Result(signals, np.arange(shots.shape[1]), np.array([0, 1]))
    )
    outputs = []
    posts = []
    for state in ("ground", "excited"):
        primary = GE_Exp().analyze(
            source,
            GEAnalyzeOptions(initial_state=cast(Any, state), backend=backend),
            plots=Plots(NonPresentingHost()),
        )
        primary.validate_calibration()
        outputs.append(primary)
        posts.append(
            GE_Exp()
            .post_analyze(
                source,
                primary,
                GEPostAnalyzeOptions(),
                plots=Plots(NonPresentingHost()),
            )
            .confusion
        )
    ground, excited = outputs
    assert ground.g_center == pytest.approx(excited.e_center, abs=1e-5)
    assert ground.e_center == pytest.approx(excited.g_center, abs=1e-5)
    assert ground.ge_s == pytest.approx(excited.ge_s, abs=1e-5)
    assert ground.fidelity == pytest.approx(excited.fidelity)
    np.testing.assert_allclose(
        ground.init_pops, excited.init_pops[::-1, ::-1], atol=1e-7
    )
    order = [1, 0, 2]
    np.testing.assert_allclose(
        posts[0].matrix,
        posts[1].matrix[np.ix_(order, order)],
        atol=1e-7,
    )
