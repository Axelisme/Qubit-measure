from dataclasses import replace
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from zcu_tools.experiment.v2.singleshot.ge import GE_Exp, GE_Result
from zcu_tools.experiment.v2_gui.adapters.singleshot.ge import (
    GEAdapter,
    GEAnalyzeParams,
    GEPostAnalyzeParams,
)
from zcu_tools.gui.app.main.adapter import AnalyzeRequest, PostAnalyzeRequest
from zcu_tools.utils.fitting.singleshot import calc_population_pdf


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
    adapter = GEAdapter()
    primary_results = []
    post_results = []
    for initial_state in ("ground", "excited"):
        state = cast(Literal["ground", "excited"], initial_state)
        raw = signals.copy() if state == "ground" else signals[::-1].copy()
        original = raw.copy()
        result = GE_Result(raw, np.arange(6000), np.array([0, 1]))
        params = GEAnalyzeParams(
            initial_state=state, backend=backend, length_ratio=0.01
        )
        primary = adapter.analyze(
            AnalyzeRequest(result, params, cast(Any, None), cast(Any, None), None)
        )
        primary_results.append(primary)
        # Editing pending parameters must not alter the completed fit's semantics.
        params.initial_state = "excited" if state == "ground" else "ground"
        post = adapter.post_analyze(
            PostAnalyzeRequest(
                result,
                primary,
                GEPostAnalyzeParams(),
                cast(Any, None),
                cast(Any, None),
                None,
            )
        )
        post_results.append(post)
        assert primary.initial_state == state
        assert abs(primary.g_center - (-1 - 0.4j)) < 0.1
        assert abs(primary.e_center - (1 + 0.4j)) < 0.1
        assert primary.init_pops[0][0] > 0.8
        assert primary.init_pops[1][1] > 0.8
        np.testing.assert_allclose(post.confusion, np.eye(3), atol=0.06)
        np.testing.assert_array_equal(raw, original)
        plt.close(primary.figure)
        plt.close(post.figure)

    ground, excited_result = primary_results
    assert ground.g_center == pytest.approx(excited_result.g_center)
    assert ground.e_center == pytest.approx(excited_result.e_center)
    assert ground.fidelity == pytest.approx(excited_result.fidelity)
    assert post_results[0].ge_radius == pytest.approx(post_results[1].ge_radius)
    np.testing.assert_allclose(post_results[0].confusion, post_results[1].confusion)
    assert [ax.get_title() for ax in post_results[0].figure.axes] == [
        ax.get_title() for ax in post_results[1].figure.axes
    ]


def test_ge_rejects_unknown_initial_state() -> None:
    result = GE_Result(
        np.zeros((2, 2), dtype=np.complex128), np.arange(2), np.array([0, 1])
    )
    with pytest.raises(ValueError, match="Unknown initial state"):
        GE_Exp().analyze(result, initial_state=cast(Any, "unknown"))


@pytest.mark.parametrize("backend", ["pca", "center"])
def test_same_strong_transition_data_supports_both_initial_states(
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
    result = GE_Result(signals, np.arange(shots.shape[1]), np.array([0, 1]))
    adapter = GEAdapter()
    outputs = []
    posts = []
    for state in ("ground", "excited"):
        primary = adapter.analyze(
            AnalyzeRequest(
                result,
                GEAnalyzeParams(initial_state=cast(Any, state), backend=backend),
                cast(Any, None),
                cast(Any, None),
                None,
            )
        )
        primary.validate_calibration()
        outputs.append(primary)
        post = adapter.post_analyze(
            PostAnalyzeRequest(
                result,
                primary,
                GEPostAnalyzeParams(),
                cast(Any, None),
                cast(Any, None),
                None,
            )
        )
        posts.append(post)
        plt.close(primary.figure)
        plt.close(post.figure)
    ground, excited = outputs
    assert ground.g_center == pytest.approx(excited.e_center, abs=1e-5)
    assert ground.e_center == pytest.approx(excited.g_center, abs=1e-5)
    assert ground.ge_s == pytest.approx(excited.ge_s, abs=1e-5)
    assert ground.fidelity == pytest.approx(excited.fidelity)
    np.testing.assert_allclose(
        ground.init_pops, np.asarray(excited.init_pops)[::-1, ::-1], atol=1e-7
    )
    order = [1, 0, 2]
    np.testing.assert_allclose(
        posts[0].confusion,
        np.asarray(posts[1].confusion)[np.ix_(order, order)],
        atol=1e-7,
    )
    # Invalid cached/fabricated results cannot become writeback proposals.
    invalid = replace(ground, init_pops=[[0.8, 0.4], [0.1, 0.9]])
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="Invalid GE calibration"):
        adapter.get_writeback_items(cast(Any, SimpleNamespace(analyze_result=invalid)))
