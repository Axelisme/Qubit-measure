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
