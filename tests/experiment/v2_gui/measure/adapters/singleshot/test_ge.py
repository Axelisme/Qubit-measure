"""GE adapter translates FIT/post into named plots and separate proposals."""

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.singleshot.ge import GE_Cfg, GE_Result
from zcu_tools.experiment.v2_gui.measure.adapters.singleshot.ge import (
    GEAdapter,
    GEAnalyzeParams,
    GEPostAnalyzeParams,
)
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    PostAnalyzeRequest,
    PostWritebackRequest,
    WritebackRequest,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots


def test_primary_post_and_writebacks_use_adopted_calibration_not_edited_form() -> None:
    rng = np.random.default_rng(83)
    excited = rng.random((2, 6000)) < np.array([0.1, 0.9])[:, None]
    signals = np.asarray(
        np.where(excited, 1 + 0.4j, -1 - 0.4j)
        + 0.2 * (rng.normal(size=excited.shape) + 1j * rng.normal(size=excited.shape)),
        dtype=np.complex128,
    )
    # Probe-off/on acquisition rows are reversed for predominantly excited init.
    source = RunRecord[GE_Cfg, GE_Result](
        None, GE_Result(signals[::-1].copy(), np.arange(6000), np.array([0, 1]))
    )
    adapter = GEAdapter()
    params = GEAnalyzeParams(initial_state="excited", length_ratio=0.01)
    fit_plots = Plots(NonPresentingHost())
    primary = adapter.analyze(
        AnalyzeRequest(source, params, cast(Any, None), cast(Any, None), None),
        plots=fit_plots,
    )
    fit_plots.finish()
    assert primary.initial_state == "excited"
    assert primary.fidelity > 0.8
    assert primary.g_center == pytest.approx(-1 - 0.4j, abs=0.1)
    assert primary.e_center == pytest.approx(1 + 0.4j, abs=0.1)
    assert list(fit_plots) == ["fit"]
    assert primary.to_summary_dict()["init_pops"] == primary.init_pops.tolist()
    assert primary.to_summary_dict()["initial_state"] == "excited"

    params.initial_state = "ground"
    post_plots = Plots(NonPresentingHost())
    post = adapter.post_analyze(
        PostAnalyzeRequest(
            source,
            primary,
            GEPostAnalyzeParams(),
            cast(Any, None),
            cast(Any, None),
            None,
        ),
        plots=post_plots,
    )
    post_plots.finish()
    assert list(post_plots) == ["post"]
    np.testing.assert_allclose(post.confusion, np.eye(3), atol=0.06)
    primary_items = adapter.get_writeback_items(
        WritebackRequest(source, primary, cast(Any, None))
    )
    post_items = adapter.get_post_writeback_items(
        PostWritebackRequest(source, primary, post, cast(Any, None))
    )
    assert {item.target_name for item in primary_items} == {
        "fid",
        "ge_s",
        "g_center",
        "e_center",
    }
    assert {item.target_name for item in post_items} == {
        "ge_radius",
        "confusion_matrix",
    }
    invalid = replace(primary, init_pops=np.array([[0.8, 0.4], [0.1, 0.9]]))
    with pytest.raises(ValueError, match="Invalid GE calibration"):
        adapter.get_writeback_items(WritebackRequest(source, invalid, cast(Any, None)))
