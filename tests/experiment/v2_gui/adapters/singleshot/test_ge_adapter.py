from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.figure import Figure
from numpy.typing import NDArray
from zcu_tools.experiment.v2.singleshot import GE_Cfg, GE_Exp
from zcu_tools.experiment.v2.singleshot.ge import GE_Result, GEConfusionResult
from zcu_tools.experiment.v2_gui.adapters.singleshot import GEAdapter
from zcu_tools.experiment.v2_gui.adapters.singleshot.ge import (
    GEAnalyzeResult,
    GEPostAnalyzeParams,
    GEPostAnalyzeResult,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    MetaDictWriteback,
    PostAnalyzeRequest,
    PostWritebackRequest,
    RunRequest,
    WritebackRequest,
)
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionValue,
    ReferenceValue,
)

from ._helpers import make_ctx as _make_ctx
from ._helpers import make_ml as _make_ml


def _make_req(ml: MagicMock | None = None) -> RunRequest:
    return RunRequest(md=MagicMock(), ml=ml or _make_ml(), soc=None, soccfg=None)


def _lower(schema: CfgSchema, req: RunRequest) -> dict[str, object]:
    return schema_to_raw_dict(schema, None, req.ml)


def _fake_signals(n: int = 16) -> NDArray[np.complex128]:
    """A minimal (2, N) signal stand-in. The adapter tests patch GE_Exp.analyze
    (the domain fitter — covered by tests/utils/fitting), so the actual values
    here are irrelevant; only the shape contract matters."""
    return np.zeros((2, n), dtype=np.complex128)


def _fake_result(n: int = 16) -> GE_Result:
    return GE_Result(
        signals=_fake_signals(n),
        shot_indices=np.arange(n, dtype=np.int64),
        prepared_states=np.array([0, 1], dtype=np.int64),
    )


def _fake_fit_result(
    g_center: complex = -1.0 + 0j, e_center: complex = 1.0 + 0j
) -> dict[str, Any]:
    """A GE_FitResult-shaped dict the patched analyze returns."""
    return {
        "ge_params": (0.0, 0.0, 0.3, 0.5, 0.5, 0.5, 0.1),
        "p0_gg": 0.9,
        "p0_ge": 0.1,
        "p0_eg": 0.1,
        "p0_ee": 0.9,
        "s": 0.3,
        "length_ratio_g": 0.1,
        "length_ratio_e": 0.1,
        "theta": 0.2,
        "threshold": 0.0,
        "g_center": g_center,
        "e_center": e_center,
    }


def test_ge_round_trip_delegates_to_make_cfg() -> None:
    ml = _make_ml()
    adapter = GEAdapter()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    modules = cast(dict[str, Any], raw["modules"])
    assert "probe_pulse" in modules
    assert "readout" in modules
    # optional reset / init_pulse disabled (no library entry) → absent
    assert "reset" not in modules
    assert "init_pulse" not in modules
    # shots present; the domain copies it into reps at run.
    assert raw["shots"] == 100000

    cfg = adapter.build_exp_cfg(raw, _make_req(ml))
    assert isinstance(cfg, GE_Cfg)


def test_ge_default_adopts_library_readout() -> None:
    from zcu_tools.meta_tool import ModuleLibrary
    from zcu_tools.program.v2 import ModuleCfgFactory

    ml = ModuleLibrary()
    ml.register_module(
        readout_dpm=ModuleCfgFactory.from_raw(
            {
                "type": "readout/pulse",
                "pulse_cfg": {
                    "waveform": {"style": "const", "length": 1.0},
                    "ch": 1,
                    "nqz": 2,
                    "freq": 6100.0,
                    "gain": 0.2,
                },
                "ro_cfg": {
                    "ro_ch": 2,
                    "ro_freq": 6100.0,
                    "ro_length": 1.0,
                    "trig_offset": 0.5,
                },
            },
            ml=ml,
        )
    )
    schema = GEAdapter().make_default_cfg(_make_ctx(cast(Any, ml)))
    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    readout = modules.fields["readout"]
    assert isinstance(readout, ReferenceValue)
    assert readout.chosen_key == "readout_dpm"


def test_ge_capabilities_is_fit() -> None:
    caps = GEAdapter.capabilities
    assert isinstance(caps, AdapterCapabilities)
    assert caps.analysis is AnalysisMode.FIT
    assert caps.requires_soc is True
    # GE opts into the post-analysis (multi-backend discrimination) layer.
    assert caps.post_analysis is True


def test_ge_run_without_soc_fast_fails() -> None:
    ml = _make_ml()
    adapter = GEAdapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    with pytest.raises(RuntimeError, match="soc is required"):
        adapter.run(_make_req(ml), schema)


def _patched_analyze(
    adapter: GEAdapter, result: GE_Result, monkeypatch: pytest.MonkeyPatch
) -> GEAnalyzeResult:
    """Run the adapter's primary analyze with GE_Exp.analyze patched and
    ``calc_confusion_matrix`` guarded to ensure primary never calculates
    confusion (ticket 05)."""
    fig = Figure()
    fit = _fake_fit_result()
    pops = np.array([[0.9, 0.1], [0.1, 0.9]])

    def fake_analyze(self: Any, run_result: Any, backend: str) -> Any:
        del self, run_result
        assert backend == "pca"
        return 0.95, pops, fit, fig

    def fail_confusion(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("primary analyze must not call calc_confusion_matrix")

    monkeypatch.setattr(GE_Exp, "analyze", fake_analyze, raising=True)
    monkeypatch.setattr(GE_Exp, "calc_confusion_matrix", fail_confusion, raising=True)
    req = AnalyzeRequest(
        run_result=result,
        analyze_params=adapter.get_analyze_params(result, _make_ctx()),
        md=MagicMock(),
        ml=_make_ml(),
        predictor=None,
    )
    return adapter.analyze(req)


def test_ge_analyze_maps_fit_result(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = GEAdapter()
    out = _patched_analyze(adapter, _fake_result(), monkeypatch)

    assert out.fidelity == pytest.approx(0.95)
    assert out.theta == pytest.approx(0.2)
    assert out.threshold == pytest.approx(0.0)
    assert out.ge_s == pytest.approx(0.3)
    assert out.g_center == -1.0 + 0j
    assert out.e_center == 1.0 + 0j
    assert out.init_pops == [[0.9, 0.1], [0.1, 0.9]]
    assert isinstance(out.figure, Figure)
    # Primary must not own radius/matrix — those fields no longer exist.
    assert not hasattr(out, "ge_radius")
    assert not hasattr(out, "confusion")
    # Summary contains only JSON-safe primary fields, no radius/matrix.
    summary = out.to_summary_dict()
    assert "fidelity" in summary
    assert "g_center" not in summary
    assert "ge_radius" not in summary
    assert "confusion" not in summary
    assert "init_pops" in summary


def test_ge_analyze_does_not_call_calc_confusion_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = GEAdapter()
    # _patched_analyze already asserts this via fail_confusion; just run it.
    out = _patched_analyze(adapter, _fake_result(), monkeypatch)
    assert isinstance(out, GEAnalyzeResult)


# ---------------------------------------------------------------------------
# Post-analysis renders a confusion diagnostic from the committed primary fit.
# It re-runs only the radius/matrix calculation, never the GE fitter.
# ---------------------------------------------------------------------------


def _make_post_req() -> PostAnalyzeRequest[Any, GEAnalyzeResult, GEPostAnalyzeParams]:
    analyze_result = GEAnalyzeResult(
        fidelity=0.95,
        theta=0.2,
        threshold=0.0,
        ge_s=0.3,
        g_center=-1.0 + 0.0j,
        e_center=1.0 + 0.0j,
        init_pops=[[0.9, 0.1], [0.1, 0.9]],
        figure=Figure(),
    )
    return PostAnalyzeRequest(
        run_result=_fake_result(),
        analyze_result=analyze_result,
        post_analyze_params=GEPostAnalyzeParams(),
        md=MagicMock(),
        ml=_make_ml(),
        predictor=None,
    )


def test_ge_post_analyze_recalculates_confusion_without_refitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    confusion = np.array([[0.96, 0.04, 0.0], [0.04, 0.96, 0.0], [0.0, 0.0, 1.0]])
    numeric = GEConfusionResult(
        radius=0.37,
        matrix=confusion,
        init_matrix=np.eye(3),
        g_classification=(0.9, 0.1, 0.0),
        e_classification=(0.1, 0.9, 0.0),
        condition_number=1.1,
    )
    figure = Figure()
    captured: dict[str, Any] = {}

    def reject_fit(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("post-analysis re-ran the GE fitter")

    def fake_calc(
        self: Any,
        init_pops: Any,
        g_center: complex,
        e_center: complex,
        sigma: float,
        **kwargs: Any,
    ) -> GEConfusionResult:
        del self
        captured.update(
            init_pops=init_pops,
            g_center=g_center,
            e_center=e_center,
            sigma=sigma,
            kwargs=kwargs,
        )
        return numeric

    def fake_plot(
        self: Any,
        confusion_result: GEConfusionResult,
        g_center: complex,
        e_center: complex,
        **kwargs: Any,
    ) -> Figure:
        del self
        assert confusion_result is numeric
        assert (g_center, e_center) == (-1.0 + 0.0j, 1.0 + 0.0j)
        assert kwargs["result"] is req.run_result
        return figure

    monkeypatch.setattr(
        "zcu_tools.experiment.v2_gui.adapters.singleshot.ge.singleshot_ge_analysis",
        reject_fit,
        raising=False,
    )
    monkeypatch.setattr(GE_Exp, "calc_confusion_matrix", fake_calc, raising=True)
    monkeypatch.setattr(GE_Exp, "plot_confusion_matrix", fake_plot, raising=True)
    req = _make_post_req()

    out = GEAdapter().post_analyze(req)

    assert out.ge_radius == pytest.approx(0.37)
    assert out.confusion == confusion.tolist()
    assert out.figure is figure
    assert np.array_equal(captured["init_pops"], req.analyze_result.init_pops)
    assert captured["g_center"] == req.analyze_result.g_center
    assert captured["e_center"] == req.analyze_result.e_center
    assert captured["sigma"] == pytest.approx(req.analyze_result.ge_s)
    assert captured["kwargs"] == {
        "radius": None,
        "result": req.run_result,
        "consider_other": False,
    }
    assert out.to_summary_dict() == {
        "ge_radius": pytest.approx(0.37),
        "confusion": confusion.tolist(),
    }


def test_ge_post_analyze_params_are_empty() -> None:
    from zcu_tools.gui.app.main.adapter import describe_analyze_params

    assert GEAdapter.post_analyze_params_cls() is GEPostAnalyzeParams
    assert describe_analyze_params(GEPostAnalyzeParams) == []
    params = GEAdapter().get_post_analyze_params(MagicMock(), cast(Any, _make_ctx()))
    assert isinstance(params, GEPostAnalyzeParams)


def test_ge_writeback_proposes_only_primary_four(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = GEAdapter()
    result = _fake_result()
    analyze_result = _patched_analyze(adapter, result, monkeypatch)

    items = adapter.get_writeback_items(
        WritebackRequest(
            run_result=result,
            analyze_result=analyze_result,
            ctx=cast(Any, _make_ctx()),
        )
    )
    for item in items:
        assert isinstance(item, MetaDictWriteback)
    targets = {
        item.target_name: item.proposed_value
        for item in items
        if isinstance(item, MetaDictWriteback)
    }
    assert set(targets) == {"fid", "ge_s", "g_center", "e_center"}
    assert targets["fid"] == pytest.approx(0.95)
    assert targets["ge_s"] == pytest.approx(0.3)
    # The complex centres are proposed verbatim (default fixture: -1+0j / 1+0j).
    assert targets["g_center"] == -1.0 + 0j
    assert targets["e_center"] == 1.0 + 0j
    assert isinstance(targets["g_center"], complex)
    # Must not propose post-owned items.
    assert "ge_radius" not in targets
    assert "confusion_matrix" not in targets


def test_ge_post_writeback_proposes_radius_and_matrix_from_same_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = GEAdapter()
    post_result = GEPostAnalyzeResult(
        ge_radius=0.37,
        confusion=[[0.96, 0.04, 0.0], [0.04, 0.96, 0.0], [0.0, 0.0, 1.0]],
        figure=Figure(),
    )
    analyze_result = GEAnalyzeResult(
        fidelity=0.95,
        theta=0.2,
        threshold=0.0,
        ge_s=0.3,
        g_center=-1.0 + 0j,
        e_center=1.0 + 0j,
        init_pops=[[0.9, 0.1], [0.1, 0.9]],
        figure=Figure(),
    )

    def fail_calc(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("post writeback must not recompute confusion")

    monkeypatch.setattr(GE_Exp, "calc_confusion_matrix", fail_calc, raising=True)
    # Also guard active-context lookup: adapter must use the request's post result,
    # not re-read State. Changing ctx must not affect proposal.
    ctx_a = _make_ctx()
    ctx_b = _make_ctx()
    # Simulate context switch by passing different ctx; proposal must stay same.
    req_a = PostWritebackRequest(
        run_result=_fake_result(),
        analyze_result=analyze_result,
        post_analyze_result=post_result,
        ctx=cast(Any, ctx_a),
    )
    req_b = PostWritebackRequest(
        run_result=_fake_result(),
        analyze_result=analyze_result,
        post_analyze_result=post_result,
        ctx=cast(Any, ctx_b),
    )

    items_a = adapter.get_post_writeback_items(req_a)
    items_b = adapter.get_post_writeback_items(req_b)

    for items in (items_a, items_b):
        assert len(items) == 2
        for item in items:
            assert isinstance(item, MetaDictWriteback)
        targets = {it.target_name: it.proposed_value for it in items}  # type: ignore[attr-defined]
        assert set(targets) == {"ge_radius", "confusion_matrix"}
        assert targets["ge_radius"] == pytest.approx(0.37)
        assert targets["confusion_matrix"] == post_result.confusion
        # Reuses the same post result objects (no recompute).
        assert targets["ge_radius"] == post_result.ge_radius

    # Both proposals identical despite ctx switch.
    targets_a = {it.target_name: it.proposed_value for it in items_a}  # type: ignore[attr-defined]
    targets_b = {it.target_name: it.proposed_value for it in items_b}  # type: ignore[attr-defined]
    assert targets_a == targets_b

    # Summary and proposal share the same radius/matrix values.
    assert post_result.to_summary_dict()["ge_radius"] == pytest.approx(0.37)
    assert post_result.to_summary_dict()["confusion"] == post_result.confusion
