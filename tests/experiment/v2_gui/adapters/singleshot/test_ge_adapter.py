from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.figure import Figure
from numpy.typing import NDArray
from zcu_tools.experiment.v2.singleshot import GE_Cfg, GE_Exp
from zcu_tools.experiment.v2.singleshot.ge import GE_Result
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
    CfgSchema,
    CfgSectionValue,
    MetaDictWriteback,
    ModuleRefValue,
    PostAnalyzeRequest,
    RunRequest,
    WritebackRequest,
)
from zcu_tools.meta_tool import MetaDict


def _make_ml() -> MagicMock:
    ml = MagicMock()
    ml.modules = {}
    ml.waveforms = {}
    ml.make_cfg.return_value = MagicMock()
    return ml


def _make_ctx(ml: MagicMock | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.ml = ml or _make_ml()
    ctx.md = MetaDict()
    ctx.qub_name = "Q1"
    return ctx


def _make_req(ml: MagicMock | None = None) -> RunRequest:
    return RunRequest(md=MagicMock(), ml=ml or _make_ml(), soc=None, soccfg=None)


def _lower(schema: CfgSchema, req: RunRequest) -> dict[str, object]:
    return schema.to_raw_dict(None, req.ml)


def _fake_signals(n: int = 16) -> NDArray[np.complex128]:
    """A minimal (2, N) signal stand-in. The adapter tests patch GE_Exp.analyze
    (the domain fitter — covered by tests/utils/fitting), so the actual values
    here are irrelevant; only the shape contract matters."""
    return np.zeros((2, n), dtype=np.complex128)


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

    adapter.build_exp_cfg(raw, _make_req(ml))
    ml.make_cfg.assert_called_once_with(raw, GE_Cfg)


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
    assert isinstance(readout, ModuleRefValue)
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
    """Run the adapter's analyze with GE_Exp.analyze patched to a fixed 4-tuple,
    isolating the adapter's result mapping from the domain fitter's numerics."""
    fig = Figure()
    fit = _fake_fit_result()

    def fake_analyze(self: Any, run_result: Any, backend: str) -> Any:
        del self, run_result
        assert backend == "pca"
        return 0.95, np.zeros((2, 3)), fit, fig

    monkeypatch.setattr(GE_Exp, "analyze", fake_analyze, raising=True)
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
    out = _patched_analyze(adapter, GE_Result(signals=_fake_signals()), monkeypatch)

    assert out.fidelity == pytest.approx(0.95)
    assert out.theta == pytest.approx(0.2)
    assert out.threshold == pytest.approx(0.0)
    assert out.ge_s == pytest.approx(0.3)
    assert out.g_center == -1.0 + 0j
    assert out.e_center == 1.0 + 0j
    assert isinstance(out.figure, Figure)
    # complex centers are skipped from the JSON summary; floats survive.
    summary = out.to_summary_dict()
    assert "fidelity" in summary
    assert "g_center" not in summary


# ---------------------------------------------------------------------------
# Post-analysis (multi-backend discrimination) — patches the domain fitter
# (singleshot_ge_analysis) so only the adapter's param→domain→result mapping is
# under test, mirroring the primary-analyze tests above.
# ---------------------------------------------------------------------------


def _patch_domain_ge(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Patch the domain ``singleshot_ge_analysis`` the adapter calls; capture the
    (angle, backend) the adapter forwards. Returns the capture dict."""
    fig = Figure()
    fit = _fake_fit_result()
    captured: dict[str, Any] = {}

    def fake_domain(
        signals: Any, angle: Any = None, backend: str = "pca", **kwargs: Any
    ) -> Any:
        del signals, kwargs
        captured["angle"] = angle
        captured["backend"] = backend
        return 0.93, np.zeros((2, 3)), fit, fig

    monkeypatch.setattr(
        "zcu_tools.experiment.v2_gui.adapters.singleshot.ge.singleshot_ge_analysis",
        fake_domain,
        raising=True,
    )
    return captured


def _make_post_req(
    params: GEPostAnalyzeParams,
) -> PostAnalyzeRequest[Any, Any, GEPostAnalyzeParams]:
    return PostAnalyzeRequest(
        run_result=GE_Result(signals=_fake_signals()),
        analyze_result=MagicMock(),
        post_analyze_params=params,
        md=MagicMock(),
        ml=_make_ml(),
        predictor=None,
    )


def test_ge_post_analyze_pca_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_domain_ge(monkeypatch)
    out = GEAdapter().post_analyze(_make_post_req(GEPostAnalyzeParams(backend="pca")))

    assert isinstance(out, GEPostAnalyzeResult)
    assert out.backend == "pca"
    assert out.fidelity == pytest.approx(0.93)
    assert out.threshold == pytest.approx(0.0)
    assert out.ge_s == pytest.approx(0.3)
    assert out.g_center == -1.0 + 0j
    assert captured == {"angle": None, "backend": "pca"}
    # complex centers skipped from JSON summary; floats survive.
    summary = out.to_summary_dict()
    assert "fidelity" in summary and "g_center" not in summary


def test_ge_post_analyze_center_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_domain_ge(monkeypatch)
    out = GEAdapter().post_analyze(
        _make_post_req(GEPostAnalyzeParams(backend="center"))
    )

    assert out.backend == "center"
    assert captured["backend"] == "center"
    assert captured["angle"] is None


def test_ge_post_analyze_manual_angle(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_domain_ge(monkeypatch)
    out = GEAdapter().post_analyze(
        _make_post_req(GEPostAnalyzeParams(backend="pca", angle=0.5))
    )

    # angle set → effective backend is "manual"; angle forwarded to the domain.
    assert out.backend == "manual"
    assert captured["angle"] == pytest.approx(0.5)


def test_ge_post_analyze_params_cls_reflects_dataclass() -> None:
    from zcu_tools.gui.app.main.adapter import describe_analyze_params

    assert GEAdapter.post_analyze_params_cls() is GEPostAnalyzeParams
    fields = {f["name"] for f in describe_analyze_params(GEPostAnalyzeParams)}
    assert fields == {"backend", "angle"}


def test_ge_get_post_analyze_params_defaults_to_pca() -> None:
    params = GEAdapter().get_post_analyze_params(MagicMock(), cast(Any, _make_ctx()))
    assert isinstance(params, GEPostAnalyzeParams)
    assert params.backend == "pca"
    assert params.angle is None


def test_ge_writeback_proposes_fid_and_ge_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = GEAdapter()
    result = GE_Result(signals=_fake_signals())
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
    assert targets == {"fid": pytest.approx(0.95), "ge_s": pytest.approx(0.3)}
