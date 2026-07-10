"""Tests for the len_rabi, t1, and t1_tone singleshot adapters.

Covers:
- cfg_spec structure + validate_schema(make_default_cfg(), ml) + filename stem
- run override: md has GE centres → correct domain run call (mock)
- run override: md missing centres → fast-fail
- uniform kwarg is correctly extracted from raw cfg and forwarded (t1 / t1_tone)
- uniform pop-before-lowering (not in exp cfg)
- analyze figure-only result (len_rabi, t1)
- analyze numeric result + writeback value (t1_tone)
- confusion_matrix forwarded / absent at analyze time
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from zcu_tools.experiment.v2.singleshot.len_rabi import LenRabiCfg, LenRabiExp
from zcu_tools.experiment.v2.singleshot.t1.t1 import T1Cfg, T1Exp
from zcu_tools.experiment.v2.singleshot.t1.t1_with_tone import (
    T1WithToneCfg,
    T1WithToneExp,
)
from zcu_tools.experiment.v2_gui.adapters.singleshot import (
    SsLenRabiAdapter,
    SsT1Adapter,
    SsT1ToneAdapter,
)
from zcu_tools.experiment.v2_gui.adapters.singleshot.len_rabi import (
    SsLenRabiAnalyzeResult,
)
from zcu_tools.experiment.v2_gui.adapters.singleshot.t1 import SsT1AnalyzeResult
from zcu_tools.experiment.v2_gui.adapters.singleshot.t1_tone import (
    SsT1ToneAnalyzeResult,
    SsT1ToneRunResult,
)
from zcu_tools.gui.app.main.adapter import (
    MetaDictWriteback,
    WritebackRequest,
)
from zcu_tools.gui.app.main.adapter.lowering import (
    schema_to_raw_dict,
    validate_schema,
)
from zcu_tools.gui.cfg import (
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ReferenceValue,
    SweepValue,
)
from zcu_tools.meta_tool import MetaDict

from ._helpers import (
    analyze_req as _analyze_req,
)
from ._helpers import (
    make_ctx as _make_ctx,
)
from ._helpers import (
    make_ml as _make_ml,
)
from ._helpers import (
    md_with_centers as _md_with_centers,
)
from ._helpers import (
    run_req as _run_req,
)

# ---------------------------------------------------------------------------
# cfg_spec structure + validate(ml) + filename stem
# ---------------------------------------------------------------------------


def test_ss_len_rabi_cfg_validates() -> None:
    ml = _make_ml()
    adapter = SsLenRabiAdapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    validate_schema(schema, ml)
    raw = schema_to_raw_dict(schema, None, ml)
    modules = cast(dict[str, Any], raw["modules"])
    assert "qub_pulse" in modules
    assert "readout" in modules
    sweep = cast(dict[str, Any], raw["sweep"])
    assert "length" in sweep


def test_ss_len_rabi_default_seed_matches_notebook_values() -> None:
    ml = _make_ml()
    schema = SsLenRabiAdapter().make_default_cfg(_make_ctx(ml))

    assert schema.value.fields["reps"] == DirectValue(1000)
    assert schema.value.fields["rounds"] == DirectValue(100)
    assert schema.value.fields["relax_delay"] == DirectValue(50.5)

    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    qub_pulse = modules.fields["qub_pulse"]
    assert isinstance(qub_pulse, ReferenceValue)
    assert qub_pulse.value.fields["gain"] == DirectValue(1.0)

    sweep = schema.value.fields["sweep"]
    assert isinstance(sweep, CfgSectionValue)
    length = sweep.fields["length"]
    assert isinstance(length, SweepValue)
    assert length.start == 0.03
    assert length.stop == 0.2
    assert length.expts == 51


def test_ss_len_rabi_default_stop_keeps_pi_len_md_link() -> None:
    ctx = _make_ctx(_make_ml())
    ctx.md.pi_len = 0.06

    schema = SsLenRabiAdapter().make_default_cfg(ctx)
    sweep = schema.value.fields["sweep"]
    assert isinstance(sweep, CfgSectionValue)
    length = sweep.fields["length"]
    assert isinstance(length, SweepValue)
    assert isinstance(length.stop, EvalValue)
    assert length.stop.expr == "4.0 * pi_len"


def test_ss_len_rabi_build_exp_cfg_passes_correct_model() -> None:
    ml = _make_ml()
    adapter = SsLenRabiAdapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    raw = schema_to_raw_dict(schema, None, ml)
    cfg = adapter.build_exp_cfg(raw, _run_req(_md_with_centers(), ml))
    assert isinstance(cfg, LenRabiCfg)


def test_ss_t1_cfg_validates() -> None:
    ml = _make_ml()
    adapter = SsT1Adapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    validate_schema(schema, ml)
    raw = schema_to_raw_dict(schema, None, ml)
    modules = cast(dict[str, Any], raw["modules"])
    assert "pi_pulse" in modules
    assert "readout" in modules
    sweep = cast(dict[str, Any], raw["sweep"])
    assert "length" in sweep
    # uniform extra field present with False default
    assert "uniform" in raw
    assert raw["uniform"] is False


def test_ss_t1_build_exp_cfg_pops_uniform() -> None:
    # ``uniform`` must not reach the cfg model (not a T1Cfg field).
    ml = _make_ml()
    adapter = SsT1Adapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    raw = schema_to_raw_dict(schema, None, ml)
    assert "uniform" in raw  # present before build
    adapter.build_exp_cfg(raw, _run_req(_md_with_centers(), ml))
    # make_cfg must not have received ``uniform`` in its first positional arg (dict).
    call_args = ml.make_cfg.call_args
    passed_dict = call_args[0][0]
    assert "uniform" not in passed_dict
    ml.make_cfg.assert_called_once_with(passed_dict, T1Cfg)


def test_ss_t1_tone_cfg_validates() -> None:
    ml = _make_ml()
    adapter = SsT1ToneAdapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    validate_schema(schema, ml)
    raw = schema_to_raw_dict(schema, None, ml)
    modules = cast(dict[str, Any], raw["modules"])
    assert "pi_pulse" in modules
    assert "probe_pulse" in modules
    assert "readout" in modules
    sweep = cast(dict[str, Any], raw["sweep"])
    assert "length" in sweep
    assert "uniform" in raw
    assert raw["uniform"] is False


def test_ss_t1_tone_probe_defaults_to_readout_frequency_and_channel() -> None:
    ctx = _make_ctx(_make_ml())
    ctx.md.readout_f = 6123.0
    ctx.md.r_f = 6000.0
    ctx.md.res_ch = 2

    schema = SsT1ToneAdapter().make_default_cfg(ctx)
    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    probe = modules.fields["probe_pulse"]
    assert isinstance(probe, ReferenceValue)
    assert probe.chosen_key == "<Custom:Pulse>"
    freq = probe.value.fields["freq"]
    ch = probe.value.fields["ch"]
    assert isinstance(freq, EvalValue)
    assert freq.expr == "readout_f"
    assert isinstance(ch, EvalValue)
    assert ch.expr == "res_ch"


def test_ss_t1_tone_build_exp_cfg_pops_uniform() -> None:
    ml = _make_ml()
    adapter = SsT1ToneAdapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    raw = schema_to_raw_dict(schema, None, ml)
    adapter.build_exp_cfg(raw, _run_req(_md_with_centers(), ml))
    call_args = ml.make_cfg.call_args
    passed_dict = call_args[0][0]
    assert "uniform" not in passed_dict
    ml.make_cfg.assert_called_once_with(passed_dict, T1WithToneCfg)


@pytest.mark.parametrize(
    ("adapter", "prefix"),
    [
        (SsLenRabiAdapter(), "Q1_ss_len_rabi_"),
        (SsT1Adapter(), "Q1_ss_t1_"),
        (SsT1ToneAdapter(), "Q1_ss_t1_tone_"),
    ],
)
def test_filename_stem(adapter: Any, prefix: str) -> None:
    assert adapter.make_filename_stem(_make_ctx()).startswith(prefix)


# ---------------------------------------------------------------------------
# run override — forwards GE centres to domain run (md present)
# ---------------------------------------------------------------------------


def test_ss_len_rabi_run_forwards_centers(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(self: Any, soc: Any, soccfg: Any, cfg: Any, g, e, radius) -> Any:
        del self, cfg
        captured.update(soc=soc, soccfg=soccfg, g=g, e=e, radius=radius)
        return MagicMock()

    monkeypatch.setattr(LenRabiExp, "run", fake_run, raising=True)
    ml = _make_ml()
    md = _md_with_centers()
    req = _run_req(md, ml)
    adapter = SsLenRabiAdapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    adapter.run(req, schema)

    assert captured["g"] == -1.5 + 2.0j
    assert captured["e"] == 1.2 - 0.7j
    assert captured["radius"] == pytest.approx(0.42)
    assert captured["soc"] is req.soc
    assert captured["soccfg"] is req.soccfg


def test_ss_len_rabi_run_fast_fails_without_centers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        LenRabiExp, "run", lambda *a, **k: pytest.fail("domain run should not run")
    )
    ml = _make_ml()
    req = _run_req(MetaDict(), ml)
    schema = SsLenRabiAdapter().make_default_cfg(_make_ctx(ml))
    with pytest.raises(RuntimeError, match="missing 'g_center'.*singleshot/ge"):
        SsLenRabiAdapter().run(req, schema)


@pytest.mark.parametrize("uniform", [False, True])
def test_ss_t1_run_forwards_centers_and_uniform(
    uniform: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(
        self: Any,
        soc: Any,
        soccfg: Any,
        cfg: Any,
        g,
        e,
        radius,
        *,
        uniform: bool = False,
    ) -> Any:
        del self, cfg
        captured.update(g=g, e=e, radius=radius, uniform=uniform)
        return MagicMock()

    monkeypatch.setattr(T1Exp, "run", fake_run, raising=True)
    ml = _make_ml()
    md = _md_with_centers()
    req = _run_req(md, ml)
    adapter = SsT1Adapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    # Override the uniform value in the schema's raw dict directly via a helper:
    # rebuild a patched schema by setting the scalar.
    schema.value.with_field("uniform", uniform)
    adapter.run(req, schema)

    assert captured["g"] == -1.5 + 2.0j
    assert captured["e"] == 1.2 - 0.7j
    assert captured["radius"] == pytest.approx(0.42)
    assert captured["uniform"] is uniform


def test_ss_t1_run_fast_fails_without_centers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        T1Exp, "run", lambda *a, **k: pytest.fail("domain run should not run")
    )
    ml = _make_ml()
    req = _run_req(MetaDict(), ml)
    schema = SsT1Adapter().make_default_cfg(_make_ctx(ml))
    with pytest.raises(RuntimeError, match="missing 'g_center'.*singleshot/ge"):
        SsT1Adapter().run(req, schema)


@pytest.mark.parametrize("uniform", [False, True])
def test_ss_t1_tone_run_forwards_centers_and_uniform(
    uniform: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(
        self: Any,
        soc: Any,
        soccfg: Any,
        cfg: Any,
        g,
        e,
        radius,
        *,
        uniform: bool = False,
    ) -> Any:
        del self, cfg
        captured.update(g=g, e=e, radius=radius, uniform=uniform)
        return MagicMock()

    monkeypatch.setattr(T1WithToneExp, "run", fake_run, raising=True)
    ml = _make_ml()
    md = _md_with_centers()
    req = _run_req(md, ml)
    adapter = SsT1ToneAdapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    schema.value.with_field("uniform", uniform)
    adapter.run(req, schema)

    assert captured["g"] == -1.5 + 2.0j
    assert captured["e"] == 1.2 - 0.7j
    assert captured["radius"] == pytest.approx(0.42)
    assert captured["uniform"] is uniform


def test_ss_t1_tone_run_fast_fails_without_centers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        T1WithToneExp, "run", lambda *a, **k: pytest.fail("domain run should not run")
    )
    ml = _make_ml()
    req = _run_req(MetaDict(), ml)
    schema = SsT1ToneAdapter().make_default_cfg(_make_ctx(ml))
    with pytest.raises(RuntimeError, match="missing 'g_center'.*singleshot/ge"):
        SsT1ToneAdapter().run(req, schema)


# ---------------------------------------------------------------------------
# analyze — figure-only (len_rabi, t1) + numeric writeback (t1_tone)
# ---------------------------------------------------------------------------


def test_ss_len_rabi_analyze_figure_only(monkeypatch: pytest.MonkeyPatch) -> None:
    fig = Figure()

    def fake_analyze(self: Any, result: Any, *, confusion_matrix=None) -> Figure:
        del self, result
        return fig

    monkeypatch.setattr(LenRabiExp, "analyze", fake_analyze, raising=True)
    out = SsLenRabiAdapter().analyze(_analyze_req(MagicMock(), MetaDict()))
    assert isinstance(out, SsLenRabiAnalyzeResult)
    assert out.figure is fig


def test_ss_len_rabi_analyze_forwards_confusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    fig = Figure()

    def fake_analyze(self: Any, result: Any, *, confusion_matrix=None) -> Figure:
        del self, result
        captured["confusion"] = confusion_matrix
        return fig

    monkeypatch.setattr(LenRabiExp, "analyze", fake_analyze, raising=True)
    md = MetaDict()
    md.confusion_matrix = [[0.9, 0.1], [0.05, 0.95]]
    SsLenRabiAdapter().analyze(_analyze_req(MagicMock(), md))
    assert captured["confusion"] == [[0.9, 0.1], [0.05, 0.95]]


def test_ss_t1_analyze_figure_only(monkeypatch: pytest.MonkeyPatch) -> None:
    fig = Figure()

    def fake_analyze(
        self: Any, result: Any, *, confusion_matrix=None, skip=0
    ) -> Figure:
        del self, result, skip
        return fig

    monkeypatch.setattr(T1Exp, "analyze", fake_analyze, raising=True)
    out = SsT1Adapter().analyze(_analyze_req(MagicMock(), MetaDict()))
    assert isinstance(out, SsT1AnalyzeResult)
    assert out.figure is fig


def test_ss_t1_analyze_forwards_confusion(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_analyze(
        self: Any, result: Any, *, confusion_matrix=None, skip=0
    ) -> Figure:
        del self, result, skip
        captured["confusion"] = confusion_matrix
        return Figure()

    monkeypatch.setattr(T1Exp, "analyze", fake_analyze, raising=True)
    md = MetaDict()
    md.confusion_matrix = [[1.0, 0.0], [0.0, 1.0]]
    SsT1Adapter().analyze(_analyze_req(MagicMock(), md))
    assert captured["confusion"] == [[1.0, 0.0], [0.0, 1.0]]


def test_ss_t1_tone_analyze_returns_t1_values(monkeypatch: pytest.MonkeyPatch) -> None:
    fig = Figure()

    def fake_analyze(
        self: Any, result: Any, *, confusion_matrix=None, skip=0
    ) -> tuple[float, float, Figure]:
        del self, result, skip
        return 42.5, 10.3, fig

    monkeypatch.setattr(T1WithToneExp, "analyze", fake_analyze, raising=True)
    out = SsT1ToneAdapter().analyze(_analyze_req(MagicMock(), MetaDict()))
    assert isinstance(out, SsT1ToneAnalyzeResult)
    assert out.figure is fig
    assert out.t1 == pytest.approx(42.5)
    assert out.t1_b == pytest.approx(10.3)


def test_ss_t1_tone_analyze_forwards_confusion(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_analyze(
        self: Any, result: Any, *, confusion_matrix=None, skip=0
    ) -> tuple[float, float, Figure]:
        del self, result, skip
        captured["confusion"] = confusion_matrix
        return 1.0, 1.0, Figure()

    monkeypatch.setattr(T1WithToneExp, "analyze", fake_analyze, raising=True)
    md = MetaDict()
    md.confusion_matrix = [[0.95, 0.05], [0.03, 0.97]]
    SsT1ToneAdapter().analyze(_analyze_req(MagicMock(), md))
    assert captured["confusion"] == [[0.95, 0.05], [0.03, 0.97]]


# ---------------------------------------------------------------------------
# writeback — t1_tone: key is ``t1_with_tone``, value is t1 float
# ---------------------------------------------------------------------------


def test_ss_t1_tone_writeback_key_and_value() -> None:
    # Use a typed stub for run_result so WritebackRequest[T1WithToneResult, ...] is
    # satisfied without importing the real dataclass.
    run_result = cast(
        Any,
        MagicMock(spec=SsT1ToneRunResult),  # type: ignore[misc]
    )
    analyze_result = SsT1ToneAnalyzeResult(t1=37.8, t1_b=9.1, figure=Figure())
    req: WritebackRequest[SsT1ToneRunResult, SsT1ToneAnalyzeResult] = WritebackRequest(
        run_result=run_result,
        analyze_result=analyze_result,
        ctx=_make_ctx(),
    )
    items = SsT1ToneAdapter().get_writeback_items(req)
    assert len(items) == 1
    item = cast(MetaDictWriteback, items[0])
    # MetaDictWriteback has target_name + proposed_value.
    assert item.target_name == "t1_with_tone"
    assert item.proposed_value == pytest.approx(37.8)
