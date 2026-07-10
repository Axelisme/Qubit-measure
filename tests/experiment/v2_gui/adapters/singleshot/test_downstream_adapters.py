"""Tests for the single-shot downstream adapters.

Covers the shared ``read_ge_centers`` helper (incl. the complex round-trip — the
high-risk silent-break point), the mist run overrides that forward the GE
classification trio to the domain ``run``, the check adapter that reads the trio
at analyze time, and each adapter's cfg/analyze surface. The domain experiment
classes' ``run`` / ``analyze`` are monkeypatched so only the adapter's
md→domain wiring is under test (the domain numerics are covered elsewhere)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from zcu_tools.experiment.v2.singleshot import CheckCfg, CheckExp
from zcu_tools.experiment.v2.singleshot.mist import (
    FreqCfg,
    FreqDepExp,
    PowerCfg,
    PowerExp,
)
from zcu_tools.experiment.v2_gui.adapters.singleshot import (
    CheckAdapter,
    MistFreqAdapter,
    MistPowerAdapter,
)
from zcu_tools.experiment.v2_gui.adapters.singleshot._shared import read_ge_centers
from zcu_tools.experiment.v2_gui.adapters.singleshot.check import CheckAnalyzeResult
from zcu_tools.experiment.v2_gui.adapters.singleshot.mist.freq import (
    MistFreqAnalyzeResult,
)
from zcu_tools.experiment.v2_gui.adapters.singleshot.mist.power import (
    MistPowerAnalyzeResult,
)
from zcu_tools.gui.app.main.adapter import (
    CfgSectionValue,
    EvalValue,
    ReferenceValue,
    SweepValue,
)
from zcu_tools.gui.app.main.adapter.lowering import (
    schema_to_raw_dict,
    validate_schema,
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
# read_ge_centers — the shared helper (complex round-trip is the key risk).
# ---------------------------------------------------------------------------


def test_read_ge_centers_in_process_complex() -> None:
    md = _md_with_centers()
    g, e, r = read_ge_centers(md)
    assert g == -1.5 + 2.0j
    assert e == 1.2 - 0.7j
    assert isinstance(g, complex) and isinstance(e, complex)
    assert r == pytest.approx(0.42)
    assert isinstance(r, float)


def test_read_ge_centers_disk_round_trip() -> None:
    # The high-risk silent-break point: complex centres written to md, persisted
    # (str dump) and reloaded (_restore_complex) must come back as complex with
    # identical values — exactly the GE-writeback → md → downstream-read path.
    md = _md_with_centers()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "md.json"
        persisted = md.clone(path)
        persisted.dump()
        reloaded = MetaDict(path)
        g, e, r = read_ge_centers(reloaded)
    assert g == -1.5 + 2.0j
    assert e == 1.2 - 0.7j
    assert isinstance(g, complex) and isinstance(e, complex)
    assert r == pytest.approx(0.42)


@pytest.mark.parametrize("missing", ["g_center", "e_center", "ge_radius"])
def test_read_ge_centers_fast_fails_when_missing(missing: str) -> None:
    md = _md_with_centers()
    delattr(md, missing)
    with pytest.raises(RuntimeError, match=f"missing '{missing}'.*singleshot/ge"):
        read_ge_centers(md)


# ---------------------------------------------------------------------------
# cfg surface — structure complete + validate(ml) clean + filename stem.
# ---------------------------------------------------------------------------

_MIST_PARAMS = [
    (MistFreqAdapter(), FreqCfg, "freq", "modules"),
    (MistPowerAdapter(), PowerCfg, "gain", "modules"),
]


@pytest.mark.parametrize(("adapter", "cfg_model", "sweep_key", "_m"), _MIST_PARAMS)
def test_mist_cfg_validates_and_delegates(
    adapter: Any, cfg_model: type, sweep_key: str, _m: str
) -> None:
    from zcu_tools.program.v2 import SweepCfg

    ml = _make_ml()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    validate_schema(schema, ml)
    raw = schema_to_raw_dict(schema, None, ml)
    modules = cast(dict[str, Any], raw["modules"])
    assert "probe_pulse" in modules
    assert "readout" in modules
    sweep = cast(dict[str, Any], raw["sweep"])
    assert isinstance(sweep[sweep_key], SweepCfg)

    cfg = adapter.build_exp_cfg(raw, _run_req(MetaDict(), ml))
    assert isinstance(cfg, cfg_model)


def test_mist_freq_probe_defaults_to_readout_frequency_sweep_and_channel() -> None:
    ctx = _make_ctx(_make_ml())
    ctx.md.readout_f = 6123.0
    ctx.md.r_f = 6000.0
    ctx.md.rf_w = 20.0
    ctx.md.res_ch = 2

    schema = MistFreqAdapter().make_default_cfg(ctx)
    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    probe = modules.fields["probe_pulse"]
    assert isinstance(probe, ReferenceValue)
    freq = probe.value.fields["freq"]
    ch = probe.value.fields["ch"]
    assert isinstance(freq, EvalValue)
    assert freq.expr == "readout_f"
    assert isinstance(ch, EvalValue)
    assert ch.expr == "res_ch"
    sweep = schema.value.fields["sweep"]
    assert isinstance(sweep, CfgSectionValue)
    freq_sweep = sweep.fields["freq"]
    assert isinstance(freq_sweep, SweepValue)
    assert isinstance(freq_sweep.start, EvalValue)
    assert isinstance(freq_sweep.stop, EvalValue)
    assert freq_sweep.start.expr == "readout_f - 1.5 * rf_w"
    assert freq_sweep.stop.expr == "readout_f + 1.5 * rf_w"


def test_mist_power_probe_defaults_to_readout_frequency_and_channel() -> None:
    ctx = _make_ctx(_make_ml())
    ctx.md.readout_f = 6123.0
    ctx.md.r_f = 6000.0
    ctx.md.res_ch = 2

    schema = MistPowerAdapter().make_default_cfg(ctx)
    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    probe = modules.fields["probe_pulse"]
    assert isinstance(probe, ReferenceValue)
    freq = probe.value.fields["freq"]
    ch = probe.value.fields["ch"]
    assert isinstance(freq, EvalValue)
    assert freq.expr == "readout_f"
    assert isinstance(ch, EvalValue)
    assert ch.expr == "res_ch"


def test_check_cfg_validates_and_has_shots() -> None:
    ml = _make_ml()
    adapter = CheckAdapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))
    validate_schema(schema, ml)
    raw = schema_to_raw_dict(schema, None, ml)
    assert raw["shots"] == 5000
    modules = cast(dict[str, Any], raw["modules"])
    assert "probe_pulse" in modules and "readout" in modules
    # check has no swept axis.
    assert "sweep" not in raw
    cfg = adapter.build_exp_cfg(raw, _run_req(MetaDict(), ml))
    assert isinstance(cfg, CheckCfg)


@pytest.mark.parametrize(
    ("adapter", "prefix"),
    [
        (MistFreqAdapter(), "Q1_mist_freq_"),
        (MistPowerAdapter(), "Q1_mist_power_"),
        (CheckAdapter(), "Q1_sh_check_"),
    ],
)
def test_filename_stem(adapter: Any, prefix: str) -> None:
    assert adapter.make_filename_stem(_make_ctx()).startswith(prefix)


# ---------------------------------------------------------------------------
# mist run override — forwards the GE trio to the domain run (md present);
# fast-fails when the trio is absent.
# ---------------------------------------------------------------------------

_MIST_RUN_PARAMS = [
    (MistFreqAdapter(), FreqDepExp),
    (MistPowerAdapter(), PowerExp),
]


@pytest.mark.parametrize(("adapter", "exp_cls"), _MIST_RUN_PARAMS)
def test_mist_run_forwards_centers(
    adapter: Any, exp_cls: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(
        self: Any, soc: Any, soccfg: Any, cfg: Any, g_center, e_center, radius
    ) -> Any:
        del self, cfg
        captured.update(soc=soc, soccfg=soccfg, g=g_center, e=e_center, radius=radius)
        return MagicMock()

    monkeypatch.setattr(exp_cls, "run", fake_run, raising=True)
    ml = _make_ml()
    md = _md_with_centers()
    req = _run_req(md, ml)
    schema = adapter.make_default_cfg(_make_ctx(ml))

    adapter.run(req, schema)

    assert captured["soc"] is req.soc
    assert captured["soccfg"] is req.soccfg
    assert captured["g"] == -1.5 + 2.0j
    assert captured["e"] == 1.2 - 0.7j
    assert captured["radius"] == pytest.approx(0.42)


@pytest.mark.parametrize(("adapter", "exp_cls"), _MIST_RUN_PARAMS)
def test_mist_run_fast_fails_without_centers(
    adapter: Any, exp_cls: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        exp_cls, "run", lambda *a, **k: pytest.fail("domain run should not run")
    )
    ml = _make_ml()
    req = _run_req(MetaDict(), ml)  # md has no centres
    schema = adapter.make_default_cfg(_make_ctx(ml))
    with pytest.raises(RuntimeError, match="missing 'g_center'.*singleshot/ge"):
        adapter.run(req, schema)


def test_check_run_uses_standard_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # check's run needs NO centres — the standard BaseAdapter.run path calls
    # CheckExp.run(soc, soccfg, cfg) with the three-arg signature.
    captured: dict[str, Any] = {}

    def fake_run(self: Any, soc: Any, soccfg: Any, cfg: Any) -> Any:
        del self, cfg
        captured.update(soc=soc, soccfg=soccfg)
        return MagicMock()

    monkeypatch.setattr(CheckExp, "run", fake_run, raising=True)
    ml = _make_ml()
    req = _run_req(MetaDict(), ml)  # no centres needed for the run
    schema = CheckAdapter().make_default_cfg(_make_ctx(ml))
    CheckAdapter().run(req, schema)
    assert captured["soc"] is req.soc


def test_mist_freq_analyze_reads_confusion(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    fig = Figure()

    def fake_analyze(self: Any, result: Any, *, confusion_matrix=None) -> Figure:
        del self, result
        captured["confusion"] = confusion_matrix
        return fig

    monkeypatch.setattr(FreqDepExp, "analyze", fake_analyze, raising=True)
    md = MetaDict()
    md.confusion_matrix = [[1.0, 0.0], [0.0, 1.0]]
    out = MistFreqAdapter().analyze(_analyze_req(MagicMock(), md))
    assert isinstance(out, MistFreqAnalyzeResult)
    assert out.figure is fig
    assert captured["confusion"] == [[1.0, 0.0], [0.0, 1.0]]


def test_mist_freq_analyze_no_confusion(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_analyze(self: Any, result: Any, *, confusion_matrix=None) -> Figure:
        del self, result
        captured["confusion"] = confusion_matrix
        return Figure()

    monkeypatch.setattr(FreqDepExp, "analyze", fake_analyze, raising=True)
    MistFreqAdapter().analyze(_analyze_req(MagicMock(), MetaDict()))
    assert captured["confusion"] is None


def test_mist_power_analyze_reads_ac_coeff(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    fig = Figure()

    def fake_analyze(
        self: Any, result: Any, *, ac_coeff=None, log_scale=False, confusion_matrix=None
    ) -> Figure:
        del self, result
        captured.update(
            ac_coeff=ac_coeff, log_scale=log_scale, confusion_matrix=confusion_matrix
        )
        return fig

    monkeypatch.setattr(PowerExp, "analyze", fake_analyze, raising=True)
    md = MetaDict()
    md.ac_stark_coeff = 3.5
    md.log_scale = True
    out = MistPowerAdapter().analyze(_analyze_req(MagicMock(), md))
    assert isinstance(out, MistPowerAnalyzeResult)
    assert out.figure is fig
    assert captured["ac_coeff"] == pytest.approx(3.5)
    assert captured["log_scale"] is True
    assert captured["confusion_matrix"] is None


def test_check_analyze_forwards_centers(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    fig = Figure()

    def fake_analyze(
        self: Any, g_center, e_center, radius, result=None, max_point=5000
    ) -> Figure:
        del self
        captured.update(g=g_center, e=e_center, radius=radius, result=result)
        return fig

    monkeypatch.setattr(CheckExp, "analyze", fake_analyze, raising=True)
    run_result = MagicMock()
    out = CheckAdapter().analyze(_analyze_req(run_result, _md_with_centers()))
    assert isinstance(out, CheckAnalyzeResult)
    assert out.figure is fig
    assert captured["g"] == -1.5 + 2.0j
    assert captured["e"] == 1.2 - 0.7j
    assert captured["radius"] == pytest.approx(0.42)
    assert captured["result"] is run_result


def test_check_analyze_fast_fails_without_centers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        CheckExp, "analyze", lambda *a, **k: pytest.fail("should not analyze")
    )
    with pytest.raises(RuntimeError, match="missing 'g_center'.*singleshot/ge"):
        CheckAdapter().analyze(_analyze_req(MagicMock(), MetaDict()))
