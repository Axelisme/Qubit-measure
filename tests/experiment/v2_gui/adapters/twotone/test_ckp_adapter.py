"""CKP adapter cfg, run, analysis, persistence, and writeback contracts."""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.figure import Figure
from zcu_tools.experiment.v2.twotone.ckp import CKP_Cfg, CKP_Exp, CKP_Result
from zcu_tools.experiment.v2_gui.adapters.twotone import CKPAdapter
from zcu_tools.experiment.v2_gui.adapters.twotone.ckp import CKPAnalyzeResult
from zcu_tools.experiment.v2_gui.registry import ADAPTERS, register_all
from zcu_tools.gui.app.main.adapter import (
    AnalyzeRequest,
    ExpAdapterProtocol,
    LoadDataRequest,
    MetaDictWriteback,
    NoAnalyzeParams,
    RunRequest,
    SaveDataRequest,
    WritebackRequest,
)
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.app.main.registry import Registry
from zcu_tools.gui.cfg import CfgSectionValue, EvalValue, SweepValue
from zcu_tools.gui.session.value_lookup import EmptyValueLookup
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _make_ctx(ml: ModuleLibrary | None = None, **md_values: float) -> MagicMock:
    ctx = MagicMock()
    ctx.ml = ml or ModuleLibrary()
    md = MetaDict()
    for key, value in md_values.items():
        setattr(md, key, value)
    ctx.md = md
    ctx.qub_name = "Q1"
    ctx.values = EmptyValueLookup()
    return ctx


def _make_req(
    ml: ModuleLibrary | None = None,
    *,
    md: MetaDict | None = None,
    soc: Any = None,
    soccfg: Any = None,
) -> RunRequest:
    return RunRequest(
        md=md or MetaDict(),
        ml=ml or ModuleLibrary(),
        soc=soc,
        soccfg=soccfg,
    )


def _build_cfg(adapter: CKPAdapter, ctx: MagicMock, ml: ModuleLibrary) -> CKP_Cfg:
    schema = adapter.make_default_cfg(ctx)
    raw = schema_to_raw_dict(schema, ctx.md, ml)
    cfg = adapter.build_exp_cfg(raw, _make_req(ml, md=ctx.md))
    assert isinstance(cfg, CKP_Cfg)
    return cfg


def _sample_result(cfg: CKP_Cfg | None = None) -> CKP_Result:
    res_freqs = np.array([5999.0, 6001.0], dtype=np.float64)
    qub_freqs = np.array([3999.0, 4001.0], dtype=np.float64)
    signals = np.arange(8, dtype=np.float64).reshape(2, 2, 2).astype(np.complex128)
    return CKP_Result(
        res_freqs=res_freqs,
        qub_freqs=qub_freqs,
        signals=signals,
        cfg_snapshot=cfg,
    )


def test_ckp_registered_listable_and_creatable() -> None:
    assert ADAPTERS["twotone/ckp"] is CKPAdapter

    registry = Registry()
    register_all(registry)
    assert "twotone/ckp" in registry.list_names()
    adapter = registry.create("twotone/ckp")
    assert isinstance(adapter, CKPAdapter)
    assert isinstance(adapter, ExpAdapterProtocol)


def test_ckp_guide_is_complete() -> None:
    guide = CKPAdapter.guide()
    assert all(
        (
            guide.behavior,
            guide.expects_md,
            guide.expects_ml,
            guide.typical_writeback,
            guide.recommended,
        )
    )
    assert "chi" in guide.typical_writeback
    assert "rf_w" in guide.typical_writeback
    assert "readout_f" in guide.typical_writeback


def test_ckp_cfg_uses_notebook_seeds_and_builds_domain_cfg() -> None:
    ml = ModuleLibrary()
    ctx = _make_ctx(
        ml,
        r_f=6000.0,
        rf_w=2.0,
        q_f=4000.0,
        qub_1_4_ch=1.0,
        res_ch=2.0,
        ro_ch=3.0,
        timeFly=0.5,
    )
    adapter = CKPAdapter()
    schema = adapter.make_default_cfg(ctx)

    sweep_section = schema.value.fields["sweep"]
    assert isinstance(sweep_section, CfgSectionValue)
    res_sweep = sweep_section.fields["res_freq"]
    qub_sweep = sweep_section.fields["qub_freq"]
    assert isinstance(res_sweep, SweepValue)
    assert isinstance(qub_sweep, SweepValue)
    assert isinstance(res_sweep.start, EvalValue)
    assert res_sweep.start.expr == "r_f - 1.5 * rf_w"
    assert isinstance(res_sweep.stop, EvalValue)
    assert res_sweep.stop.expr == "r_f + 1.5 * rf_w"
    assert isinstance(qub_sweep.start, EvalValue)
    assert qub_sweep.start.expr == "q_f - 10"
    assert isinstance(qub_sweep.stop, EvalValue)
    assert qub_sweep.stop.expr == "q_f + 5"

    raw = schema_to_raw_dict(schema, ctx.md, ml)
    assert raw["reps"] == 100
    assert raw["rounds"] == 100
    assert raw["relax_delay"] == pytest.approx(10.1)

    cfg = adapter.build_exp_cfg(raw, _make_req(ml, md=ctx.md))
    assert isinstance(cfg, CKP_Cfg)
    assert cfg.modules.reset is None
    assert cfg.modules.pi_pulse.ch == 1
    assert cfg.modules.res_pulse.ch == 2
    assert cfg.modules.res_pulse.freq == 0.0
    assert cfg.modules.res_pulse.gain == pytest.approx(0.015)
    assert cfg.modules.res_pulse.waveform.length == pytest.approx(
        1.5 + 5.1 / (2 * math.pi * 2.0)
    )
    assert cfg.modules.qub_pulse.ch == 1
    assert cfg.modules.qub_pulse.freq == 0.0
    assert cfg.modules.qub_pulse.gain == pytest.approx(0.01)
    assert cfg.modules.qub_pulse.waveform.length == pytest.approx(1.5)
    assert cfg.modules.qub_pulse.pre_delay == pytest.approx(5.0 / (2 * math.pi * 2.0))
    assert cfg.modules.qub_pulse.post_delay == pytest.approx(3.1 / (2 * math.pi * 2.0))
    assert cfg.sweep.res_freq.start == pytest.approx(5997.0)
    assert cfg.sweep.res_freq.stop == pytest.approx(6003.0)
    assert cfg.sweep.res_freq.expts == 101
    assert cfg.sweep.qub_freq.start == pytest.approx(3990.0)
    assert cfg.sweep.qub_freq.stop == pytest.approx(4005.0)
    assert cfg.sweep.qub_freq.expts == 101


def test_ckp_cfg_without_calibration_uses_literal_fallbacks() -> None:
    ml = ModuleLibrary()
    ctx = _make_ctx(ml)
    adapter = CKPAdapter()
    schema = adapter.make_default_cfg(ctx)
    cfg = _build_cfg(adapter, ctx, ml)

    sweep_section = schema.value.fields["sweep"]
    assert isinstance(sweep_section, CfgSectionValue)
    res_sweep = sweep_section.fields["res_freq"]
    qub_sweep = sweep_section.fields["qub_freq"]
    assert isinstance(res_sweep, SweepValue)
    assert isinstance(qub_sweep, SweepValue)
    assert res_sweep.start == pytest.approx(5750.0)
    assert res_sweep.stop == pytest.approx(7250.0)
    assert qub_sweep.start == pytest.approx(4990.0)
    assert qub_sweep.stop == pytest.approx(5005.0)
    assert cfg.modules.res_pulse.waveform.length == pytest.approx(
        1.5 + 5.1 / (2 * math.pi * 5.0)
    )
    assert cfg.modules.qub_pulse.pre_delay == pytest.approx(5.0 / (2 * math.pi * 5.0))
    assert cfg.modules.qub_pulse.post_delay == pytest.approx(3.1 / (2 * math.pi * 5.0))


def test_ckp_cfg_fast_fails_nonpositive_rf_width() -> None:
    with pytest.raises(ValueError, match="rf_w.*positive finite"):
        CKPAdapter().make_default_cfg(_make_ctx(rf_w=0.0))


def test_ckp_run_uses_standard_domain_path(monkeypatch: pytest.MonkeyPatch) -> None:
    ml = ModuleLibrary()
    ctx = _make_ctx(ml, r_f=6000.0, rf_w=2.0, q_f=4000.0)
    schema = CKPAdapter().make_default_cfg(ctx)
    soc = MagicMock(name="soc")
    soccfg = MagicMock(name="soccfg")
    req = _make_req(ml, md=ctx.md, soc=soc, soccfg=soccfg)
    sentinel = _sample_result()
    captured: dict[str, Any] = {}

    def fake_run(
        self: CKP_Exp, run_soc: object, run_soccfg: object, cfg: CKP_Cfg
    ) -> CKP_Result:
        del self
        captured.update(soc=run_soc, soccfg=run_soccfg, cfg=cfg)
        return sentinel

    monkeypatch.setattr(CKP_Exp, "run", fake_run, raising=True)

    result = CKPAdapter().run(req, schema)

    assert result is sentinel
    assert captured["soc"] is soc
    assert captured["soccfg"] is soccfg
    assert isinstance(captured["cfg"], CKP_Cfg)


def test_ckp_run_without_soc_fast_fails() -> None:
    ml = ModuleLibrary()
    ctx = _make_ctx(ml)
    schema = CKPAdapter().make_default_cfg(ctx)

    with pytest.raises(RuntimeError, match="soc is required"):
        CKPAdapter().run(_make_req(ml, md=ctx.md), schema)


def test_ckp_analyze_projects_typed_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    figure = Figure()

    def fake_analyze(
        self: CKP_Exp, result: CKP_Result | None = None
    ) -> tuple[float, float, float, Figure]:
        del self
        assert result is not None
        return 1.25, 0.75, 6001.5, figure

    monkeypatch.setattr(CKP_Exp, "analyze", fake_analyze, raising=True)
    run_result = _sample_result()

    result = CKPAdapter().analyze(
        AnalyzeRequest(
            run_result=run_result,
            analyze_params=NoAnalyzeParams(),
            md=MetaDict(),
            ml=ModuleLibrary(),
            predictor=None,
        )
    )

    assert result == CKPAnalyzeResult(
        chi=1.25,
        kappa=0.75,
        res_freq=6001.5,
        figure=figure,
    )
    assert result.to_summary_dict() == {
        "chi": 1.25,
        "kappa": 0.75,
        "res_freq": 6001.5,
    }


def test_ckp_writeback_proposes_notebook_targets() -> None:
    result = CKPAnalyzeResult(
        chi=1.25,
        kappa=0.75,
        res_freq=6001.5,
        figure=Figure(),
    )
    items = list(
        CKPAdapter().get_writeback_items(
            WritebackRequest(
                run_result=_sample_result(),
                analyze_result=result,
                ctx=_make_ctx(),
            )
        )
    )

    assert len(items) == 3
    assert all(isinstance(item, MetaDictWriteback) for item in items)
    by_name = {item.target_name: item for item in items}
    assert set(by_name) == {"chi", "rf_w", "readout_f"}
    chi_item = by_name["chi"]
    rf_w_item = by_name["rf_w"]
    readout_f_item = by_name["readout_f"]
    assert isinstance(chi_item, MetaDictWriteback)
    assert isinstance(rf_w_item, MetaDictWriteback)
    assert isinstance(readout_f_item, MetaDictWriteback)
    assert chi_item.proposed_value == pytest.approx(1.25)
    assert rf_w_item.proposed_value == pytest.approx(0.75)
    assert readout_f_item.proposed_value == pytest.approx(6001.5)


def test_ckp_adapter_canonical_save_load_roundtrip(tmp_path) -> None:
    ml = ModuleLibrary()
    ctx = _make_ctx(ml, r_f=6000.0, rf_w=2.0, q_f=4000.0)
    cfg = _build_cfg(CKPAdapter(), ctx, ml)
    result = _sample_result(cfg)
    path = str(tmp_path / "ckp.hdf5")
    adapter = CKPAdapter()

    adapter.save(
        SaveDataRequest(
            data_path=path,
            run_result=result,
            md=ctx.md,
            ml=ml,
            chip_name="chip",
            qub_name="Q1",
            res_name="R1",
            active_label="1",
        )
    )
    loaded = adapter.load(LoadDataRequest(data_path=path, md=ctx.md, ml=ml))

    assert isinstance(loaded, CKP_Result)
    np.testing.assert_allclose(loaded.res_freqs, result.res_freqs)
    np.testing.assert_allclose(loaded.qub_freqs, result.qub_freqs)
    np.testing.assert_allclose(loaded.signals, result.signals)
    assert isinstance(loaded.cfg_snapshot, CKP_Cfg)
