"""Tests for FakeFreqAdapter — build_exp_cfg and spec structure."""

from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.experiment.v2_gui.adapters.onetone.fakefreq import FakeFreqAdapter
from zcu_tools.gui.adapter import CfgSchema, RunRequest


def _make_ml():
    ml = MagicMock()
    ml.modules = {}
    ml.waveforms = {}

    def make_cfg(raw, cls, **kwargs):
        return cls(**{**raw, **kwargs})

    ml.make_cfg.side_effect = make_cfg
    return ml


def _make_ctx(ml=None):
    if ml is None:
        ml = _make_ml()
    ctx = MagicMock()
    ctx.ml = ml
    ctx.md = MagicMock(spec=[])
    return ctx


def _make_req(ml=None):
    if ml is None:
        ml = _make_ml()
    return RunRequest(md=MagicMock(), ml=ml, soc=None, soccfg=None)


def _lower(schema: CfgSchema, req: RunRequest) -> dict:
    from zcu_tools.gui.adapter.lowering import schema_to_dict

    return schema_to_dict(schema, req.ml)


def test_fakefreq_build_exp_cfg_basic():
    """build_exp_cfg delegates entirely to ml.make_cfg with the lowered dict."""
    ml = _make_ml()
    ctx = _make_ctx(ml)
    req = _make_req(ml)

    adapter = FakeFreqAdapter(fast_mode=True)
    schema = adapter.make_default_cfg(ctx)
    raw = _lower(schema, req)

    # Verify top-level keys match FakeFreqCfg structure
    assert "reps" in raw
    assert "rounds" in raw
    assert "sweep" in raw
    assert "model" in raw
    assert "modules" in raw

    # sweep has freq SweepCfg
    from zcu_tools.program.v2 import SweepCfg

    assert isinstance(raw["sweep"]["freq"], SweepCfg)

    # model has all expected keys
    for key in ("freq", "Ql", "Qc_abs", "phi", "a0_abs", "edelay", "noise_scale"):
        assert key in raw["model"], f"missing model key: {key}"

    # optional modules absent (disabled)
    assert "init_pulse" not in raw["modules"]
    assert "reset" not in raw["modules"]

    # ml.make_cfg was called
    adapter.build_exp_cfg(raw, req)
    ml.make_cfg.assert_called_once()
    call_kwargs = ml.make_cfg.call_args
    assert call_kwargs.kwargs.get("fast_mode") is True


def test_fakefreq_build_exp_cfg_without_optional_modules():
    """When optional modules are disabled, lowering omits their keys."""
    ml = _make_ml()
    ctx = _make_ctx(ml)
    req = _make_req(ml)

    schema = FakeFreqAdapter(fast_mode=True).make_default_cfg(ctx)
    raw = _lower(schema, req)

    assert "init_pulse" not in raw.get("modules", {})
    assert "reset" not in raw.get("modules", {})


def test_fakefreq_make_default_cfg_spec_structure():
    """Spec structure matches FakeFreqCfg nesting."""
    from zcu_tools.gui.adapter import CfgSectionSpec, ModuleRefSpec, SweepSpec

    ctx = _make_ctx()
    schema = FakeFreqAdapter().make_default_cfg(ctx)
    spec = schema.spec

    assert "sweep" in spec.fields
    sweep_spec = spec.fields["sweep"]
    assert isinstance(sweep_spec, CfgSectionSpec)
    assert "freq" in sweep_spec.fields
    assert isinstance(sweep_spec.fields["freq"], SweepSpec)

    assert "model" in spec.fields
    model_spec = spec.fields["model"]
    assert isinstance(model_spec, CfgSectionSpec)
    for key in ("freq", "Ql", "Qc_abs"):
        assert key in model_spec.fields

    modules_spec = spec.fields["modules"]
    assert isinstance(modules_spec, CfgSectionSpec)
    assert isinstance(modules_spec.fields["init_pulse"], ModuleRefSpec)
    assert modules_spec.fields["init_pulse"].optional is True
    assert isinstance(modules_spec.fields["reset"], ModuleRefSpec)
    assert modules_spec.fields["reset"].optional is True
