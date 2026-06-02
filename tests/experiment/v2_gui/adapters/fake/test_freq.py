"""Tests for FakeFreqAdapter — build_exp_cfg and spec structure."""

from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.experiment.v2_gui.adapters.fake.freq import FakeFreqAdapter
from zcu_tools.gui.adapter import CfgSchema, RunRequest
from zcu_tools.meta_tool import MetaDict


def _make_ml():
    ml = MagicMock()
    ml.modules = {}
    ml.waveforms = {}
    ml.make_cfg.return_value = MagicMock()
    return ml


def _make_ctx(ml=None):
    if ml is None:
        ml = _make_ml()
    ctx = MagicMock()
    ctx.ml = ml
    ctx.md = MetaDict()
    return ctx


def _make_req(ml=None):
    if ml is None:
        ml = _make_ml()
    return RunRequest(md=MagicMock(), ml=ml, soc=None, soccfg=None)


def _lower(schema: CfgSchema, req: RunRequest) -> dict:

    return schema.to_raw_dict(None, req.ml)


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
    assert "modules" in raw
    # The simulated resonance is held in the adapter (__init__), not the cfg.
    assert "model" not in raw

    # sweep has freq SweepCfg
    from zcu_tools.program.v2 import SweepCfg

    assert isinstance(raw["sweep"]["freq"], SweepCfg)

    # one-tone modules: readout only. No init_pulse (no qubit-drive pulse) and
    # no reset (one-tone runs without a qubit reset).
    assert "init_pulse" not in raw["modules"]
    assert "reset" not in raw["modules"]

    # ml.make_cfg was called
    adapter.build_exp_cfg(raw, req)
    ml.make_cfg.assert_called_once()
    call_kwargs = ml.make_cfg.call_args
    assert call_kwargs.kwargs.get("fast_mode") is True


def test_fakefreq_modules_are_readout_only():
    """The one-tone modules expose readout only — no init_pulse, no reset."""
    ml = _make_ml()
    ctx = _make_ctx(ml)
    req = _make_req(ml)

    schema = FakeFreqAdapter(fast_mode=True).make_default_cfg(ctx)
    raw = _lower(schema, req)

    assert set(raw["modules"]) == {"readout"}


def test_fakefreq_make_default_cfg_spec_structure():
    """Spec structure matches FakeFreqCfg nesting."""
    from zcu_tools.gui.adapter import CfgSectionSpec, SweepSpec

    ctx = _make_ctx()
    schema = FakeFreqAdapter().make_default_cfg(ctx)
    spec = schema.spec

    assert "sweep" in spec.fields
    sweep_spec = spec.fields["sweep"]
    assert isinstance(sweep_spec, CfgSectionSpec)
    assert "freq" in sweep_spec.fields
    assert isinstance(sweep_spec.fields["freq"], SweepSpec)

    # No 'model' block: simulated resonance is an __init__ arg, not a cfg field.
    assert "model" not in spec.fields

    # Modules mirror the real one-tone ExpCfg: readout only (no init_pulse, no
    # reset).
    modules_spec = spec.fields["modules"]
    assert isinstance(modules_spec, CfgSectionSpec)
    assert set(modules_spec.fields) == {"readout"}


# ---------------------------------------------------------------------------
# Blind-sweep tests — the resonance lives in __init__, the sweep is set
# independently (centred on r_f), so the analysis must genuinely *find* the dip
# rather than read an aligned cfg field.
# ---------------------------------------------------------------------------


def _real_ctx():
    """A real ExpContext so run/analyze actually compute (not MagicMock)."""
    from zcu_tools.gui.adapter import ExpContext
    from zcu_tools.meta_tool import ModuleLibrary

    return ExpContext(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)


def _run_and_fit(adapter: FakeFreqAdapter):
    from zcu_tools.gui.adapter import AnalyzeRequest
    from zcu_tools.gui.adapter import RunRequest as RR

    ctx = _real_ctx()
    schema = adapter.make_default_cfg(ctx)
    result = adapter.run(RR(md=ctx.md, ml=ctx.ml, soc=None, soccfg=None), schema)
    return adapter.analyze(
        AnalyzeRequest(
            run_result=result,
            analyze_params=adapter.get_analyze_params(result, ctx),
            md=ctx.md,
            ml=ctx.ml,
            predictor=ctx.predictor,
        )
    )


def test_hanger_blind_sweep_finds_hidden_resonance():
    """sweep centres on r_f=6000 (cfg default); the true dip is 6080, supplied
    only via __init__ — the fit must locate it without the cfg telling it. Ql is
    lowered so the dip is several sweep-steps wide (the default sweep is ~2 MHz/
    point over [5800, 6200]); a sub-resolution dip is a separate concern."""
    from zcu_tools.experiment.v2_gui.adapters.fake.freq import HangerSimParams

    adapter = FakeFreqAdapter(
        params=HangerSimParams(freq=6080.0, Ql=500.0, Qc_abs=600.0),
        fast_mode=True,
    )
    analyze_result = _run_and_fit(adapter)

    assert abs(analyze_result.freq - 6080.0) < 5.0


def test_transmission_blind_sweep_finds_hidden_resonance():
    from zcu_tools.experiment.v2_gui.adapters.fake.freq import TransmissionSimParams

    adapter = FakeFreqAdapter(
        model_type="t",
        params=TransmissionSimParams(freq=6088.0, Ql=500.0),
        fast_mode=True,
    )
    analyze_result = _run_and_fit(adapter)

    assert abs(analyze_result.freq - 6088.0) < 5.0


def _run_for_save(adapter: FakeFreqAdapter):
    from zcu_tools.gui.adapter import RunRequest as RR

    ctx = _real_ctx()
    schema = adapter.make_default_cfg(ctx)
    return adapter.run(RR(md=ctx.md, ml=ctx.ml, soc=None, soccfg=None), schema)


def _save_request(result, data_path: str):
    from zcu_tools.gui.adapter import SaveDataRequest

    return SaveDataRequest(
        run_result=result,
        data_path=data_path,
        md=MetaDict(readonly=True),
        ml=MagicMock(),  # fake save() does not touch ml
        chip_name="chip",
        qub_name="q",
        res_name="r",
        active_label="0.0",
        comment="",
    )


def test_save_persist_true_writes_hdf5(tmp_path):
    # persist_data=True (the registry default) → save writes a real HDF5 so
    # "data saved to <path>" is truthful and the file exists (Phase 130 #1).
    import glob

    adapter = FakeFreqAdapter(fast_mode=True, persist_data=True)
    result = _run_for_save(adapter)
    data_path = str(tmp_path / "fake_save")
    adapter.save(_save_request(result, data_path))
    # save_data appends an extension / safe-name; assert a file actually landed.
    assert glob.glob(str(tmp_path / "fake_save*")), "no data file written"


def test_save_persist_false_is_noop(tmp_path):
    adapter = FakeFreqAdapter(fast_mode=True, persist_data=False)
    result = _run_for_save(adapter)
    data_path = str(tmp_path / "fake_save")
    adapter.save(_save_request(result, data_path))
    assert not list(tmp_path.iterdir()), "no-op save must not write anything"


def test_model_type_params_mismatch_is_fast_fail():
    """A hanger run with transmission params is a bug — reject at construction."""
    import pytest
    from zcu_tools.experiment.v2_gui.adapters.fake.freq import TransmissionSimParams

    with pytest.raises(TypeError, match="HangerSimParams"):
        FakeFreqAdapter(model_type="hm", params=TransmissionSimParams())
