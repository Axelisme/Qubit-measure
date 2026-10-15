"""Unit tests for zcu_tools.gui.app.measure.adapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.cfg_model import PulseSpec
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    ModuleWriteback,
    NoAnalysisResult,
    ParameterWriteback,
    require_soc_handles,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.adapter.protocol import NoAnalyzeParams
from zcu_tools.plotting.plots import NonPresentingHost, Plots


def test_require_soc_handles_is_framework_request_validation() -> None:
    from zcu_tools.gui.app.measure.adapter import RunRequest

    with pytest.raises(RuntimeError, match="soc is required"):
        require_soc_handles(RunRequest(soc=None, soccfg=MagicMock(), cfg_context=None))


def test_parameter_writeback_valid():
    item = ParameterWriteback(target_name="freq", description="d", proposed_value=1)
    assert item.target_name == "freq"
    assert item.proposed_value == 1
    # session_id is stamped by the service, not the adapter; editor identity is
    # private to the service-owned draft.
    assert item.session_id == ""


def test_module_writeback_valid():
    # target_name is the apply destination; the proposal is a persistent Spec.
    proposal = PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.1, waveform="Q1.waveforms.a")
    item = ModuleWriteback(target_name="pulse_a", description="d", proposal=proposal)
    assert item.target_name == "pulse_a"
    assert item.proposal is proposal
    assert item.role_id is None
    assert item.session_id == ""


def _make_concrete_adapter() -> BaseAdapter:
    """Create a minimal concrete no-analysis adapter for shared-method tests."""

    class _FakeAdapter(BaseAdapter):
        capabilities = AdapterCapabilities(analysis=AnalysisMode.NONE)
        exp_cls = MagicMock()

        def make_filename_stem(self, ctx):
            return "fake_stem"

    return _FakeAdapter()


def _make_ctx(**kwargs):
    from zcu_tools.gui.app.measure.adapter import ContextReadiness, SessionEnv

    from tests.gui.app.measure.remote._helpers import make_ctx

    defaults = dict(
        parameters=make_ctx().parameters,
        soc=None,
        soccfg=None,
        result_dir="/res",
        database_path="/db",
        active_label="label",
        readiness=ContextReadiness.ACTIVE,
    )
    defaults.update(kwargs)
    return SessionEnv(**defaults)  # type: ignore[arg-type]


def test_make_default_save_paths_raises_without_database_path():
    adapter = _make_concrete_adapter()
    ctx = _make_ctx(database_path="")
    with pytest.raises(RuntimeError, match="database_path is required"):
        adapter.make_default_save_paths(ctx)


def test_make_default_save_paths_raises_without_result_dir():
    adapter = _make_concrete_adapter()
    ctx = _make_ctx(result_dir="")
    with pytest.raises(RuntimeError, match="result_dir is required"):
        adapter.make_default_save_paths(ctx)


def test_make_default_save_paths_raises_without_active_label():
    adapter = _make_concrete_adapter()
    ctx = _make_ctx(active_label="")
    with pytest.raises(RuntimeError, match="active_label is required"):
        adapter.make_default_save_paths(ctx)


def test_base_adapter_get_analyze_params_returns_noparams_when_no_params():
    # A NoAnalyzeParams adapter (the 4th generic arg defaults to it) inherits the
    # base get_analyze_params, which returns the empty params instance — no
    # per-adapter boilerplate override needed.
    adapter = _make_concrete_adapter()
    from zcu_tools.gui.app.measure.adapter import SessionEnv

    assert isinstance(
        adapter.get_analyze_params(
            MagicMock(), SessionEnv(parameters=None, soc=None, soccfg=None)
        ),
        NoAnalyzeParams,
    )


def test_base_adapter_validates_forgotten_real_analyze_params_override():
    # An adapter whose analyze-params REQUIRE values (a field without a default)
    # but forgetting the override still Fast-Fails (the guard). Params whose every
    # field has a default instead inherit the base get_analyze_params (covered by
    # test_base_validation).
    from dataclasses import dataclass

    @dataclass
    class _RequiredParams:
        flag: bool  # no default → base get_analyze_params cannot construct it

    with pytest.raises(TypeError, match="get_analyze_params"):

        class _Adapter(
            BaseAdapter[MagicMock, MagicMock, NoAnalysisResult, _RequiredParams]
        ):
            capabilities = AdapterCapabilities(analysis=AnalysisMode.FIT)
            exp_cls = MagicMock()

            def make_filename_stem(self, ctx):
                return "stem"

            def analyze(self, req, *, plots: Plots):
                return NoAnalysisResult()


def test_base_adapter_analyze_raises_by_default():
    from zcu_tools.gui.app.measure.adapter import AnalyzeRequest

    adapter = _make_concrete_adapter()
    req = AnalyzeRequest(
        run_result=MagicMock(),
        analyze_params=NoAnalyzeParams(),
        parameters=None,
    )
    with pytest.raises(NotImplementedError, match="analyze"):
        adapter.analyze(req, plots=Plots(NonPresentingHost()))


def test_analyze_params_cls_fallback_when_no_annotation():

    class _UnannotatedAdapter(BaseAdapter):
        capabilities = AdapterCapabilities(analysis=AnalysisMode.FIT)
        exp_cls = MagicMock()

        def make_filename_stem(self, ctx):
            return "stem"

        def get_analyze_params(self, result, ctx):
            return NoAnalyzeParams()

        def analyze(self, req, *, plots: Plots):
            return NoAnalysisResult()

    # The overridden get_analyze_params has no type annotation → fallback
    cls = _UnannotatedAdapter.analyze_params_cls()
    assert cls is NoAnalyzeParams
