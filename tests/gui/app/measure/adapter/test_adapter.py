"""Unit tests for zcu_tools.gui.app.measure.adapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.cfg_model import ConfigBase, PulseSpec
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    ModuleWriteback,
    ParameterWriteback,
    require_soc_handles,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.gui._adapter_fakes import DummyAdapter, DummyResult


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


def _make_concrete_adapter():
    """Use the shared synthetic definition without primary analysis."""
    return Adapter(
        spec=DummyAdapter.spec,
        make_form=DummyAdapter.make_form,
        run=DummyAdapter.run,
        guide=None,
    )


def test_base_adapter_get_analyze_params_returns_noparams_when_no_params():
    # Omitted params use None without a per-adapter override.
    adapter = _make_concrete_adapter()
    from zcu_tools.gui.app.measure.adapter import SessionEnv

    assert (
        adapter.get_analyze_params(
            MagicMock(), SessionEnv(parameters=None, soc=None, soccfg=None)
        )
        is None
    )


def test_base_adapter_analyze_raises_by_default():
    from zcu_tools.gui.app.measure.adapter import AnalyzeRequest

    adapter = _make_concrete_adapter()
    req = AnalyzeRequest(
        run_result=RunRecord(ConfigBase(), DummyResult()),
        analyze_params=None,
        parameters=None,
    )
    with pytest.raises(NotImplementedError, match="analysis"):
        adapter.analyze(req, plots=Plots(NonPresentingHost()))
