"""Result-bound Primary/Post parameter preparation through TabService and State."""

from collections.abc import Callable
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    ContextReadiness,
    PostAnalyzeRequest,
)
from zcu_tools.gui.app.measure.services.analyze import AnalyzeService
from zcu_tools.gui.app.measure.services.plot_lifecycle import release_retired_plots
from zcu_tools.gui.app.measure.services.post_analyze import PostAnalyzeService
from zcu_tools.gui.app.measure.services.tab import TabService
from zcu_tools.gui.app.measure.state import Session, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.interactive import PluginDefinition
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.types import ParameterContext, SessionEnv
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.gui.app.measure._form_fakes import form_resources, form_sources
from tests.gui.app.measure._load_support import historical_metadata
from tests.gui.app.measure._record_support import require_record
from tests.gui.session.ui._parameter_fakes import make_parameters


@pytest.mark.parametrize("loaded", [False, True], ids=["run", "load"])
@pytest.mark.parametrize("context_free", [False, True], ids=["captured", "none"])
def test_default_analysis_params_retain_result_source(
    result_sources: tuple[ParameterContext, ParameterContext],
    loaded: bool,
    context_free: bool,
) -> None:
    first, current = result_sources
    saved = None if context_free else first
    state = State(SessionEnv(parameters=current, soc=None, soccfg=None))
    adapter = MagicMock()
    adapter.capabilities = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=True
    )
    primary_params, post_params = object(), object()
    primary_contexts: list[ParameterContext | None] = []
    post_contexts: list[ParameterContext | None] = []

    def primary(_result: object, ctx: SessionEnv) -> object:
        primary_contexts.append(ctx.parameters)
        return primary_params

    def post(_source: object, _result: object, ctx: SessionEnv) -> object:
        post_contexts.append(ctx.parameters)
        return post_params

    adapter.get_analyze_params.side_effect = primary
    adapter.get_post_analyze_params.side_effect = post
    state.add_tab(
        "tab",
        Session(
            adapter_name="source",
            adapter=adapter,
            cfg=MagicMock(),
        ),
    )
    result = RunRecord(None, object())
    metadata = historical_metadata()
    retired = (
        state.update_tab_loaded_result(
            "tab", replace(result, metadata=metadata), "native.h5", parameters=saved
        )
        if loaded
        else state.update_tab_result("tab", result, parameters=saved)
    )
    release_retired_plots(retired)
    service = TabService(
        state,
        MagicMock(),
        MagicMock(),
        form_resources(state),
        form_sources=lambda aliases: form_sources(state, aliases),
    )
    preparation = service.prepare_result_analysis("tab")
    assert preparation.has_params and preparation.error is None
    assert state.get_tab("tab").analysis.params is primary_params
    assert primary_contexts == [saved]
    release_retired_plots(state.update_tab_analyze("tab", object(), None))
    assert service.initialize_tab_post_analyze_params("tab") is post_params
    assert state.get_tab("tab").post_analysis.params is post_params
    assert post_contexts == [saved]
    assert state.get_tab("tab").run.parameters is saved
    assert state.session_env.parameters is current
    if saved is not None:
        assert saved.value("$source_qubit.rate") == 1.0
    if loaded:
        assert state.get_tab("tab").run.source_operation_id is None
        assert require_record(state.get_tab("tab").run).metadata == metadata
        assert state.get_tab("tab").run.source_path == "native.h5"


class _CapturedBackground:
    """Record work and owner callbacks for explicit in-process delivery."""

    def __init__(self) -> None:
        self.last_work: Callable[[], object] | None = None
        self.last_on_done: Callable[[object], None] | None = None

    def submit(
        self,
        work: Callable[[], object],
        *,
        run_in_pool: bool,
        on_done: Callable[[object], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        """Capture the submitted work without starting an independent worker."""
        self.last_work = work
        self.last_on_done = on_done


def _make_source_state() -> State:
    state = State(
        SessionEnv(
            parameters=None, soc=None, soccfg=None, readiness=ContextReadiness.ACTIVE
        )
    )
    state.add_tab(
        "tab1", Session(adapter_name="fake", adapter=MagicMock(), cfg=MagicMock())
    )
    state.update_tab_result("tab1", RunRecord(None, object()))
    return state


def _make_primary_service(
    state: State, bus: EventBus, *, writeback: MagicMock
) -> tuple[AnalyzeService, _CapturedBackground]:
    background, handles = _CapturedBackground(), OperationHandles()
    runner = OperationRunner(None, handles, None, background, bus)
    writeback.create_draft.return_value = None
    return AnalyzeService(state, runner, bus, writeback, handles), background


def _make_post_service(
    state: State, bus: EventBus, *, writeback: MagicMock
) -> tuple[PostAnalyzeService, _CapturedBackground]:
    background, handles = _CapturedBackground(), OperationHandles()
    runner = OperationRunner(None, handles, None, background, bus)
    writeback.create_draft.return_value = None
    return PostAnalyzeService(state, runner, bus, handles, writeback), background


def _plots() -> Plots:
    return Plots(NonPresentingHost())


@pytest.mark.parametrize("context_free", [False, True])
def test_fit_uses_run_source_for_request_and_writeback_after_rebinding(
    qapp,
    result_sources: tuple[ParameterContext, ParameterContext],
    context_free: bool,
):
    state = _make_source_state()
    q1, q2 = result_sources
    captured = None if context_free else q1
    tab = state.get_tab("tab1")
    state.update_tab_result("tab1", RunRecord(None, object()), parameters=captured)
    state.set_context(replace(state.session_env, parameters=q2))
    writeback = MagicMock()
    service, background = _make_primary_service(state, EventBus(), writeback=writeback)
    service.start_analyze("tab1", object(), plots=_plots())
    state.set_context(
        replace(state.session_env, parameters=make_parameters({"rate": 3.0}))
    )

    assert background.last_work is not None
    result = background.last_work()
    assert isinstance(tab.adapter, MagicMock)
    request = tab.adapter.analyze.call_args.args[0]
    assert isinstance(request, AnalyzeRequest)
    assert request.parameters is captured
    if request.parameters is not None:
        assert request.parameters.resolve_path("$source_qubit.rate") == "Q1.rate"
        assert request.parameters.value("$source_qubit.rate") == 1.0
    assert background.last_on_done is not None
    background.last_on_done(result)
    proposal = tab.adapter.get_writeback_items.call_args.args[0]
    assert proposal.ctx.parameters is captured
    assert writeback.create_draft.call_args.kwargs["parameters"] is captured
    if proposal.ctx.parameters is not None:
        assert proposal.ctx.parameters.resolve_path("$source_qubit.rate") == "Q1.rate"
    assert tab.analysis.result is result


@pytest.mark.parametrize("context_free", [False, True])
def test_interactive_writeback_retains_run_source_after_rebinding(
    qapp,
    result_sources: tuple[ParameterContext, ParameterContext],
    context_free: bool,
) -> None:
    state = _make_source_state()
    q1, q2 = result_sources
    captured = None if context_free else q1
    state.update_tab_result("tab1", RunRecord(None, object()), parameters=captured)
    tab = state.get_tab("tab1")
    state.set_context(replace(state.session_env, parameters=q2))
    writeback = MagicMock()
    service, _background = _make_primary_service(state, EventBus(), writeback=writeback)
    plugin = PluginDefinition(
        "pick", 0, (), lambda _state: None, lambda _state: object()
    )
    service.start_plugin(
        "tab1",
        plugin,
        ManualOwnerScheduler(),
        analyze_params_instance="submitted",
        plots=_plots(),
    )
    state.set_context(replace(state.session_env, parameters=None))
    assert service.finish_plugin("tab1") is True

    assert isinstance(tab.adapter, MagicMock)
    proposal = tab.adapter.get_writeback_items.call_args.args[0]
    assert proposal.ctx.parameters is captured
    assert writeback.create_draft.call_args.kwargs["parameters"] is captured
    if proposal.ctx.parameters is not None:
        assert proposal.ctx.parameters.resolve_path("$source_qubit.rate") == "Q1.rate"
    assert tab.analysis.result is not None
    assert service.is_interactive_active("tab1") is False


@pytest.mark.parametrize("context_free", [False, True])
def test_post_uses_run_source_for_request_and_writeback_after_rebinding(
    qapp,
    result_sources: tuple[ParameterContext, ParameterContext],
    context_free: bool,
) -> None:
    state = _make_source_state()
    q1, q2 = result_sources
    captured = None if context_free else q1
    tab = state.get_tab("tab1")
    run_result, primary_result = RunRecord(None, object()), object()
    state.update_tab_result("tab1", run_result, parameters=captured)
    state.update_tab_analyze("tab1", primary_result, None)
    state.set_context(replace(state.session_env, parameters=q2))
    writeback = MagicMock()
    service, background = _make_post_service(state, EventBus(), writeback=writeback)
    service.start_post_analyze("tab1", object(), plots=_plots())
    state.set_context(replace(state.session_env, parameters=None))

    assert background.last_work is not None
    result = background.last_work()
    assert isinstance(tab.adapter, MagicMock)
    request = tab.adapter.post_analyze.call_args.args[0]
    assert isinstance(request, PostAnalyzeRequest)
    assert request.run_result is run_result
    assert request.analyze_result is primary_result
    assert request.parameters is captured
    if request.parameters is not None:
        assert request.parameters.resolve_path("$source_qubit.rate") == "Q1.rate"
        assert request.parameters.value("$source_qubit.rate") == 1.0
    assert background.last_on_done is not None
    background.last_on_done(result)
    proposal = tab.adapter.get_post_writeback_items.call_args.args[0]
    assert proposal.ctx.parameters is captured
    assert writeback.create_draft.call_args.kwargs["parameters"] is captured
    if proposal.ctx.parameters is not None:
        assert proposal.ctx.parameters.resolve_path("$source_qubit.rate") == "Q1.rate"
    assert tab.post_analysis.result is result
