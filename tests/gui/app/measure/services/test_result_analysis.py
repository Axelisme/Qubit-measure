"""Result-bound Primary/Post parameter preparation through TabService and State."""

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.measure.services.plot_lifecycle import release_retired_plots
from zcu_tools.gui.app.measure.services.tab import TabService
from zcu_tools.gui.app.measure.state import Session, State
from zcu_tools.gui.session.types import ParameterContext, SessionEnv

from tests.gui.app.measure._cfg_fakes import cfg_resolution, cfg_resources
from tests.gui.app.measure._load_support import historical_metadata


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

    def post(_result: object, ctx: SessionEnv) -> object:
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
            parameters=current,
        ),
    )
    result = object()
    metadata = historical_metadata()
    retired = (
        state.update_tab_loaded_result(
            "tab", result, "native.h5", metadata=metadata, parameters=saved
        )
        if loaded
        else state.update_tab_result("tab", result, parameters=saved)
    )
    release_retired_plots(retired)
    service = TabService(
        state,
        MagicMock(),
        MagicMock(),
        cfg_resources(state),
        cfg_resolution=lambda parameters: cfg_resolution(state, parameters),
        captured_parameters=lambda _roles, _target: state.session_env.parameters,
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
    assert state.get_tab("tab").parameters is current
    if saved is not None:
        assert saved.value("source_qubit.rate") == 1.0
    if loaded:
        assert state.get_tab("tab").run.source_operation_id is None
        assert state.get_tab("tab").run.metadata is metadata
        assert state.get_tab("tab").run.source_path == "native.h5"
