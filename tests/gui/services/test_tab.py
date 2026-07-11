from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.gui.app.main.adapter import ContextReadiness, ExpContext, SavePaths
from zcu_tools.gui.app.main.services.tab import TabService
from zcu_tools.gui.app.main.state import Session, State


def test_tab_snapshot_is_single_pure_render_model() -> None:
    state = State(
        ExpContext(
            md=MagicMock(),
            ml=MagicMock(),
            soc=MagicMock(),
            soccfg=MagicMock(),
            readiness=ContextReadiness.ACTIVE,
        )
    )
    analyze_params = object()
    state.add_tab(
        "tab",
        Session(
            adapter_name="fake",
            adapter=MagicMock(),
            cfg_schema=MagicMock(),
            run_result=object(),
            analyze_result=object(),
            analyze_param_instance=analyze_params,
            # M4: save paths now come from the tab aggregate's
            # effective_save_paths; an override exercises it without the adapter.
            save_path_overrides=SavePaths("data.h5", "image.png"),
        ),
    )
    writeback = MagicMock()
    writeback.get_tab_writeback_items.return_value = []
    # TabService's render model depends only on State + a writeback query port;
    # readiness / save paths come off State's aggregates, not sibling
    # app-services. The registry is unused by get_snapshot.
    service = TabService(state, MagicMock(), writeback)

    snapshot = service.get_snapshot("tab")

    assert snapshot.tab_id == "tab"
    assert snapshot.interaction is not None  # render path fills every live field
    assert snapshot.interaction.has_run_result is True
    assert snapshot.interaction.has_active_context is True  # ctx.readiness=ACTIVE
    assert snapshot.analyze_params is analyze_params
    assert snapshot.save_paths == SavePaths("data.h5", "image.png")
    assert state.get_tab("tab").analyze_param_instance is analyze_params


def _active_state() -> State:
    return State(
        ExpContext(
            md=MagicMock(),
            ml=MagicMock(),
            soc=MagicMock(),
            soccfg=MagicMock(),
            readiness=ContextReadiness.ACTIVE,
        )
    )


def test_snapshot_carries_post_analyze_fields() -> None:
    state = _active_state()
    post_params = object()
    post_fig = object()
    state.add_tab(
        "tab",
        Session(
            adapter_name="ge",
            adapter=MagicMock(),
            cfg_schema=MagicMock(),
            run_result=object(),
            analyze_result=MagicMock(),
            post_analyze_result=object(),
            post_figure=post_fig,  # type: ignore[arg-type]
            post_analyze_param_instance=post_params,
        ),
    )
    writeback = MagicMock()
    writeback.get_tab_writeback_items.return_value = []
    service = TabService(state, MagicMock(), writeback)

    snapshot = service.get_snapshot("tab")

    assert snapshot.post_analyze_params is post_params
    assert snapshot.post_figure is post_fig
    assert snapshot.interaction is not None
    assert snapshot.interaction.has_post_analyze_result is True


def test_initialize_post_analyze_params_seeds_from_primary_result() -> None:
    state = _active_state()
    adapter = MagicMock()
    built = object()
    adapter.get_post_analyze_params.return_value = built
    state.add_tab(
        "tab",
        Session(
            adapter_name="ge",
            adapter=adapter,
            cfg_schema=MagicMock(),
            run_result=object(),
            analyze_result=MagicMock(),  # primary result present
        ),
    )
    service = TabService(state, MagicMock(), MagicMock())

    out = service.initialize_tab_post_analyze_params("tab")

    assert out is built
    assert state.get_tab("tab").post_analyze_param_instance is built


def test_initialize_post_analyze_params_fast_fails_without_primary_result() -> None:
    import pytest

    state = _active_state()
    state.add_tab(
        "tab",
        Session(
            adapter_name="ge",
            adapter=MagicMock(),
            cfg_schema=MagicMock(),
            run_result=object(),
            analyze_result=None,  # no primary analyze result
        ),
    )
    service = TabService(state, MagicMock(), MagicMock())

    with pytest.raises(RuntimeError, match="primary analyze result"):
        service.initialize_tab_post_analyze_params("tab")
