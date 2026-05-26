from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.gui.adapter import ContextReadiness, ExpContext, SavePaths
from zcu_tools.gui.services.tab_view import TabViewService
from zcu_tools.gui.state import State, TabState


def test_tab_view_snapshot_is_single_pure_render_model() -> None:
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
        TabState(
            adapter_name="fake",
            adapter=MagicMock(),
            cfg_schema=MagicMock(),
            run_result=object(),
            analyze_result=object(),
            analyze_param_instance=analyze_params,
        ),
    )
    tabs = MagicMock()
    tabs.get_tab_save_paths.return_value = SavePaths("data.h5", "image.png")
    writeback = MagicMock()
    writeback.get_tab_writeback_items.return_value = []
    context = MagicMock()
    context.has_context.return_value = True
    context.is_active_context.return_value = True
    service = TabViewService(state, tabs, writeback, context)

    snapshot = service.get_snapshot("tab")

    assert snapshot.tab_id == "tab"
    assert snapshot.interaction.has_run_result is True
    assert snapshot.interaction.has_active_context is True
    assert snapshot.analyze_params is analyze_params
    assert snapshot.save_paths == SavePaths("data.h5", "image.png")
    assert state.get_tab("tab").analyze_param_instance is analyze_params
