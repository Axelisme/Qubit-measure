from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.gui.adapter import ContextReadiness, ExpContext, SavePaths
from zcu_tools.gui.services.tab import TabService
from zcu_tools.gui.state import Session, State


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
