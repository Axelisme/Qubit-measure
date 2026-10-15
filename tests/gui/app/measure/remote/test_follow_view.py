"""Explicit write commands select the view before starting their mutation."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.run_saving import SaveReport
from zcu_tools.gui.app.measure.adapter import AnalysisMode
from zcu_tools.gui.app.measure.remote.handlers.analysis import (
    h_tab_analyze,
    h_tab_post_analyze,
)
from zcu_tools.gui.app.measure.remote.handlers.run_save import (
    h_tab_run_start,
    h_tab_save_artifacts,
)
from zcu_tools.gui.app.measure.remote.handlers.writeback import (
    h_tab_writeback_write,
)
from zcu_tools.gui.remote.errors import RemoteError
from zcu_tools.gui.session.context_types import CommandOrigin


@dataclass
class Params:
    threshold: float = 0.5


@pytest.mark.parametrize("headless", [False, True])
@pytest.mark.parametrize(
    "handler,pane,method",
    [
        (h_tab_run_start, "run", "start_run"),
        (h_tab_analyze, "analysis", "analyze"),
        (h_tab_post_analyze, "post_analysis", "start_post_analyze"),
    ],
)
def test_write_follow_precedes_mutation_and_headless_still_works(
    handler, pane, method, headless
):
    adapter = MagicMock()
    control = adapter.run_analyze_control
    control.get_tab_snapshot.return_value = SimpleNamespace(
        interaction=None,
        capabilities=SimpleNamespace(analysis=AnalysisMode.FIT),
        analysis=SimpleNamespace(params=Params(), has_writeback_draft=False),
        post_analysis=SimpleNamespace(
            params=Params(), has_writeback_draft=False, result=None
        ),
    )
    adapter.tab_control.get_running_tab_id.return_value = None
    owner = control
    order = []
    if headless:
        adapter.render_view = None
    else:
        adapter.render_view.select_tab_pane.side_effect = lambda tab, selected: (
            order.append((tab, selected))
        )
    getattr(owner, method).side_effect = lambda *args, **kwargs: (
        order.append("mutation") or MagicMock()
    )
    handler(
        adapter,
        {
            "tab_id": "t",
            "updates": {},
            "edits": [],
            "agent_edit": True,
        },
    )
    assert order == (["mutation"] if headless else [("t", pane), "mutation"])


@pytest.mark.parametrize("headless", [False, True])
def test_save_selects_data_before_start_and_supports_headless(headless):
    adapter = MagicMock()
    order = []
    if headless:
        adapter.render_view = None
    else:
        adapter.render_view.select_tab_pane.side_effect = lambda tab, pane: (
            order.append((tab, pane))
        )

    def save(*args, **kwargs):
        order.append("save")
        return SaveReport(run_id="run-t", outputs=())

    adapter.save_control.save_artifacts.side_effect = save
    adapter.tab_control.get_tab_snapshot.return_value = SimpleNamespace(artifacts=())
    result = h_tab_save_artifacts(
        adapter,
        {
            "tab_id": "t",
            "artifacts": "all",
            "run_operation_id": None,
            "operation_id": None,
            "comment": None,
        },
    )
    assert result == {"run_id": "run-t", "artifacts": {}}
    assert order == (["save"] if headless else [("t", "data"), "save"])


def test_failed_save_view_selection_does_not_start_save():
    adapter = MagicMock()
    adapter.render_view.select_tab_pane.side_effect = ValueError("unavailable pane")
    with pytest.raises(ValueError, match="unavailable pane"):
        h_tab_save_artifacts(
            adapter,
            {
                "tab_id": "t",
                "artifacts": "all",
                "run_operation_id": None,
                "operation_id": None,
                "comment": None,
            },
        )
    adapter.save_control.save_artifacts.assert_not_called()


@pytest.mark.parametrize("pane", ["analysis", "post_analysis"])
@pytest.mark.parametrize("headless", [False, True])
def test_writeback_selects_its_pane_before_shared_draft_changes(pane, headless):
    adapter = MagicMock()
    order = []
    if headless:
        adapter.render_view = None
    else:
        adapter.render_view.select_tab_pane.side_effect = lambda tab, selected: (
            order.append((tab, selected))
        )

    def write(*args):
        order.append("write")
        return []

    adapter.writeback_control.write_writeback_for_pane.side_effect = write
    assert h_tab_writeback_write(
        adapter,
        {
            "tab_id": "t",
            "subtab_id": pane,
            "write": [],
        },
    ) == {"written": []}
    assert order == (["write"] if headless else [("t", pane), "write"])


def test_failed_writeback_view_selection_leaves_draft_untouched():
    adapter = MagicMock()
    adapter.render_view.select_tab_pane.side_effect = ValueError("unavailable pane")
    with pytest.raises(ValueError, match="unavailable pane"):
        h_tab_writeback_write(
            adapter,
            {
                "tab_id": "t",
                "subtab_id": "analysis",
                "write": [{"id": "param-1"}],
            },
        )
    adapter.writeback_control.write_writeback_for_pane.assert_not_called()


def test_run_forwards_without_editing_sources():
    adapter = MagicMock()
    adapter.render_view = None
    adapter.run_analyze_control.start_run.return_value = 19
    assert h_tab_run_start(adapter, {"tab_id": "t"}) == {"operation_id": 19}
    adapter.run_analyze_control.start_run.assert_called_once_with(
        "t",
        aliases=None,
        origin=CommandOrigin("mcp"),
    )


@pytest.mark.parametrize("target", ["", "   ", 1, []])
def test_run_rejects_malformed_target_before_operation_or_view_changes(target):
    adapter = MagicMock()
    with pytest.raises(RemoteError) as caught:
        h_tab_run_start(
            adapter,
            {
                "tab_id": "t",
                "aliases": {"qubit": target},
            },
        )
    assert caught.value.reason == "invalid_aliases"
    adapter.run_analyze_control.start_run.assert_not_called()
    adapter.render_view.select_tab_pane.assert_not_called()


def test_failed_view_selection_does_not_start_an_operation():
    adapter = MagicMock()
    failure = ValueError("unavailable pane")
    adapter.render_view.select_tab_pane.side_effect = failure
    with pytest.raises(ValueError, match="unavailable pane") as caught:
        h_tab_run_start(
            adapter,
            {
                "tab_id": "t",
            },
        )
    assert caught.value is failure
    adapter.run_analyze_control.start_run.assert_not_called()
