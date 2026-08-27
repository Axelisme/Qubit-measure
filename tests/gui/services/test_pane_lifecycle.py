from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    ContextReadiness,
    ExpContext,
    MetaDictWriteback,
    SavePaths,
)
from zcu_tools.gui.app.main.services.analyze import AnalyzeService
from zcu_tools.gui.app.main.services.guard import (
    AnalyzePermit,
    GuardService,
    LoadPermit,
)
from zcu_tools.gui.app.main.services.load import LoadDataError, LoadService
from zcu_tools.gui.app.main.services.tab import TabService
from zcu_tools.gui.app.main.state import (
    Session,
    State,
)
from zcu_tools.gui.cfg import CfgSchema, CfgSectionSpec, CfgSectionValue
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import ExpectedErrorCategory
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

from tests.gui.services._progress_fakes import DirectProgressTransport


@dataclass
class _Draft:
    items: tuple[Any, ...]


class _Writeback:
    def __init__(self) -> None:
        self.created: list[_Draft] = []
        self.torn_down: list[_Draft] = []
        self.fail_teardown = False
        self.fail_create = False

    def create_draft(self, items: list[Any]) -> _Draft:
        if self.fail_create:
            raise RuntimeError("draft creation failed")
        draft = _Draft(tuple(items))
        self.created.append(draft)
        return draft

    def preview_draft(self, draft: _Draft) -> list[Any]:
        return list(draft.items)

    def teardown_draft(self, draft: _Draft) -> None:
        self.torn_down.append(draft)
        if self.fail_teardown:
            raise RuntimeError("teardown failed")


class _Bg:
    def submit(
        self, work: Any, *, run_in_pool: bool, on_done: Any, on_error: Any
    ) -> None:
        self.work = work
        self.on_done = on_done
        self.on_error = on_error


def _state() -> tuple[State, str, MagicMock, ExpContext]:
    ctx = ExpContext(
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=MagicMock(),
        soccfg=MagicMock(),
        database_path="/db",
        result_dir="/result",
        active_label="ctx",
        readiness=ContextReadiness.ACTIVE,
    )
    state = State(ctx)
    adapter = MagicMock()
    adapter.make_save_paths.return_value = SavePaths("/db/data.h5", "/result/base.png")
    state.add_tab(
        "tab",
        Session(
            adapter_name="fake",
            adapter=adapter,
            cfg_schema=CfgSchema(spec=CfgSectionSpec(), value=CfgSectionValue()),
        ),
    )
    state.update_tab_result("tab", "run")
    return state, "tab", adapter, ctx


def _analyze_service(
    state: State, writeback: _Writeback, bus: EventBus | None = None
) -> tuple[AnalyzeService, _Bg]:
    bus = bus or EventBus()
    handles = OperationHandles()
    bg = _Bg()
    runner = OperationRunner(
        MagicMock(), handles, ProgressService(DirectProgressTransport()), bg, bus
    )
    return AnalyzeService(state, runner, bus, cast(Any, writeback), handles), bg


def test_snapshot_exposes_independent_panes_and_paths() -> None:
    state, tab_id, adapter, _ctx = _state()
    primary = object()
    post = object()
    state.replace_analysis_pane(
        tab_id,
        result=primary,
        figure="primary-figure",  # type: ignore[arg-type]
        params="primary-params",
        writeback_draft="primary-draft",
    )
    state.replace_post_analysis_pane(
        tab_id,
        result=post,
        figure="post-figure",  # type: ignore[arg-type]
        params="post-params",
        writeback_draft="post-draft",
    )
    state.update_tab_data_path_override(tab_id, "/custom/data.h5")
    state.update_tab_analysis_image_path_override(tab_id, "/custom/a.png")
    state.update_tab_post_analysis_image_path_override(tab_id, "/custom/p.png")

    # Final contract: TabService composes pane-owned drafts via preview_draft.
    primary_draft = MagicMock()
    post_draft = MagicMock()
    writeback = MagicMock()
    writeback.preview_draft.side_effect = lambda d: (
        ["primary-item"]
        if d is primary_draft
        else ["post-item"]
        if d is post_draft
        else []
    )
    # Attach drafts to panes so snapshot can preview them.
    state.get_tab(tab_id).analysis.writeback_draft = primary_draft  # type: ignore[assignment]
    state.get_tab(tab_id).post_analysis.writeback_draft = post_draft  # type: ignore[assignment]
    snapshot = TabService(state, MagicMock(), writeback).get_snapshot(tab_id)

    assert snapshot.run is not None and snapshot.run.result == "run"
    assert snapshot.analysis is not None and snapshot.analysis.result is primary
    assert snapshot.post_analysis is not None and snapshot.post_analysis.result is post
    assert snapshot.analysis.writeback_items == ("primary-item",)
    assert snapshot.post_analysis.writeback_items == ("post-item",)
    assert snapshot.paths is not None
    assert snapshot.paths.data.path == "/custom/data.h5"
    assert snapshot.paths.analysis_image.path == "/custom/a.png"
    assert snapshot.paths.post_analysis_image.path == "/custom/p.png"
    assert adapter.make_save_paths.call_count == 0


def test_primary_swap_returns_all_retired_dependents_and_invalidates_post() -> None:
    state, tab_id, _adapter, _ctx = _state()
    old_primary = object()
    old_post = object()
    old_primary_draft = object()
    old_post_draft = object()
    state.replace_analysis_pane(
        tab_id,
        result=old_primary,
        figure="old-primary",  # type: ignore[arg-type]
        writeback_draft=old_primary_draft,
    )
    state.replace_post_analysis_pane(
        tab_id,
        result=old_post,
        figure="old-post",  # type: ignore[arg-type]
        writeback_draft=old_post_draft,
    )

    retired = state.replace_analysis_pane(
        tab_id,
        result="new-primary",
        figure="new-primary-figure",  # type: ignore[arg-type]
        writeback_draft="new-primary-draft",
    )

    tab = state.get_tab(tab_id)
    assert retired.analysis.result is old_primary
    assert retired.analysis.writeback_draft is old_primary_draft
    assert retired.post_analysis.result is old_post
    assert retired.post_analysis.writeback_draft is old_post_draft
    assert retired.writeback_drafts == (old_primary_draft, old_post_draft)
    assert tab.analysis.result == "new-primary"
    assert tab.analysis.writeback_draft == "new-primary-draft"
    assert tab.post_analysis.result is None
    assert tab.post_analysis.writeback_draft is None


def test_post_swap_does_not_replace_primary() -> None:
    state, tab_id, _adapter, _ctx = _state()
    state.replace_analysis_pane(tab_id, result="primary", figure=None)
    primary_pane = state.get_tab(tab_id).analysis

    state.replace_post_analysis_pane(tab_id, result="post", figure=None)

    tab = state.get_tab(tab_id)
    assert tab.analysis is primary_pane
    assert tab.analysis.result == "primary"
    assert tab.post_analysis.result == "post"


def test_analyze_uses_captured_inputs_and_cleans_retired_after_commit() -> None:
    state, tab_id, adapter, ctx = _state()
    old_primary_draft = _Draft(())
    old_post_draft = _Draft(())
    state.replace_analysis_pane(
        tab_id, result="old-primary", figure=None, writeback_draft=old_primary_draft
    )
    state.replace_post_analysis_pane(
        tab_id, result="old-post", figure=None, writeback_draft=old_post_draft
    )
    adapter.get_writeback_items.return_value = [MetaDictWriteback("new", "new", 1.0)]
    writeback = _Writeback()
    service, bg = _analyze_service(state, writeback)
    service.start_analyze(AnalyzePermit(tab_id), "params")

    new_ctx = ExpContext(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)
    state.set_context(new_ctx)
    state.get_tab(tab_id).run.result = "changed-after-start"
    result = MagicMock(figure=None)
    bg.on_done(result)

    request = adapter.get_writeback_items.call_args.args[0]
    assert request.run_result == "run"
    assert request.ctx is ctx
    assert state.get_tab(tab_id).analysis.result is result
    assert state.get_tab(tab_id).post_analysis.result is None
    assert old_primary_draft in writeback.torn_down
    assert old_post_draft in writeback.torn_down


def test_failed_draft_build_preserves_previous_primary_and_post() -> None:
    state, tab_id, adapter, _ctx = _state()
    adapter.get_writeback_items.return_value = []
    state.replace_analysis_pane(tab_id, result="old-primary", figure=None)
    state.replace_post_analysis_pane(tab_id, result="old-post", figure=None)
    writeback = _Writeback()
    writeback.fail_create = True
    service, bg = _analyze_service(state, writeback)
    service.start_analyze(AnalyzePermit(tab_id), "params")

    bg.on_done(MagicMock(figure=None))

    tab = state.get_tab(tab_id)
    assert tab.analysis.result == "old-primary"
    assert tab.post_analysis.result == "old-post"
    assert writeback.created == []


def test_retired_teardown_failure_does_not_roll_back_committed_pane() -> None:
    state, tab_id, adapter, _ctx = _state()
    old_primary_draft = _Draft(())
    old_post_draft = _Draft(())
    state.replace_analysis_pane(
        tab_id, result="old-primary", figure=None, writeback_draft=old_primary_draft
    )
    state.replace_post_analysis_pane(
        tab_id, result="old-post", figure=None, writeback_draft=old_post_draft
    )
    adapter.get_writeback_items.return_value = []
    writeback = _Writeback()
    writeback.fail_teardown = True
    service, bg = _analyze_service(state, writeback)
    service.start_analyze(AnalyzePermit(tab_id), "params")
    new_result = MagicMock(figure=None)

    bg.on_done(new_result)

    tab = state.get_tab(tab_id)
    assert tab.analysis.result is new_result
    assert tab.analysis.writeback_draft is writeback.created[-1]
    assert tab.post_analysis.result is None
    assert tab.post_analysis.writeback_draft is None
    assert writeback.torn_down == [old_primary_draft, old_post_draft]


def test_load_capability_gate_rejects_concrete_disabled_adapter() -> None:
    state, tab_id, _adapter, _ctx = _state()

    class DisabledAdapter:
        capabilities = AdapterCapabilities(load_data=False)

        def load(self, _request: object) -> object:
            return object()

    adapter = DisabledAdapter()
    state.get_tab(tab_id).adapter = adapter  # type: ignore[assignment]
    load = LoadService(state, MagicMock())

    with pytest.raises(LoadDataError) as exc_info:
        load.load_result(LoadPermit(tab_id), "/tmp/x")

    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert exc_info.value.reason_code == "unsupported_load"


def test_guard_and_load_accept_enabled_concrete_adapter() -> None:
    state, tab_id, _adapter, _ctx = _state()

    class EnabledAdapter:
        capabilities = AdapterCapabilities(load_data=True)

        def load(self, _request: object) -> object:
            return object()

    state.get_tab(tab_id).adapter = EnabledAdapter()  # type: ignore[assignment]
    permit = GuardService(state).acquire_load_permit(tab_id)
    assert permit.tab_id == tab_id
