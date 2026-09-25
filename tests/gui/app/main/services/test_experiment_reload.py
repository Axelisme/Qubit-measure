"""RAM recovery through real workspace/tab/state seams, without hardware."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter import ExpContext
from zcu_tools.gui.app.main.catalog import CatalogReloadError, ExperimentAccess
from zcu_tools.gui.app.main.events.tab import TabClosedPayload
from zcu_tools.gui.app.main.registry import Registry
from zcu_tools.gui.app.main.services.experiment_reload import ExperimentReloadService
from zcu_tools.gui.app.main.services.tab import TabService
from zcu_tools.gui.app.main.services.tab_control import TabControlFacet
from zcu_tools.gui.app.main.services.workspace import WorkspaceService
from zcu_tools.gui.app.main.state import State
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionValue,
    DirectValue,
    schema_to_raw,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError

from tests.gui.app.main._reload_fakes import Loader, NewAdapter, OldAdapter


@dataclass
class App:
    reload: ExperimentReloadService
    state: State
    workspace: WorkspaceService
    controls: TabControlFacet
    registry: Registry
    loader: Loader
    access: ExperimentAccess
    writeback: MagicMock
    operations: MagicMock
    bus: BaseEventBus


@pytest.fixture
def app() -> App:
    state = State(ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None))
    bus = BaseEventBus()
    registry = Registry()
    registry.register("demo", OldAdapter)
    writeback = MagicMock()
    tabs = TabService(state, registry, writeback)
    workspace = WorkspaceService(state, tabs, bus)
    loader = Loader()
    operations = MagicMock(return_value=0)
    access = ExperimentAccess()
    return App(
        ExperimentReloadService(
            state=state,
            workspace=workspace,
            registry=registry,
            loader=loader,
            active_operations=operations,
            access=access,
        ),
        state,
        workspace,
        TabControlFacet(
            state=state, tab=tabs, workspace=workspace, bus=bus, access=access
        ),
        registry,
        loader,
        access,
        writeback,
        operations,
        bus,
    )


def test_reload_preserves_config_order_selection_and_uses_new_instances(
    app: App,
) -> None:
    first = app.controls.new_tab("demo")
    second = app.controls.new_tab("demo")
    app.state.set_active_tab(first)
    app.state.get_tab(first).run.result = object()
    draft = object()
    app.state.get_tab(first).analysis.writeback_draft = draft
    expected = app.workspace.capture_session()
    closed: list[str] = []
    app.bus.subscribe(TabClosedPayload, lambda event: closed.append(event.tab_id))

    report = app.reload.reload_confirmed(app.reload.prepare_reload())

    assert report.restored_tabs == 2
    assert report.rejected_tabs == ()
    assert app.workspace.capture_session() == expected
    assert closed == [first, second]
    assert set(app.state.list_tab_ids()).isdisjoint({first, second})
    assert all(isinstance(tab.adapter, NewAdapter) for tab in app.state.tabs.values())
    assert all(tab.run.result is None for tab in app.state.tabs.values())
    app.writeback.teardown_draft.assert_called_once_with(draft)


def test_prepare_without_confirmation_does_not_close_tabs(app: App) -> None:
    original = app.controls.new_tab("demo")
    app.reload.prepare_reload()
    assert app.state.list_tab_ids() == [original]
    assert app.loader.loads == 0


@pytest.mark.parametrize("change", ["new_tab", "selection", "version", "result"])
def test_confirmation_cannot_discard_changed_state(app: App, change: str) -> None:
    first = app.controls.new_tab("demo")
    app.controls.new_tab("demo")
    preview = app.reload.prepare_reload()
    if change == "new_tab":
        app.controls.new_tab("demo")
    elif change == "selection":
        app.state.set_active_tab(first)
    elif change == "version":
        app.state.version.bump("context")
    else:
        app.state.get_tab(first).run.result = object()
    ids = app.state.list_tab_ids()
    with pytest.raises(FailedPreconditionError, match="changed while confirming"):
        app.reload.reload_confirmed(preview)
    assert app.state.list_tab_ids() == ids
    assert app.loader.loads == 0


@pytest.mark.parametrize("busy", ["operation", "save", "analysis", "shutdown"])
def test_busy_or_shutdown_prevents_reload(app: App, busy: str) -> None:
    tab_id = app.controls.new_tab("demo")
    preview = app.reload.prepare_reload()
    if busy == "operation":
        app.operations.return_value = 1
    elif busy == "save":
        app.state.get_tab(tab_id).is_saving_data = True
    elif busy == "analysis":
        app.state.get_tab(tab_id).is_analyzing = True
    else:
        app.access.shutting_down = True
    with pytest.raises(FailedPreconditionError):
        app.reload.reload_confirmed(preview)
    assert app.state.list_tab_ids() == [tab_id]


def test_capture_failure_keeps_tabs_and_catalog(
    app: App, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab_id = app.controls.new_tab("demo")
    preview = app.reload.prepare_reload()
    monkeypatch.setattr(
        app.workspace,
        "capture_session",
        MagicMock(side_effect=ValueError("bad snapshot")),
    )
    with pytest.raises(ValueError, match="bad snapshot"):
        app.reload.reload_confirmed(preview)
    assert app.state.list_tab_ids() == [tab_id]
    assert isinstance(app.registry.create("demo"), OldAdapter)


def test_late_preflight_failure_does_not_close_tabs(app: App) -> None:
    tab_id = app.controls.new_tab("demo")
    preview = app.reload.prepare_reload()
    app.loader.prepare_failure = CatalogReloadError("invalid source")
    with pytest.raises(CatalogReloadError):
        app.reload.reload_confirmed(preview)
    assert app.state.list_tab_ids() == [tab_id]


def test_failed_catalog_retains_ram_snapshot_and_retry_uses_it(app: App) -> None:
    app.controls.new_tab("demo")
    expected = app.workspace.capture_session()
    app.loader.failure = CatalogReloadError("bad import")
    report = app.reload.reload_confirmed(app.reload.prepare_reload())
    assert report.catalog_error == "bad import"
    assert app.state.list_tab_ids() == []
    assert app.registry.list_names() == []
    assert app.reload.can_retry_reload
    with pytest.raises(FailedPreconditionError):
        app.controls.new_tab("demo")
    app.loader.failure = None
    assert app.reload.retry_failed_reload().restored_tabs == 1
    assert app.workspace.capture_session() == expected
    assert not app.reload.can_retry_reload
    with pytest.raises(FailedPreconditionError):
        app.reload.retry_failed_reload()


@pytest.mark.parametrize("restart_required", [False, True])
def test_retry_preflight_preserves_recovery_and_error_disposition(
    app: App, restart_required: bool
) -> None:
    app.controls.new_tab("demo")
    app.loader.failure = CatalogReloadError("broken catalog")
    app.reload.reload_confirmed(app.reload.prepare_reload())
    app.loader.failure = None
    app.loader.prepare_failure = CatalogReloadError(
        "preflight rejected", restart_required=restart_required
    )

    report = app.reload.retry_failed_reload()

    assert report.catalog_error == "preflight rejected"
    assert report.restart_required is restart_required
    assert app.reload.can_retry_reload is not restart_required
    assert app.state.list_tab_ids() == []
    assert app.loader.loads == 1
    app.loader.prepare_failure = None
    if restart_required:
        with pytest.raises(FailedPreconditionError):
            app.reload.retry_failed_reload()
    else:
        assert app.reload.retry_failed_reload().restored_tabs == 1


def test_restart_required_failure_cannot_retry(app: App) -> None:
    app.controls.new_tab("demo")
    app.loader.failure = CatalogReloadError("unsafe", restart_required=True)
    report = app.reload.reload_confirmed(app.reload.prepare_reload())
    assert report.restart_required
    assert not app.reload.can_retry_reload
    with pytest.raises(FailedPreconditionError, match="Restart"):
        app.reload.retry_failed_reload()


def test_partial_restore_retries_only_skipped_entries_without_stealing_selection(
    app: App,
) -> None:
    app.registry.register("missing", OldAdapter)
    app.controls.new_tab("demo")
    app.controls.new_tab("missing")
    report = app.reload.reload_confirmed(app.reload.prepare_reload())
    assert report.restored_tabs == 1
    assert len(report.rejected_tabs) == 1
    retained = app.state.list_tab_ids()[0]
    app.state.set_active_tab(retained)
    app.registry.register("missing", NewAdapter)
    report = app.reload.retry_skipped_tabs()
    assert report.restored_tabs == 1
    assert app.state.list_tab_ids()[0] == retained
    assert app.state.active_tab_id == retained
    assert app.reload.skipped_count == 0
    assert app.reload.retry_skipped_tabs().restored_tabs == 0
    assert len(app.state.list_tab_ids()) == 2


def test_invalid_defaults_are_reported_without_discarding_other_tabs(app: App) -> None:
    class InvalidAdapter(OldAdapter):
        def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
            raise ValueError("missing calibration")

    app.registry.register("bad", OldAdapter)
    app.loader.candidate.register("bad", InvalidAdapter)
    app.controls.new_tab("bad")
    app.controls.new_tab("demo")
    report = app.reload.reload_confirmed(app.reload.prepare_reload())
    assert report.restored_tabs == 1
    assert "invalid adapter defaults" in report.rejected_tabs[0].message
    assert app.reload.skipped_count == 1


def test_driving_gate_and_nested_reload_are_blocked_during_import(app: App) -> None:
    app.controls.new_tab("demo")

    def during_load() -> None:
        with pytest.raises(FailedPreconditionError):
            app.controls.new_tab("demo")
        with pytest.raises(FailedPreconditionError):
            app.reload.prepare_reload()

    app.loader.during_load = during_load
    assert app.reload.reload_confirmed(app.reload.prepare_reload()).restored_tabs == 1


def test_new_reload_discard_is_explicit_in_preview(app: App) -> None:
    app.registry.register("missing", OldAdapter)
    app.controls.new_tab("missing")
    app.reload.reload_confirmed(app.reload.prepare_reload())
    preview = app.reload.prepare_reload()
    assert preview.skipped_count == 1
    assert app.reload.skipped_count == 1
    app.reload.reload_confirmed(preview)
    assert app.reload.skipped_count == 0


def test_ram_snapshot_is_detached_from_retired_cfg(app: App) -> None:
    key = app.controls.new_tab("demo")
    retired = app.state.get_tab(key)
    expected = schema_to_raw(retired.cfg_schema)
    app.loader.failure = CatalogReloadError("bad import")
    app.reload.reload_confirmed(app.reload.prepare_reload())
    retired.cfg_schema = CfgSchema(
        spec=retired.cfg_schema.spec,
        value=CfgSectionValue(fields={"knob": DirectValue(99)}),
    )
    app.loader.failure = None
    app.reload.retry_failed_reload()
    fresh = app.state.get_tab(app.state.list_tab_ids()[0])
    assert schema_to_raw(fresh.cfg_schema) == expected
