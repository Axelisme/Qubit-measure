"""Synchronous Save readiness through service and assembled driving seams."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.gui.app.measure.catalog import ExperimentAccess
from zcu_tools.gui.app.measure.events.tab import TabInteractionChangedPayload
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.app_services import build_app_services
from zcu_tools.gui.app.measure.services.save import SaveService
from zcu_tools.gui.app.measure.state import Session, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler

from tests.gui.app.measure._form_fakes import make_form
from tests.gui.app.measure._native_save_support import (
    GUI_ORIGIN,
    native_save_session,
)
from tests.gui.app.measure._synthetic import ALIAS_PRESET


def _make_service(
    *, bus: EventBus, access: ExperimentAccess | None = None
) -> tuple[SaveService, State]:
    state = State(MagicMock())
    state.add_tab(
        "tab",
        Session(
            adapter_name="fake",
            adapter=MagicMock(),
            cfg=make_form(Form(ConfigBase, {}), state=state),
        ),
    )
    state.update_tab_result("tab", object())
    return SaveService(state, bus, access=access), state


def test_save_preflight_default_access_is_available_without_side_effects() -> None:
    bus = EventBus()
    facts: list[TabInteractionChangedPayload] = []
    bus.subscribe(TabInteractionChangedPayload, facts.append)
    svc, state = _make_service(bus=bus)
    before = state.get_artifact_snapshots("tab")

    assert svc.require_save_available("tab") is None

    assert state.get_artifact_snapshots("tab") == before
    assert not state.is_tab_busy("tab")
    assert facts == []


@pytest.mark.parametrize(
    ("switching", "available", "shutting_down", "busy", "reason"),
    [
        (False, True, False, False, None),
        (True, True, False, False, "unavailable"),
        (False, False, False, False, "unavailable"),
        (False, True, True, False, "unavailable"),
        (False, True, False, True, "busy"),
        (True, True, False, True, "unavailable"),
        (False, False, False, True, "unavailable"),
        (False, True, True, True, "unavailable"),
    ],
)
def test_save_preflight_observes_access_before_busy_without_side_effects(
    switching: bool,
    available: bool,
    shutting_down: bool,
    busy: bool,
    reason: str | None,
) -> None:
    access = ExperimentAccess()
    access.switching = switching
    access.available = available
    access.shutting_down = shutting_down
    bus = EventBus()
    facts: list[TabInteractionChangedPayload] = []
    bus.subscribe(TabInteractionChangedPayload, facts.append)
    svc, state = _make_service(bus=bus, access=access)
    state.set_tab_analyzing("tab", analyzing=busy)
    before = state.get_artifact_snapshots("tab")

    if reason is None:
        assert svc.require_save_available("tab") is None
    else:
        with pytest.raises(FailedPreconditionError, match=reason):
            svc.require_save_available("tab")

    assert state.get_artifact_snapshots("tab") == before
    assert state.is_tab_busy("tab") is busy
    assert facts == []


@pytest.mark.parametrize("access_state", ("switching", "unavailable", "shutting_down"))
def test_assembled_save_driving_entry_observes_shared_catalog_access(
    access_state: str, tmp_path: Path, qapp: QApplication
) -> None:
    with native_save_session(tmp_path, qapp) as session:
        services = build_app_services(
            alias_presets={"default": ALIAS_PRESET},
            state=session.state,
            bus=session.bus,
            registry=Registry(),
            progress_transport=MagicMock(),
            resource_versions=lambda: {},
            render_host=lambda: None,
            owner_scheduler=QtOwnerScheduler(),
            project_root=str(tmp_path),
        )
        try:
            services.experiment_access.switching = access_state == "switching"
            services.experiment_access.available = access_state != "unavailable"
            services.experiment_access.shutting_down = access_state == "shutting_down"
            before = session.state.get_artifact_snapshots("tab")

            with pytest.raises(FailedPreconditionError, match="unavailable"):
                services.save_control.save_data("tab", comment="new", origin=GUI_ORIGIN)

            assert not (session.entry.database_path / "runs").exists()
            assert not session.state.is_tab_busy("tab")
            assert session.state.get_tab("tab").save.comment == ""
            assert session.state.get_artifact_snapshots("tab") == before
            assert services.handles.live_count() == 0
            services.experiment_access.switching = False
            services.experiment_access.available = True
            services.experiment_access.shutting_down = False
            assert services.save.require_save_available("tab") is None
        finally:
            services.background.quiesce()
            services.background.deleteLater()
            qapp.processEvents()
