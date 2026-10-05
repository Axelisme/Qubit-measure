"""SaveService driving readiness and direct-operation policy contracts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.adapter import ContextReadiness, SavePaths, SessionEnv
from zcu_tools.gui.app.measure.artifact_tracker import ArtifactKey, ArtifactKind
from zcu_tools.gui.app.measure.catalog import ExperimentAccess
from zcu_tools.gui.app.measure.events.tab import TabInteractionChangedPayload
from zcu_tools.gui.app.measure.figure_export import SAVE_DPI
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.app_services import build_app_services
from zcu_tools.gui.app.measure.services.guard import SavePermit
from zcu_tools.gui.app.measure.services.ports import SaveDestination
from zcu_tools.gui.app.measure.services.save import SaveService
from zcu_tools.gui.app.measure.state import Session, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.plotting.plots import NonPresentingHost, Plots


def _make_service(
    *,
    handles: OperationHandles | None = None,
    gate: MagicMock | None = None,
    bus: EventBus | None = None,
    access: ExperimentAccess | None = None,
) -> tuple[SaveService, State, MagicMock]:
    state = State(MagicMock())
    adapter = MagicMock()
    adapter.make_save_paths.return_value = SavePaths("/db/data.h5", "/result/base.png")
    state.add_tab(
        "tab",
        Session(adapter_name="fake", adapter=adapter, cfg=MagicMock()),
    )
    state.update_tab_result("tab", object())
    bg = MagicMock()  # BackgroundRunner stand-in; submit() is inspected per-test
    bus = bus if bus is not None else EventBus()
    runner = OperationRunner(
        gate if gate is not None else MagicMock(),
        handles if handles is not None else OperationHandles(),
        MagicMock(),
        bg,
        bus,
    )
    svc = SaveService(state, runner, bus, owner_scheduler=MagicMock(), access=access)
    return svc, state, bg


def test_save_preflight_default_access_is_available_without_side_effects() -> None:
    bus = EventBus()
    facts: list[TabInteractionChangedPayload] = []
    bus.subscribe(TabInteractionChangedPayload, facts.append)
    svc, state, background = _make_service(bus=bus)
    before = state.get_artifact_snapshots("tab")

    assert svc.require_save_available("tab") is None

    assert state.get_artifact_snapshots("tab") == before
    assert not state.is_tab_busy("tab")
    assert svc.active_save_operations() == ()
    assert facts == []
    background.submit.assert_not_called()


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
    svc, state, background = _make_service(bus=bus, access=access)
    state.set_tab_analyzing("tab", analyzing=busy)
    before = state.get_artifact_snapshots("tab")

    if reason is None:
        assert svc.require_save_available("tab") is None
    else:
        with pytest.raises(FailedPreconditionError, match=reason):
            svc.require_save_available("tab")

    assert state.get_artifact_snapshots("tab") == before
    assert state.is_tab_busy("tab") is busy
    assert svc.active_save_operations() == ()
    assert facts == []
    background.submit.assert_not_called()


@pytest.mark.parametrize("entrypoint", ("data", "artifacts", "image"))
def test_direct_save_operations_keep_busy_only_policy_during_shutdown(
    entrypoint: str, tmp_path: Path, qapp
) -> None:
    access = ExperimentAccess()
    access.shutting_down = True
    handles = OperationHandles()
    svc, state, background = _make_service(access=access, handles=handles)
    figure = MagicMock()
    figure.get_size_inches.return_value = (6.0, 4.0)
    plots = Plots(NonPresentingHost())
    plots.adopt("fit", figure)
    plots.finish()
    state.update_tab_analyze("tab", object(), plots)
    permit = SavePermit(tab_id="tab")
    path = str(tmp_path / "output" / "saved")
    with pytest.raises(FailedPreconditionError, match="unavailable"):
        svc.require_save_available("tab")

    if entrypoint == "image":
        svc.save_image_sync(permit, ArtifactKey(ArtifactKind.ANALYSIS, "fit"), path)
        figure.savefig.assert_called_once_with(f"{path}.png", dpi=SAVE_DPI)
        assert handles.live_count() == 0
        assert not state.is_tab_busy("tab")
        background.submit.assert_not_called()
    else:
        if entrypoint == "data":
            submission = svc.start_save_data(permit, path)
        else:
            submission = svc.start_save_artifacts(
                permit, (SaveDestination(ArtifactKey(ArtifactKind.DATA), path),)
            )
        assert submission.operation_id > 0
        assert handles.live_count() == 1
        assert state.is_tab_busy("tab")
        background.submit.assert_called_once()


@pytest.mark.parametrize("access_state", ("switching", "unavailable", "shutting_down"))
def test_assembled_save_driving_entry_observes_shared_catalog_access(
    access_state: str, tmp_path: Path, qapp
) -> None:
    state = State(
        SessionEnv(
            md=MagicMock(),
            ml=MagicMock(),
            soc=None,
            soccfg=None,
            result_dir=str(tmp_path / "result"),
            database_path=str(tmp_path / "database"),
            readiness=ContextReadiness.ACTIVE,
        )
    )
    state.add_tab(
        "tab", Session(adapter_name="fake", adapter=MagicMock(), cfg=MagicMock())
    )
    state.update_tab_result("tab", object())
    services = build_app_services(
        state=state,
        bus=EventBus(),
        registry=Registry(),
        io_manager=MagicMock(),
        cfg_editor_ctrl=MagicMock(),
        progress_transport=MagicMock(),
        notify_info=lambda _message: None,
        resource_versions=lambda: {},
        render_host=lambda: None,
        owner_scheduler=QtOwnerScheduler(),
        project_root=str(tmp_path),
    )
    try:
        services.experiment_access.switching = access_state == "switching"
        services.experiment_access.available = access_state != "unavailable"
        services.experiment_access.shutting_down = access_state == "shutting_down"
        path = tmp_path / "output" / "data.h5"
        before = state.get_artifact_snapshots("tab")

        with pytest.raises(FailedPreconditionError, match="unavailable"):
            services.save_control.save_data("tab", str(path), comment="new")

        assert not path.parent.exists()
        assert not state.is_tab_busy("tab")
        assert state.get_tab("tab").save.comment == ""
        assert state.get_artifact_snapshots("tab") == before
        assert services.handles.live_count() == 0
        services.experiment_access.switching = False
        services.experiment_access.available = True
        services.experiment_access.shutting_down = False
        assert services.save.require_save_available("tab") is None
    finally:
        services.background.quiesce()
        services.background.deleteLater()
        qapp.processEvents()
