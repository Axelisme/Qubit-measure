from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.services.persistence_types import (
    PersistedDeviceEntry,
    PersistedStartup,
)
from zcu_tools.gui.app.measure.state import DeviceState, DeviceStatus, State
from zcu_tools.gui.result_scope import ResultScopeError, ResultScopeManager
from zcu_tools.gui.session.services.project_settings import (
    ConnectionPreferences,
    ProjectRequest,
    ProjectSettingsService,
    SetupPreferences,
)
from zcu_tools.gui.session.state import DEFAULT_LEFT_PANEL_WIDTH
from zcu_tools.resources.context import MetaDict, ModuleLibrary


def _make_service(
    tmp_path,
) -> tuple[ProjectSettingsService, MagicMock, MagicMock, State]:
    context = MagicMock()
    devices = MagicMock()
    state = State(MagicMock())
    svc = ProjectSettingsService(context, devices, state, ResultScopeManager(tmp_path))
    return svc, context, devices, state


def test_apply_project_resolves_generated_scope_and_records_prefs(tmp_path) -> None:
    svc, context, _, state = _make_service(tmp_path)
    req = ProjectRequest("chip", "qubit", "res")

    resolved = svc.apply_project(req)

    expected_result = str(tmp_path / "result" / "chip" / "qubit")
    args = context.set_project_context.call_args.args
    assert isinstance(args[0], MetaDict)
    assert isinstance(args[1], ModuleLibrary)
    assert args[2:6] == ("chip", "qubit", "res", expected_result)
    assert args[6].startswith(str(tmp_path / "Database" / "chip" / "qubit"))
    context.setup_project.assert_called_once_with(expected_result)
    assert resolved.result_dir == expected_result
    assert (tmp_path / "result" / "chip" / "qubit" / "params.json").exists()
    # Recorded as prefs (not written to disk here).
    prefs = state.preferences
    assert prefs.chip_name == "chip"
    assert prefs.qub_name == "qubit"
    assert prefs.scope_id == resolved.scope_id
    assert prefs.result_dir == expected_result
    assert prefs.database_path.startswith(str(tmp_path / "Database" / "chip" / "qubit"))


def test_apply_same_project_again_leaves_the_context_alone(tmp_path) -> None:
    svc, context, _, state = _make_service(tmp_path)
    req = ProjectRequest("chip", "qubit", "res")
    first = svc.apply_project(req)
    context.reset_mock()
    # The context now carries the applied identity, as ContextService would set it.
    state.set_context(
        MagicMock(
            chip_name="chip",
            qub_name="qubit",
            res_name="res",
            result_dir=first.result_dir,
            database_path=first.database_path,
        )
    )

    again = svc.apply_project(req)

    assert again == first
    context.set_project_context.assert_not_called()
    context.setup_project.assert_not_called()


def test_apply_project_with_unknown_scope_changes_nothing(tmp_path) -> None:
    svc, context, _, state = _make_service(tmp_path)
    req = ProjectRequest(
        "chip", "qubit", "res", scope_id=str(tmp_path / "result" / "missing" / "scope")
    )

    with pytest.raises(ResultScopeError):
        svc.apply_project(req)

    context.set_project_context.assert_not_called()
    context.setup_project.assert_not_called()
    assert state.preferences.chip_name == ""
    assert state.preferences.scope_id == ""


def test_apply_project_uses_discovered_scope_id(tmp_path) -> None:
    manager = ResultScopeManager(tmp_path)
    scope = manager.ensure_scope(chip_name="chip", qub_name="qubit")
    svc, context, _, _ = _make_service(tmp_path)
    req = ProjectRequest("chip", "qubit", "res", scope_id=scope.scope_id)

    resolved = svc.apply_project(req)

    assert resolved.scope_id == scope.scope_id
    context.setup_project.assert_called_once_with(scope.result_dir)


def test_remember_connection_records_prefs() -> None:
    svc, _, _, state = _make_service("/tmp")
    svc.remember_connection(ConnectionPreferences(ip="10.0.0.2", port=1234))
    assert state.preferences.ip == "10.0.0.2"
    assert state.preferences.port == 1234


def test_setup_preferences_default_before_anything_is_remembered() -> None:
    svc, _, _, _ = _make_service("/tmp")

    assert svc.get_setup_preferences() == SetupPreferences(
        chip_name="",
        qub_name="",
        res_name="",
        scope_id="",
        ip="192.168.10.1",
        port=8887,
    )


def test_setup_preferences_follow_apply_and_connection(tmp_path) -> None:
    svc, _, _, _ = _make_service(tmp_path)

    resolved = svc.apply_project(ProjectRequest("chip", "qubit", "res"))
    svc.remember_connection(ConnectionPreferences(ip="10.0.0.2", port=1234))

    assert svc.get_setup_preferences() == SetupPreferences(
        chip_name="chip",
        qub_name="qubit",
        res_name="res",
        scope_id=resolved.scope_id,
        ip="10.0.0.2",
        port=1234,
    )


def test_scope_id_capture_restore_roundtrips_with_connection(tmp_path) -> None:
    manager = ResultScopeManager(tmp_path)
    scope = manager.ensure_scope(chip_name="chip", qub_name="qubit")
    svc, _, _, state = _make_service(tmp_path)

    svc.apply_project(
        ProjectRequest(
            "chip",
            "qubit",
            "res",
            scope_id=scope.scope_id,
        )
    )
    svc.remember_connection(ConnectionPreferences(ip="10.0.0.2", port=1234))

    memento = svc.capture_settings(left_panel_width=321)

    assert memento.scope_id == scope.scope_id
    assert memento.ip == "10.0.0.2"
    assert memento.port == 1234

    restored_svc, _, _, restored_state = _make_service(tmp_path)
    restored_svc.restore_settings(memento)

    assert restored_state.preferences.scope_id == scope.scope_id
    assert restored_state.preferences.ip == "10.0.0.2"
    assert restored_state.preferences.port == 1234


def test_derive_project_paths_scopes_under_chip_qubit() -> None:
    from datetime import datetime

    from zcu_tools.gui.session.services.project_settings import derive_project_paths

    result_dir, database_path = derive_project_paths("Q5_2D", "Q1", "/root")
    assert result_dir == "/root/result/Q5_2D/Q1"
    today = datetime.today()
    yy, mm, dd = today.strftime("%Y-%m-%d").split("-")
    assert database_path == f"/root/Database/Q5_2D/Q1/{yy}/{mm}/Data_{mm}{dd}"


def _dev(name: str, *, remember: bool) -> DeviceState:
    return DeviceState(
        name=name,
        type_name="FakeDevice",
        address=f"{name}-addr",
        status=DeviceStatus.CONNECTED,
        remember=remember,
    )


def test_capture_settings_projects_remembered_devices_and_prefs() -> None:
    """capture_settings re-projects the remember=True device set from State and
    composes it with the prefs + the given left-panel width into a memento."""
    svc, _, _, state = _make_service("/tmp")
    svc.remember_connection(ConnectionPreferences(ip="host", port=8887))
    state.put_device(_dev("flux", remember=True))
    state.put_device(_dev("probe", remember=True))
    state.put_device(_dev("scratch", remember=False))

    memento = svc.capture_settings(left_panel_width=321)

    assert isinstance(memento, PersistedStartup)
    assert memento.ip == "host"
    assert memento.left_panel_width == 321
    assert sorted(e.name for e in memento.devices) == ["flux", "probe"]


def test_left_panel_width_is_default_until_restored_then_queryable() -> None:
    svc, _, _, _ = _make_service("/tmp")
    assert svc.get_left_panel_width() == DEFAULT_LEFT_PANEL_WIDTH

    svc.restore_settings(PersistedStartup(left_panel_width=321))

    assert svc.get_left_panel_width() == 321


def test_restore_settings_seeds_prefs_and_registers_devices() -> None:
    svc, context, devices, state = _make_service("/tmp")
    data = PersistedStartup(
        chip_name="chip",
        qub_name="qub",
        res_name="res",
        scope_id="/tmp/result/chip/qub",
        ip="host",
        port=9999,
        devices=(
            PersistedDeviceEntry(type_name="FakeDevice", name="flux", address="a"),
        ),
        left_panel_width=222,
    )

    svc.restore_settings(data)

    # Prefs seeded (so the setup dialog prefills) — project NOT auto-applied.
    assert state.preferences.chip_name == "chip"
    assert state.preferences.scope_id == "/tmp/result/chip/qub"
    assert state.preferences.port == 9999
    assert state.preferences.left_panel_width == 222
    (entries,) = devices.register_remembered_devices.call_args.args
    assert entries[0].name == "flux"
    assert svc.get_setup_preferences().ip == "host"
    context.set_project_context.assert_not_called()
    context.setup_project.assert_not_called()


def test_connection_request_validates_port() -> None:
    with pytest.raises(ValueError, match="port"):
        ConnectionPreferences(ip="host", port=0)
