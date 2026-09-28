"""Owner-thread noninteractive close policy through the real GUI socket."""

from dataclasses import replace
from unittest.mock import patch

import pytest
from zcu_tools.gui.app.main.artifact_tracker import (
    ArtifactKind,
    ArtifactSnapshot,
    SaveStatus,
)
from zcu_tools.gui.app.main.services.operation_control import ActiveOperation

from ._helpers import Fixture, call, open_client


@pytest.fixture()
def fx(qapp):
    fixture = Fixture()
    fixture.start()
    yield fixture
    fixture.stop()


@pytest.mark.parametrize("method", ["tab.close", "app.shutdown"])
@pytest.mark.parametrize("kind", list(ArtifactKind))
@pytest.mark.parametrize("status", [SaveStatus.NOT_SAVED, SaveStatus.UNSAVED_CHANGES])
def test_remote_close_requires_discard_for_each_unsaved_artifact(
    fx, method, kind, status
):
    tab_id = fx.ctrl.new_tab("fake")
    params = {"tab_id": tab_id} if method == "tab.close" else {}
    snapshot = fx.service.tab_control.get_tab_snapshot(tab_id)
    unsaved = replace(
        snapshot, artifacts=(ArtifactSnapshot(kind, status, None, None, False),)
    )
    sock = open_client(fx.service.port)
    try:
        assert call(sock, "tab.snapshot", {"tab_id": tab_id})["ok"]
        with patch.object(
            fx.service.tab_control, "get_tab_snapshot", return_value=unsaved
        ):
            reply = call(sock, method, params)
            assert reply["error"]["reason"] == "unsaved"
            keys = {
                ArtifactKind.DATA: "data",
                ArtifactKind.ANALYSIS: "analysis",
                ArtifactKind.POST_ANALYSIS: "post",
            }
            assert reply["error"]["data"]["unsaved"] == [
                {"tab": tab_id, "artifacts": [keys[kind]]}
            ]
            assert fx.ctrl.has_tab(tab_id)
            fx.view.request_shutdown.assert_not_called()
            allowed = call(sock, method, {**params, "discard_unsaved": True})
            assert allowed["ok"] is True
            if method == "app.shutdown":
                fx.view.request_shutdown.assert_called_once()
            else:
                assert not fx.ctrl.has_tab(tab_id)
    finally:
        sock.close()


@pytest.mark.parametrize("kind", ["run", "analyze", "device", "save"])
@pytest.mark.parametrize("discard", [False, True])
def test_shutdown_rejects_every_active_operation_even_when_discarding(
    fx, kind, discard
):
    operation = ActiveOperation(op=71, tab=None, kind=kind)
    sock = open_client(fx.service.port)
    try:
        with patch.object(
            fx.service.operation_control, "active_operations", return_value=(operation,)
        ):
            reply = call(sock, "app.shutdown", {"discard_unsaved": discard})
            assert reply["error"]["reason"] == "busy"
            fx.view.request_shutdown.assert_not_called()
    finally:
        sock.close()


@pytest.mark.parametrize("kind", ["run", "analyze", "save"])
def test_tab_close_does_not_discard_an_active_operation(fx, kind):
    tab_id = fx.ctrl.new_tab("fake")
    operation = ActiveOperation(op=71, tab=tab_id, kind=kind)
    sock = open_client(fx.service.port)
    try:
        assert call(sock, "tab.snapshot", {"tab_id": tab_id})["ok"]
        with patch.object(
            fx.service.operation_control, "active_operations", return_value=(operation,)
        ):
            reply = call(sock, "tab.close", {"tab_id": tab_id, "discard_unsaved": True})
            assert reply["error"]["reason"] == "busy"
            assert fx.ctrl.has_tab(tab_id)
    finally:
        sock.close()
