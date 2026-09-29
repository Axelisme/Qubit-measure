"""Owner-thread noninteractive close policy through the real GUI socket."""

import os
from dataclasses import replace
from unittest.mock import patch

import pytest
from zcu_tools.gui.app.measure.artifact_tracker import (
    ArtifactKey,
    ArtifactKind,
    ArtifactSnapshot,
    SaveStatus,
)
from zcu_tools.gui.app.measure.services.operation_control import ActiveOperation

from ._helpers import Fixture, call, open_client


@pytest.fixture()
def fx(qapp):
    fixture = Fixture()
    fixture.start()
    yield fixture
    fixture.stop()


def _key(kind: ArtifactKind) -> ArtifactKey:
    return ArtifactKey(kind, None if kind is ArtifactKind.DATA else "fit")


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
        snapshot, artifacts=(ArtifactSnapshot(_key(kind), status, None, None, False),)
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
                ArtifactKind.ANALYSIS: "analysis:fit",
                ArtifactKind.POST_ANALYSIS: "post:fit",
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
                assert allowed["result"]["pid"] == os.getpid()
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


def test_tab_close_allows_an_operation_on_another_tab(fx):
    target = fx.ctrl.new_tab("fake")
    other = fx.ctrl.new_tab("fake")
    operation = ActiveOperation(op=71, tab=other, kind="save")
    sock = open_client(fx.service.port)
    try:
        assert call(sock, "tab.snapshot", {"tab_id": target})["ok"]
        with patch.object(
            fx.service.operation_control, "active_operations", return_value=(operation,)
        ):
            reply = call(sock, "tab.close", {"tab_id": target})
        assert reply["ok"] is True
        assert not fx.ctrl.has_tab(target)
        assert fx.ctrl.has_tab(other)
    finally:
        sock.close()


def test_shutdown_collects_all_unsaved_tabs_and_checks_busy_first(fx):
    first = fx.ctrl.new_tab("fake")
    second = fx.ctrl.new_tab("fake")
    snapshots = {
        tab: replace(
            fx.service.tab_control.get_tab_snapshot(tab),
            artifacts=tuple(
                ArtifactSnapshot(_key(kind), status, None, None, False)
                for kind, status in states
            ),
        )
        for tab, states in (
            (first, [(ArtifactKind.DATA, SaveStatus.NOT_SAVED)]),
            (
                second,
                [
                    (ArtifactKind.DATA, SaveStatus.SAVED),
                    (ArtifactKind.ANALYSIS, SaveStatus.UNSAVED_CHANGES),
                    (ArtifactKind.POST_ANALYSIS, SaveStatus.NOT_SAVED),
                ],
            ),
        )
    }
    sock = open_client(fx.service.port)
    try:
        with (
            patch.object(
                fx.service.tab_control,
                "get_tab_snapshot",
                side_effect=snapshots.__getitem__,
            ),
            patch.object(
                fx.service.operation_control,
                "active_operations",
                return_value=(ActiveOperation(op=71, tab=second, kind="save"),),
            ) as active,
        ):
            busy = call(sock, "app.shutdown", {})
            assert busy["error"]["reason"] == "busy"
            active.return_value = ()
            unsaved = call(sock, "app.shutdown", {})
            assert unsaved["error"]["reason"] == "unsaved"
            assert unsaved["error"]["data"]["unsaved"] == [
                {"tab": first, "artifacts": ["data"]},
                {"tab": second, "artifacts": ["analysis:fit", "post:fit"]},
            ]
            fx.view.request_shutdown.assert_not_called()
            assert fx.ctrl.has_tab(first) and fx.ctrl.has_tab(second)
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
