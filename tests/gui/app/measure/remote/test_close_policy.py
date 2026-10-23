"""Owner-thread noninteractive close policy through the real GUI socket."""

import os
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.experiment.artifacts import ArtifactRef
from zcu_tools.gui.app.measure.artifacts import (
    ArtifactSectionSnapshot,
    ArtifactSnapshot,
    SaveStatus,
)
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.services.operation_control import ActiveOperation

from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._native_save_support import (
    AnalysisSavingAdapter,
    native_save_session,
)

from ._helpers import call, make_view, open_client


@pytest.fixture()
def fx(
    tmp_path: Path, qapp: QApplication
) -> Generator[tuple[Controller, RemoteControlAdapter, MagicMock]]:
    """Serve real tab controls over loopback with a native captured storage context."""
    with native_save_session(tmp_path, qapp) as session:
        # Retire the helper's figure-free seed before Controller owns new tabs.
        session.state.remove_tab("tab")
        registry = Registry()
        registry.register("saving", AnalysisSavingAdapter)
        view = make_view()
        ctrl = Controller(
            state=session.state, registry=registry, bus=session.bus, view=view
        )
        service = RemoteControlAdapter(
            ctrl, ControlOptions(port=0), ctrl.owner_scheduler, render_view=view
        )
        service.start()
        try:
            yield ctrl, service, view
        finally:
            service.stop()
            closed: list[bool] = []
            ctrl.begin_shutdown(lambda: closed.append(True))
            assert closed == [True]
            qapp.processEvents()


def _artifact(section: str, status: SaveStatus) -> ArtifactSectionSnapshot:
    name = "measurement_data" if section == "run" else "results"
    return ArtifactSectionSnapshot(
        section, (ArtifactSnapshot(ArtifactRef(section, name), (), status),)
    )


@pytest.mark.parametrize("method", ["tab.close", "app.shutdown"])
@pytest.mark.parametrize("section", ["run", "analysis", "post_analysis"])
@pytest.mark.parametrize("status", list(SaveStatus))
def test_remote_close_only_requires_saved_measurement_data(fx, method, section, status):
    ctrl, service, view = fx
    tab_id = ctrl.new_tab("saving")
    params = {"tab_id": tab_id} if method == "tab.close" else {}
    snapshot = service.tab_control.get_tab_snapshot(tab_id)
    artifacts = replace(snapshot, artifacts=(_artifact(section, status),))
    sock = open_client(service.port)
    try:
        assert call(sock, "tab.snapshot", {"tab_id": tab_id})["ok"]
        with patch.object(
            service.tab_control, "get_tab_snapshot", return_value=artifacts
        ):
            reply = call(sock, method, params)
            if section == "run" and status is not SaveStatus.SAVED:
                assert reply["error"]["reason"] == "unsaved"
                assert reply["error"]["data"]["unsaved"] == [
                    {"tab": tab_id, "artifacts": ["run/measurement_data"]}
                ]
                assert ctrl.has_tab(tab_id)
                view.request_shutdown.assert_not_called()
                reply = call(sock, method, {**params, "discard_unsaved": True})
            assert reply["ok"] is True
            if method == "app.shutdown":
                qt_until(
                    lambda: view.request_shutdown.called,
                    label="accepted shutdown owner continuation",
                )
                view.request_shutdown.assert_called_once()
                assert reply["result"]["pid"] == os.getpid()
            else:
                assert not ctrl.has_tab(tab_id)
    finally:
        sock.close()


@pytest.mark.parametrize("kind", ["run", "analyze", "device"])
@pytest.mark.parametrize("discard", [False, True])
def test_shutdown_rejects_every_active_operation_even_when_discarding(
    fx, kind, discard
):
    _ctrl, service, view = fx
    operation = ActiveOperation(op=71, tab=None, kind=kind)
    sock = open_client(service.port)
    try:
        with patch.object(
            service.operation_control, "active_operations", return_value=(operation,)
        ):
            reply = call(sock, "app.shutdown", {"discard_unsaved": discard})
            assert reply["error"]["reason"] == "busy"
            view.request_shutdown.assert_not_called()
    finally:
        sock.close()


def test_tab_close_allows_an_operation_on_another_tab(fx):
    ctrl, service, _view = fx
    target = ctrl.new_tab("saving")
    other = ctrl.new_tab("saving")
    operation = ActiveOperation(op=71, tab=other, kind="run")
    sock = open_client(service.port)
    try:
        assert call(sock, "tab.snapshot", {"tab_id": target})["ok"]
        with patch.object(
            service.operation_control, "active_operations", return_value=(operation,)
        ):
            reply = call(sock, "tab.close", {"tab_id": target})
        assert reply["ok"] is True
        assert not ctrl.has_tab(target)
        assert ctrl.has_tab(other)
    finally:
        sock.close()


def test_shutdown_collects_all_unsaved_tabs_and_checks_busy_first(fx):
    ctrl, service, view = fx
    first = ctrl.new_tab("saving")
    second = ctrl.new_tab("saving")
    third = ctrl.new_tab("saving")
    snapshots = {
        tab: replace(
            service.tab_control.get_tab_snapshot(tab),
            artifacts=tuple(_artifact(section, status) for section, status in states),
        )
        for tab, states in (
            (first, [("run", SaveStatus.NOT_SAVED)]),
            (
                second,
                [
                    ("run", SaveStatus.SAVED),
                    ("analysis", SaveStatus.UNSAVED_CHANGES),
                    ("post_analysis", SaveStatus.NOT_SAVED),
                ],
            ),
            (third, [("run", SaveStatus.PARTIALLY_SAVED)]),
        )
    }
    sock = open_client(service.port)
    try:
        with (
            patch.object(
                service.tab_control,
                "get_tab_snapshot",
                side_effect=snapshots.__getitem__,
            ),
            patch.object(
                service.operation_control,
                "active_operations",
                return_value=(ActiveOperation(op=71, tab=second, kind="run"),),
            ) as active,
        ):
            busy = call(sock, "app.shutdown", {})
            assert busy["error"]["reason"] == "busy"
            active.return_value = ()
            unsaved = call(sock, "app.shutdown", {})
            assert unsaved["error"]["reason"] == "unsaved"
            assert unsaved["error"]["data"]["unsaved"] == [
                {"tab": first, "artifacts": ["run/measurement_data"]},
                {"tab": third, "artifacts": ["run/measurement_data"]},
            ]
            view.request_shutdown.assert_not_called()
            assert ctrl.has_tab(first) and ctrl.has_tab(second) and ctrl.has_tab(third)
    finally:
        sock.close()


@pytest.mark.parametrize("kind", ["run", "analyze"])
def test_tab_close_does_not_discard_an_active_operation(fx, kind):
    ctrl, service, _view = fx
    tab_id = ctrl.new_tab("saving")
    operation = ActiveOperation(op=71, tab=tab_id, kind=kind)
    sock = open_client(service.port)
    try:
        assert call(sock, "tab.snapshot", {"tab_id": tab_id})["ok"]
        with patch.object(
            service.operation_control, "active_operations", return_value=(operation,)
        ):
            reply = call(sock, "tab.close", {"tab_id": tab_id, "discard_unsaved": True})
            assert reply["error"]["reason"] == "busy"
            assert ctrl.has_tab(tab_id)
    finally:
        sock.close()
