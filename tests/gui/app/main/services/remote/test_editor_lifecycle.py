"""Editor ownership through the shipped NDJSON connection lifecycle."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.gui.app.main.adapter import ContextReadiness, ExpContext
from zcu_tools.gui.app.main.controller import Controller
from zcu_tools.gui.app.main.registry import Registry
from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError
from zcu_tools.gui.app.main.services.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.main.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.gui.session.services.io_manager import IOManager
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.program.v2 import WaveformCfgFactory

pytestmark = pytest.mark.requires_loopback


class Client:
    def __init__(self, port: int, scheduler: ManualOwnerScheduler) -> None:
        self.socket = socket.create_connection(("127.0.0.1", port), timeout=1)
        self.scheduler = scheduler
        self.buffer = bytearray()
        self.sequence = 0

    def close(self) -> None:
        self.socket.close()
        self.buffer.clear()

    def call(self, method: str, **params: object) -> dict[str, object]:
        self.sequence += 1
        rid = str(self.sequence)
        self.socket.settimeout(1)
        self.socket.sendall(
            (
                json.dumps({"id": rid, "method": method, "params": params}) + "\n"
            ).encode()
        )
        self.socket.setblocking(False)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                chunk = self.socket.recv(65536)
                assert chunk, "connection closed before reply"
                self.buffer.extend(chunk)
            except BlockingIOError:
                pass
            while b"\n" in self.buffer:
                line, _, remaining = self.buffer.partition(b"\n")
                self.buffer = bytearray(remaining)
                response: dict[str, object] = json.loads(line)
                if response.get("id") == rid:
                    return response
            self.scheduler.pump_once(block=True, timeout=0.005)
        pytest.fail(f"no reply for {method}")

    def result(self, method: str, **params: object) -> dict[str, object]:
        response = self.call(method, **params)
        assert response["ok"] is True, response
        result = response["result"]
        assert isinstance(result, dict)
        return result

    def open_editor(self) -> str:
        result = self.result("editor.new", item_kind="waveform", from_name="seed")
        editor_id = result["editor_id"]
        assert isinstance(editor_id, str)
        return editor_id

    def assert_missing(self, editor_id: str) -> None:
        for method, params in (
            ("editor.get", {}),
            ("editor.set_field", {"path": "length", "value": 0.5}),
        ):
            response = self.call(method, editor_id=editor_id, **params)
            assert response["ok"] is False
            error = response["error"]
            assert isinstance(error, dict)
            assert error["code"] == "invalid_params"


class Connections:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.scheduler = ManualOwnerScheduler()
        self.library = ModuleLibrary()
        self.library.waveforms["seed"] = WaveformCfgFactory.from_raw(
            {"style": "const", "length": 0.1}
        )
        context = ExpContext(
            md=MetaDict(),
            ml=self.library,
            soc=None,
            soccfg=None,
            result_dir=str(tmp_path),
            database_path=str(tmp_path / "database"),
            active_label="test",
            readiness=ContextReadiness.ACTIVE,
        )
        self.controller = Controller(
            State(context),
            Registry(),
            IOManager(),
            None,
            BaseEventBus(),
            project_root=str(tmp_path),
        )
        self.reclaimed: list[tuple[list[str], int]] = []
        discard = self.controller.discard_cfg_editors

        def record_discard(editor_ids: list[str]) -> None:
            self.reclaimed.append((list(editor_ids), threading.get_ident()))
            discard(editor_ids)

        monkeypatch.setattr(self.controller, "discard_cfg_editors", record_discard)
        self.service = RemoteControlAdapter(
            self.controller, ControlOptions(port=0), self.scheduler
        )
        self.port = self.service.start()
        self.clients: list[Client] = []

    def connect(self) -> Client:
        client = Client(self.port, self.scheduler)
        self.clients.append(client)
        client.result("resources.versions")
        return client

    def wait(self, predicate: Callable[[], bool]) -> None:
        deadline = time.monotonic() + 3
        while not predicate() and time.monotonic() < deadline:
            self.scheduler.pump_once(block=True, timeout=0.01)
        assert predicate(), "owner callback did not complete"

    def close(self) -> None:
        self.service.stop()
        for client in self.clients:
            client.close()
        self.scheduler.pump_all()


@pytest.fixture
def connections(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Connections]:
    connections = Connections(tmp_path, monkeypatch)
    try:
        yield connections
    finally:
        connections.close()
        qapp.processEvents()


def test_disconnect_reclaims_only_that_clients_editors_on_owner(
    connections: Connections,
) -> None:
    a, b = connections.connect(), connections.connect()
    a_ids = [a.open_editor(), a.open_editor()]
    b_id = b.open_editor()
    before = b.result("editor.get", editor_id=b_id)
    a.close()
    connections.wait(lambda: bool(connections.reclaimed))
    assert len(connections.reclaimed) == 1
    reclaimed, owner = connections.reclaimed[0]
    assert set(reclaimed) == set(a_ids)
    assert owner == threading.get_ident()
    for editor_id in a_ids:
        b.assert_missing(editor_id)
    b.result("editor.set_field", editor_id=b_id, path="length", value=0.5)
    assert b.result("editor.get", editor_id=b_id) != before


@pytest.mark.parametrize("terminal", ["editor.commit", "editor.discard"])
def test_successful_terminal_is_not_reclaimed_again(
    connections: Connections,
    terminal: str,
) -> None:
    a, observer = connections.connect(), connections.connect()
    finished, unfinished = a.open_editor(), a.open_editor()
    params: dict[str, object] = {"editor_id": finished}
    if terminal == "editor.commit":
        params["name"] = "saved"
    a.result(terminal, **params)
    observer.assert_missing(finished)
    if terminal == "editor.commit":
        assert (
            connections.library.waveforms["saved"]
            == connections.library.waveforms["seed"]
        )
    a.close()
    connections.wait(lambda: bool(connections.reclaimed))
    assert connections.reclaimed == [([unfinished], threading.get_ident())]
    observer.assert_missing(unfinished)


def test_failed_commit_still_belongs_to_connection(
    connections: Connections,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a, observer = connections.connect(), connections.connect()
    editor_id = a.open_editor()

    def reject_commit(editor_id: str, name: str) -> None:
        raise InvalidInputError(f"cannot commit {editor_id} as {name}")

    with monkeypatch.context() as patch:
        patch.setattr(connections.controller, "commit_cfg_editor", reject_commit)
        response = a.call("editor.commit", editor_id=editor_id, name="saved")
    assert response["ok"] is False
    error = response["error"]
    assert isinstance(error, dict)
    assert error["code"] == "invalid_params"
    a.result("editor.get", editor_id=editor_id)
    a.close()
    connections.wait(lambda: bool(connections.reclaimed))
    assert connections.reclaimed == [([editor_id], threading.get_ident())]
    observer.assert_missing(editor_id)


def test_stop_reclaims_live_clients_synchronously_and_only_once(
    connections: Connections,
) -> None:
    a, b = connections.connect(), connections.connect()
    connections.connect()
    ids = [a.open_editor(), b.open_editor()]
    connections.service.stop()
    assert {
        editor_id for batch, _ in connections.reclaimed for editor_id in batch
    } == set(ids)
    assert len(connections.reclaimed) == 2
    assert all(owner == threading.get_ident() for _, owner in connections.reclaimed)
    for editor_id in ids:
        with pytest.raises(CfgEditorError):
            connections.controller.get_cfg_editor_draft(editor_id)
    before = list(connections.reclaimed)
    connections.service.stop()
    connections.scheduler.pump_all()
    assert connections.reclaimed == before


def test_non_editor_request_and_empty_disconnect_do_not_reclaim(
    connections: Connections,
) -> None:
    client = connections.connect()
    client.result("resources.versions")
    client.close()
    connections.wait(lambda: not connections.service.has_live_client())
    connections.service.stop()
    connections.scheduler.pump_all()
    assert connections.reclaimed == []
