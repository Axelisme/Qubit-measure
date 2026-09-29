"""Shared recording GUI and socket helpers for remote contract tests."""

from __future__ import annotations

import json
import socket
import time
from typing import Any
from unittest.mock import MagicMock

from qtpy.QtCore import QCoreApplication
from zcu_tools.experiment.v2_gui.measure.adapters.fake import FakeAdapter
from zcu_tools.experiment.v2_gui.measure.registry import register_all
from zcu_tools.gui.app.measure.adapter import ContextReadiness, SessionEnv
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import (
    ControlOptions,
    RemoteControlAdapter,
)
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.services.io_manager import IOManager
from zcu_tools.program.v2.mocksoc import make_mock_soccfg


def make_ctx() -> SessionEnv:
    return SessionEnv(
        md=MagicMock(),
        ml=MagicMock(),
        soc=MagicMock(),
        soccfg=make_mock_soccfg(),
        res_name="fake_res",
        result_dir="/tmp/zcu_result",
        database_path="/tmp/zcu_db/fake_chip/fake_qubit",
        active_label="ctx001",
        readiness=ContextReadiness.ACTIVE,
    )


def make_view() -> MagicMock:
    view = MagicMock()
    view.show_status_message = MagicMock()
    view.make_run_container = MagicMock(return_value=None)
    # tab.list_all / overview read active_tab_id off the render view; return a real
    # (JSON-serializable) snapshot so the wire reply encodes cleanly.
    view.get_view_snapshot = MagicMock(
        return_value={"active_tab_id": None, "tab_ids": []}
    )
    return view


class RemoteCoreFixture:
    """Hold strong refs to Controller + service to survive GC mid-test."""

    def __init__(self, opts: ControlOptions | None = None) -> None:
        self.state = State(make_ctx())
        self.registry = Registry()
        register_all(self.registry)
        if not self.registry.has("fake"):
            self.registry.register("fake", FakeAdapter)
        self.view = make_view()
        io_manager = MagicMock(spec=IOManager, has_project=True, has_context=True)
        io_manager.get_active_label.return_value = "ctx001"
        self.bus = EventBus()
        self.ctrl = Controller(
            state=self.state,
            registry=self.registry,
            io_manager=io_manager,
            view=self.view,
            bus=self.bus,
        )
        if opts is None:
            opts = ControlOptions(port=0)
        # tab.list_all now reads active_tab_id off the render view (a view
        # projection), so the fixture must supply one — mirror _helpers.Fixture.
        self.service = RemoteControlAdapter(
            controller=self.ctrl,
            opts=opts,
            owner_scheduler=QtOwnerScheduler(),
            render_view=self.view,
        )

    def start(self) -> int:
        return self.service.start()

    def stop(self) -> None:
        self.service.stop()


def send(sock: socket.socket, obj: dict[str, Any]) -> None:
    sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))


def recv_response(sock: socket.socket, timeout_s: float = 3.0) -> dict[str, Any]:
    """Wait for one NDJSON response, pumping the Qt event loop in between."""
    app = QCoreApplication.instance()
    assert app is not None
    deadline = time.monotonic() + timeout_s
    buf = bytearray()
    sock.setblocking(False)
    while time.monotonic() < deadline:
        try:
            chunk = sock.recv(4096)
            if chunk:
                buf.extend(chunk)
                if b"\n" in buf:
                    line, _, _ = bytes(buf).partition(b"\n")
                    return json.loads(line.decode("utf-8"))
                    # rest discarded — single-response helper
            else:
                # peer closed cleanly mid-recv → return ""
                if buf and b"\n" in buf:
                    line, _, _ = bytes(buf).partition(b"\n")
                    return json.loads(line.decode("utf-8"))
                raise AssertionError("peer closed without a response line")
        except BlockingIOError:
            pass
        app.processEvents()
        time.sleep(0.005)
    raise AssertionError(f"no response within {timeout_s}s")


def open_client(port: int) -> socket.socket:
    return socket.create_connection(("127.0.0.1", port), timeout=1.0)
