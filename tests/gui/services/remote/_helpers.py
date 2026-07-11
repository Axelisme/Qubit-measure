"""Shared test helpers for the RemoteControlAdapter remote test suites.

Single source of fixture + socket-pumping helpers so the remote test files do
not duplicate the boilerplate. Anything suite-specific (e.g. event-push
reception) lives next to the tests that need it.
"""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Callable, Mapping
from typing import Any
from unittest.mock import MagicMock

from qtpy.QtCore import QCoreApplication
from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
from zcu_tools.experiment.v2_gui.registry import register_all
from zcu_tools.gui.app.main.adapter import ContextReadiness, ExpContext
from zcu_tools.gui.app.main.controller import Controller
from zcu_tools.gui.app.main.registry import Registry
from zcu_tools.gui.app.main.services.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.main.services.remote.dialogs import DialogName
from zcu_tools.gui.app.main.state import State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import ExpectedError
from zcu_tools.gui.remote.errors import remote_error_from_expected
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.services.io_manager import IOManager


def make_ctx() -> ExpContext:
    return ExpContext(
        md=MagicMock(),
        ml=MagicMock(),
        soc=MagicMock(),
        soccfg=MagicMock(),
        res_name="fake_res",
        result_dir="/tmp/zcu_result",
        database_path="/tmp/zcu_db/fake_chip/fake_qubit",
        active_label="ctx001",
        readiness=ContextReadiness.ACTIVE,
    )


def make_view() -> MagicMock:
    view = MagicMock()
    view.show_status_message = MagicMock()
    view.show_error_dialog = MagicMock()
    view.make_live_container = MagicMock(return_value=None)
    # shaped View surface so Controller.open_dialog / take_figure_screenshot
    # / get_view_snapshot have somewhere to land in tests.
    view._open_dialogs = []

    def _open_dialog(name: DialogName) -> None:
        if name not in view._open_dialogs:
            view._open_dialogs.append(name)

    def _close_dialog(name: DialogName) -> None:
        if name in view._open_dialogs:
            view._open_dialogs.remove(name)

    view.open_dialog = MagicMock(side_effect=_open_dialog)
    view.close_dialog = MagicMock(side_effect=_close_dialog)
    view.list_open_dialogs = MagicMock(side_effect=lambda: list(view._open_dialogs))
    view.register_dialog = MagicMock()
    view.get_view_snapshot = MagicMock(
        return_value={
            "active_tab_id": None,
            "tab_ids": [],
            "context_label": "ctx001",
            "predictor_label": "none",
            "status": "Ready",
            "open_dialogs": [],
        }
    )
    # Static PNG bytes (1x1 transparent PNG).
    _PNG = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00"
        b"\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cb"
        b"\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    view.take_figure_screenshot = MagicMock(return_value=_PNG)
    view.take_dialog_screenshot = MagicMock(return_value=_PNG)
    view.take_window_screenshot = MagicMock(return_value=_PNG)
    return view


class Fixture:
    """Holds strong refs to Controller + service to survive GC mid-test."""

    def __init__(
        self, opts: ControlOptions | None = None, project_root: str | None = None
    ) -> None:
        self.state = State(make_ctx())
        self.registry = Registry()
        register_all(self.registry)
        if not self.registry.has("fake"):
            self.registry.register("fake", FakeAdapter)
        self.view = make_view()
        io_manager = IOManager()
        io_manager._em = MagicMock()
        self.bus = EventBus()
        self.ctrl = Controller(
            state=self.state,
            registry=self.registry,
            io_manager=io_manager,
            view=self.view,
            bus=self.bus,
            project_root=project_root,
        )
        if opts is None:
            opts = ControlOptions(port=0)
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


class FakeTransport:
    """Synchronous in-memory Transport for McpBridge tests (no socket/thread).

    Implements the ``zcu_tools.mcp.core.bridge.Transport`` protocol. On
    ``send_line`` it records the outgoing ``(method, params)`` in ``sent`` and
    immediately delivers a reply (from ``replies[method]``, defaulting to
    ``{ok: True, result: {}}``) via the bridge's ``deliver_reply`` callback — so
    a synchronous ``send_rpc_raw`` round-trip completes without a real GUI. Inject
    via ``bridge.set_transport(FakeTransport())``; populate ``replies`` per test.
    """

    def __init__(self) -> None:
        self.replies: dict[str, dict] = {}
        self.sent: list[tuple[str, dict]] = []
        self._deliver_reply: Callable[[dict], None] | None = None

    def attach(self, deliver_reply, deliver_event, on_closed) -> None:
        # Reply-only fake: the event / on_closed callbacks are unused.
        del deliver_event, on_closed
        self._deliver_reply = deliver_reply

    @property
    def is_open(self) -> bool:
        return True

    def send_line(self, payload: dict) -> None:
        self.sent.append((payload["method"], payload["params"]))
        resp = dict(self.replies.get(payload["method"], {"ok": True, "result": {}}))
        resp["id"] = payload["id"]
        assert self._deliver_reply is not None, "FakeTransport used before attach()"
        self._deliver_reply(resp)

    def close(self) -> None:
        pass


def dispatch_handler(ctrl: Any, method: str, params: dict) -> Mapping[str, object]:
    """Invoke a handler plus the shared expected-error projection boundary.

    Handlers now receive the ``RemoteControlAdapter`` (not the bare ctrl) and
    reach the façade via ``adapter.ctrl`` (ADR-0013). This wraps ``ctrl`` in a
    minimal adapter stub so unit tests can drive a single handler without a live
    socket. The lightweight adapter delegates nominal ``ExpectedError`` mapping
    to the production translator; unexpected exceptions escape unchanged.
    Cast keeps the typed ``Handler`` signature satisfied.
    """
    from types import SimpleNamespace
    from typing import cast

    from zcu_tools.gui.app.main.services.remote.dispatch import METHOD_REGISTRY

    def _facet_or_self(name: str) -> Any:
        if isinstance(ctrl, MagicMock) and name not in ctrl.__dict__:
            return ctrl
        return getattr(ctrl, name, ctrl)

    adapter = cast(
        RemoteControlAdapter,
        SimpleNamespace(
            ctrl=ctrl,
            run_analyze_control=_facet_or_self("run_analyze_control"),
            operation_control=_facet_or_self("operation_control"),
            save_control=_facet_or_self("save_control"),
            writeback_control=_facet_or_self("writeback_control"),
            context_control=_facet_or_self("context_control"),
            device_control=_facet_or_self("device_control"),
            predictor_control=_facet_or_self("predictor_control"),
        ),
    )
    try:
        return METHOD_REGISTRY[method].handler(adapter, params)
    except ExpectedError as exc:
        raise remote_error_from_expected(exc) from exc


def send(sock: socket.socket, obj: dict) -> None:
    sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))


# Per-socket inbox: lines arriving while a caller was waiting for a different
# match get parked here so a subsequent ``recv_*`` call can still observe
# them. Keyed by socket fileno (sockets are not hashable across all qtpy
# backends but their fileno is stable).
_INBOX: dict[int, list[dict]] = {}
_INBOX_BUF: dict[int, bytearray] = {}


def _inbox(sock: socket.socket) -> list[dict]:
    return _INBOX.setdefault(sock.fileno(), [])


def _inbox_buf(sock: socket.socket) -> bytearray:
    return _INBOX_BUF.setdefault(sock.fileno(), bytearray())


def recv_until(
    sock: socket.socket,
    accept: Callable[[dict], bool],
    timeout_s: float = 3.0,
) -> dict:
    """Wait for the first NDJSON line where ``accept(msg)`` is True.

    Lines that ``accept`` rejects are **parked** in a per-socket inbox so a
    later ``recv_*`` call can still observe them — this is important when
    a test waits for a reply that may arrive interleaved with event pushes,
    or vice versa. Pumps the Qt event loop between recv attempts so
    marshalled handlers and EventBus emits make progress.
    """
    app = QCoreApplication.instance()
    assert app is not None
    inbox = _inbox(sock)
    # Scan the parked queue first before touching the socket.
    for idx, msg in enumerate(inbox):
        if accept(msg):
            return inbox.pop(idx)
    deadline = time.monotonic() + timeout_s
    buf = _inbox_buf(sock)
    sock.setblocking(False)
    while time.monotonic() < deadline:
        try:
            chunk = sock.recv(4096)
            if not chunk:
                raise AssertionError("peer closed without a matching line")
            buf.extend(chunk)
        except BlockingIOError:
            pass
        while True:
            nl = buf.find(b"\n")
            if nl < 0:
                break
            line = bytes(buf[:nl])
            del buf[: nl + 1]
            if not line:
                continue
            msg = json.loads(line.decode("utf-8"))
            if accept(msg):
                return msg
            inbox.append(msg)
        app.processEvents()
        time.sleep(0.005)
    raise AssertionError(f"no matching message within {timeout_s}s")


def reset_inbox(sock: socket.socket) -> None:
    """Clear any parked messages for ``sock`` (call between unrelated tests)."""
    _INBOX.pop(sock.fileno(), None)
    _INBOX_BUF.pop(sock.fileno(), None)


def recv_response(sock: socket.socket, rid: str, timeout_s: float = 3.0) -> dict:
    """Wait for a NDJSON response with ``id == rid``; drops pushes."""
    return recv_until(
        sock,
        lambda msg: isinstance(msg, dict) and msg.get("id") == rid,
        timeout_s,
    )


def recv_push(sock: socket.socket, event: str, timeout_s: float = 3.0) -> dict:
    """Wait for a push line whose ``event`` matches; drops replies."""
    return recv_until(
        sock,
        lambda msg: isinstance(msg, dict) and msg.get("event") == event,
        timeout_s,
    )


def open_client(port: int) -> socket.socket:
    return socket.create_connection(("127.0.0.1", port), timeout=1.0)


def call(
    sock: socket.socket,
    method: str,
    params: dict | None = None,
    *,
    rid: str = "1",
    timeout_s: float = 3.0,
) -> dict:
    """Send a single RPC and wait for its matching reply."""
    send(sock, {"id": rid, "method": method, "params": params or {}})
    return recv_response(sock, rid, timeout_s)


__all__ = [
    "Fixture",
    "call",
    "make_ctx",
    "make_view",
    "open_client",
    "recv_push",
    "recv_response",
    "recv_until",
    "send",
    "Any",  # re-export for type-loose tests
]
