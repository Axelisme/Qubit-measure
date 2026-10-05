"""Shared test helpers for the RemoteControlAdapter remote test suites.

Single source of fixture + socket-pumping helpers so the remote test files do
not duplicate the boilerplate. Anything suite-specific (e.g. event-push
reception) lives next to the tests that need it.
"""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from PIL import Image
from qtpy.QtCore import QCoreApplication
from zcu_tools.experiment.v2_gui.measure.adapters.fake import FakeAdapter
from zcu_tools.experiment.v2_gui.measure.registry import register_all
from zcu_tools.gui.app.measure.adapter import (
    ContextReadiness,
    ExpAdapterProtocol,
    SessionEnv,
)
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.remote.dialogs import DialogName
from zcu_tools.gui.app.measure.remote.wire_version import WIRE_VERSION
from zcu_tools.gui.app.measure.role_catalog import RoleCatalog
from zcu_tools.gui.app.measure.services.tab_cfg import TabCfgResources
from zcu_tools.gui.app.measure.state import Session, State
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.cfg.resource import CfgObservation, CfgResource
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import ExpectedError
from zcu_tools.gui.remote.errors import remote_error_from_expected
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.services.io_manager import IOManager
from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.mcp.measure.assembly import build_measure_tools
from zcu_tools.mcp.measure.recipe import RecipeDefinition
from zcu_tools.mcp.measure.session import MeasureMcpSession
from zcu_tools.mcp.measure.tool_context import MeasureToolContext
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.resources.context import MetaDict, ModuleLibrary


def make_ctx() -> SessionEnv:
    return SessionEnv(
        md=MetaDict(None),
        ml=ModuleLibrary(None),
        soc=MagicMock(),
        soccfg=make_mock_soccfg(),
        res_name="fake_res",
        result_dir="/tmp/zcu_result",
        database_path="/tmp/zcu_db/fake_chip/fake_qubit",
        active_label="ctx001",
        readiness=ContextReadiness.ACTIVE,
    )


def observe_run_inputs(
    fx, tab_id: str, invoke: Callable[[str, dict[str, Any]], Any]
) -> None:
    """Explicitly read each run dependency; tab cfg exists without a widget."""
    for method, params in (
        ("tab.snapshot", {"tab_id": tab_id}),
        ("tab.get_cfg", {"tab_id": tab_id}),
        ("soc.info", {"include_cfg": True}),
    ):
        invoke(method, params)
    devices = invoke("device.list", {})["devices"]
    for device in devices:
        invoke("device.snapshot", {"name": device["name"]})


def make_png() -> bytes:
    """A complete decodable PNG for an in-process render collaborator."""
    buffer = BytesIO()
    Image.new("RGBA", (1, 1)).save(buffer, format="PNG")
    return buffer.getvalue()


def make_view() -> MagicMock:
    view = MagicMock()
    view.show_status_message = MagicMock()
    view.show_error_dialog = MagicMock()
    view.make_run_container = MagicMock(return_value=None)
    # shaped View surface so Controller.open_dialog / take_figure_screenshot_for_subtab
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
    png = make_png()
    view.take_figure_screenshot_for_subtab = MagicMock(return_value=png)
    view.take_dialog_screenshot = MagicMock(return_value=png)
    view.take_window_screenshot = MagicMock(return_value=png)
    return view


class Fixture:
    """Holds strong refs to Controller + service to survive GC mid-test."""

    def __init__(
        self,
        opts: ControlOptions | None = None,
        project_root: str | None = None,
        *,
        active_label: str | None = None,
        role_catalog: RoleCatalog | None = None,
        empty_project: bool = False,
        headless: bool = False,
    ) -> None:
        initial = (
            SessionEnv(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)
            if empty_project
            else make_ctx()
        )
        self.state = State(initial)
        self.registry = Registry()
        register_all(self.registry)
        if not self.registry.has("fake"):
            self.registry.register("fake", FakeAdapter)
        self.view = None if headless else make_view()
        io_manager = IOManager()
        if not empty_project:
            exp_manager = MagicMock()
            if active_label is not None:
                exp_manager.current_label = active_label
            io_manager._em = exp_manager
        self.bus = EventBus()
        self.ctrl = Controller(
            state=self.state,
            registry=self.registry,
            io_manager=io_manager,
            view=self.view,
            bus=self.bus,
            role_catalog=role_catalog,
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

    def prepare_tab(
        self,
        tab_id: str,
        adapter: ExpAdapterProtocol,
        schema: CfgSchema,
        *,
        adapter_name: str = "fake",
    ) -> CfgResource:
        """Install a custom test tab through the application lifetime owner."""
        resources = cast(TabCfgResources, self.ctrl.cfg_resources)
        cfg = resources.create(tab_id, lambda: schema, initial=schema)
        self.state.add_tab(
            tab_id, Session(adapter_name=adapter_name, adapter=adapter, cfg=cfg)
        )

        def published(_observation: CfgObservation) -> None:
            self.state.version.bump(f"tab:{tab_id}:cfg")

        cfg.watch(published)
        return cfg

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
    reach the façade via ``adapter.ctrl`` (ADR-0068). This wraps ``ctrl`` in a
    minimal adapter stub so unit tests can drive a single handler without a live
    socket. The lightweight adapter delegates nominal ``ExpectedError`` mapping
    to the production translator; unexpected exceptions escape unchanged.
    Cast keeps the typed ``Handler`` signature satisfied.
    """
    from types import SimpleNamespace
    from typing import cast

    from zcu_tools.gui.app.measure.remote.dispatch import METHOD_REGISTRY

    def _facet_or_self(name: str) -> Any:
        if isinstance(ctrl, MagicMock) and name not in ctrl.__dict__:
            return ctrl
        return getattr(ctrl, name, ctrl)

    adapter = cast(
        RemoteControlAdapter,
        SimpleNamespace(
            ctrl=ctrl,
            tab_control=_facet_or_self("tab_control"),
            run_analyze_control=_facet_or_self("run_analyze_control"),
            operation_control=_facet_or_self("operation_control"),
            save_control=_facet_or_self("save_control"),
            writeback_control=_facet_or_self("writeback_control"),
            context_control=_facet_or_self("context_control"),
            device_control=_facet_or_self("device_control"),
            predictor_control=_facet_or_self("predictor_control"),
            cfg_lookup=lambda tab_id: _facet_or_self("cfg_resources").lookup(tab_id),
            render_view=_facet_or_self("render_view"),
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
    sock = socket.create_connection(("127.0.0.1", port), timeout=1.0)
    # The OS may recycle a closed socket's fileno from an earlier test.
    reset_inbox(sock)
    return sock


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


def call_mcp_with_qt(
    tools: ToolTable, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    # A Future propagates the worker's exception when the Qt-pumping owner reads it.
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        worker = pool.submit(tools[name]["handler"], arguments)
        deadline = time.monotonic() + 5
        while not worker.done() and time.monotonic() < deadline:
            QCoreApplication.processEvents()
            time.sleep(0.005)
        assert worker.done(), "MCP tool did not receive a GUI reply"
        reply = worker.result()
        # These GUI contracts inspect structured data; stdio image delivery has its own owner.
        return reply.data if isinstance(reply, ToolReply) else reply
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def mcp_client(
    port: int,
    tmp_path: Path,
    *,
    request: pytest.FixtureRequest | None = None,
    recipes: Sequence[RecipeDefinition] = (),
) -> tuple[McpBridge, Callable[[str, dict[str, Any]], dict[str, Any]]]:
    """Build a fixed-session MCP caller for the test-owned loopback GUI.

    port is the fixture service port. tmp_path owns unused launch paths; no GUI
    process is launched. request registers session.close for teardown when given.
    recipes is the exact injected tool set; empty exposes only generic tools.
    Return the bridge and a Qt-pumping caller that propagates tool failures.
    """
    config = MCPBridgeConfig(
        tool_prefix="",
        server_display_name="measure-test",
        server_instructions="",
        app_name="gui",
        default_port=port,
        mcp_version=82,
        wire_version=WIRE_VERSION,
        pid_file=tmp_path / "unused.pid",
        log_file=tmp_path / "unused.log",
        run_script_name="run_measure_gui.py",
    )

    def resolver(config: MCPBridgeConfig, requested: int | None) -> int:
        return config.default_port if requested is None else requested

    session = MeasureMcpSession(
        config,
        recipes=recipes,
        resolve_connect_port=resolver,
        port_is_open=lambda _: True,
    )
    bridge = McpBridge(config)
    session.attach_bridge(bridge)
    if request is not None:
        request.addfinalizer(session.close)
    tools = build_measure_tools(
        MeasureToolContext(config, session, resolve_connect_port=resolver),
        recipes=recipes,
    )
    return bridge, partial(call_mcp_with_qt, tools)


__all__ = [
    "Fixture",
    "call",
    "call_mcp_with_qt",
    "mcp_client",
    "make_ctx",
    "make_view",
    "open_client",
    "recv_push",
    "recv_response",
    "recv_until",
    "send",
    "Any",  # re-export for type-loose tests
]
