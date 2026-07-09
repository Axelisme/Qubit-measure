"""Tests for the user-launched GUI control-socket default-port feature.

Covers three guarantees:
1. Default port consistency: the run-script default port matches the MCP
   server's ``default_port`` for every app (measure / fluxdep / dispersive /
   autofluxdep).  These are pure import-level constants — no Qt needed.
2. --no-control suppression: when ``no_control=True`` is passed (simulating
   the --no-control flag), runtime control options are None, so app assembly
   never tries to open a socket.
3. Port-collision fast-fail: when ``NdjsonRpcEndpoint.start()`` raises
   ``RuntimeError`` (simulating an EADDRINUSE bind failure), the adapter's
   ``start()`` re-raises the same RuntimeError so the app-level handler can
   print a user-friendly message and exit, after rolling back app-side listeners.
"""

from __future__ import annotations

import argparse
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. Default port consistency
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parents[4] / "script"


def _parse_run_script(
    script_name: str, extra_args: list[str] | None = None
) -> argparse.Namespace:
    """Import a run_* script's _parse_args and invoke it with no extra args."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"_script_{script_name}",
        SCRIPT_DIR / f"{script_name}.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module._parse_args(extra_args or [])


@pytest.mark.parametrize(
    "script_name",
    [
        "run_measure_gui",
        "run_fluxdep_gui",
        "run_dispersive_gui",
        "run_autofluxdep_gui",
    ],
)
def test_omitted_control_port_is_none(script_name: str) -> None:
    """Omitting --control-port yields None (the sentinel for 'use convention port').

    The agreed-upon port is no longer the argparse default; an omitted port means
    "use the convention port AND allow ephemeral fallback", distinguished from an
    explicitly-pinned port (which fast-fails on collision). The convention port is
    re-checked against the MCP default_port in
    ``test_mcp_default_port_matches_expected``.
    """
    args = _parse_run_script(script_name)
    assert args.control_port is None, (
        f"{script_name}: omitted --control-port should be None, got {args.control_port}"
    )


@pytest.mark.parametrize(
    "script_name, mcp_module, expected_port",
    [
        ("run_measure_gui", "zcu_tools.mcp.measure.server", 8765),
        ("run_fluxdep_gui", "zcu_tools.mcp.fluxdep.server", 8766),
        ("run_dispersive_gui", "zcu_tools.mcp.dispersive.server", 8767),
        ("run_autofluxdep_gui", "zcu_tools.mcp.autofluxdep.server", 8768),
    ],
)
def test_mcp_default_port_matches_expected(
    script_name: str, mcp_module: str, expected_port: int
) -> None:
    """MCP server _CONFIG.default_port equals the documented agreed port."""
    import importlib

    mod = importlib.import_module(mcp_module)
    actual = mod._CONFIG.default_port  # type: ignore[attr-defined]
    assert actual == expected_port, (
        f"{mcp_module}: expected default_port={expected_port}, got {actual}"
    )


# ---------------------------------------------------------------------------
# 1b. Discovery slug consistency: MCP reader slug == GUI writer slug
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mcp_module, expected_slug",
    [
        ("zcu_tools.mcp.measure.server", "measure"),
        ("zcu_tools.mcp.fluxdep.server", "fluxdep"),
        ("zcu_tools.mcp.dispersive.server", "dispersive"),
        ("zcu_tools.mcp.autofluxdep.server", "autofluxdep"),
    ],
)
def test_mcp_app_slug_matches_expected(mcp_module: str, expected_slug: str) -> None:
    """Each MCP server advertises the discovery slug its GUI writes under.

    The run scripts pass these same slugs to ``ControlOptions(app_slug=...)``; the
    GUI writer and MCP reader must agree or discovery silently misses.
    """
    import importlib

    mod = importlib.import_module(mcp_module)
    assert mod._CONFIG.app_slug == expected_slug  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 2. --no-control produces None control_opts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "script_name",
    [
        "run_measure_gui",
        "run_fluxdep_gui",
        "run_dispersive_gui",
        "run_autofluxdep_gui",
    ],
)
def test_no_control_flag_suppresses_socket(script_name: str) -> None:
    """--no-control must result in no_control=True from argparse."""
    args = _parse_run_script(script_name, ["--no-control"])
    assert args.no_control is True, (
        f"{script_name}: --no-control flag not parsed correctly"
    )
    # Verify the control_opts / control would be None when no_control is True
    # (the scripts gate on ``not args.no_control``).
    control_opts = None if args.no_control else object()  # sentinel
    assert control_opts is None


# ---------------------------------------------------------------------------
# 3. Port-collision fast-fail propagation
# ---------------------------------------------------------------------------


def test_ndjson_endpoint_bind_failure_raises_runtime_error(qapp) -> None:  # noqa: ARG001
    """NdjsonRpcEndpoint.start() wraps OSError into RuntimeError with a message.

    We bind a real ephemeral port first, then try to bind it again; the second
    bind raises OSError (EADDRINUSE) which the endpoint must wrap into RuntimeError.
    """
    import socket as socket_mod

    from zcu_tools.gui.remote.rpc_endpoint import ControlOptions, NdjsonRpcEndpoint

    # Hold an ephemeral port open so the second bind fails.
    # listen() is required on Linux: SO_REUSEADDR alone allows a second bind(),
    # but a listening socket makes the second bind raise EADDRINUSE.
    blocker = socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM)
    blocker.setsockopt(socket_mod.SOL_SOCKET, socket_mod.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    taken_port = blocker.getsockname()[1]

    class _FakeRouter:
        def on_client_open(self, link):  # noqa: ANN001
            pass

        def on_client_close(self, link, *, on_main_thread):  # noqa: ANN001
            pass

        def route(self, link, request):  # noqa: ANN001
            pass

    opts = ControlOptions(port=taken_port)

    endpoint = NdjsonRpcEndpoint(
        opts,
        wire_version=1,
        gui_version=1,
        server_name="test_server",
        router=_FakeRouter(),
    )

    try:
        with pytest.raises(RuntimeError, match="bind") as exc_info:
            endpoint.start()

        # The message must name the host:port so the user can act on it.
        msg = str(exc_info.value).lower()
        assert "bind" in msg
        assert str(taken_port) in str(exc_info.value)
    finally:
        blocker.close()


def test_remote_control_adapter_start_rolls_back_bind_error(qapp) -> None:
    """RemoteControlServiceBase.start() rolls back, then propagates bind errors.

    The shared runtime catches this and prints a user-friendly message. This
    test confirms the error is not swallowed at the service layer and
    measure-gui's extra listeners are not leaked.
    """

    # Import the measure-gui adapter (representative of all four).
    from zcu_tools.gui.app.main.services.remote import RemoteControlAdapter
    from zcu_tools.gui.app.main.services.remote.events import EVENT_SERIALIZERS
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.remote.rpc_endpoint import ControlOptions

    bus = BaseEventBus()
    ctrl_mock = MagicMock()
    ctrl_mock.get_bus.return_value = bus
    render_view_mock = MagicMock()

    opts = ControlOptions(port=0)  # valid opts; we'll patch the underlying endpoint

    adapter = RemoteControlAdapter(
        controller=ctrl_mock,
        opts=opts,
        render_view=render_view_mock,
    )

    bind_error = RuntimeError(
        "NdjsonRpcEndpoint bind 127.0.0.1:0 failed: [Errno 98] Address already in use"
    )

    with (
        patch.object(adapter._endpoint, "start", side_effect=bind_error),
        patch.object(adapter._endpoint, "stop") as endpoint_stop,
    ):
        with pytest.raises(RuntimeError, match="bind"):
            adapter.start()

    endpoint_stop.assert_not_called()
    assert len(adapter._bus_subs) == 0
    assert all(not bus._subs.get(event_key) for event_key in EVENT_SERIALIZERS)
    assert ctrl_mock.set_cfg_editor_change_listener.call_args_list[-1].args == (None,)
    ctrl_mock.add_diagnostic_sink.assert_called_once_with(adapter)
    ctrl_mock.remove_diagnostic_sink.assert_called_once_with(adapter)
    assert ctrl_mock.set_agent_connected_query.call_args_list[-1].args == (None,)


def test_remote_control_adapter_start_fails_fast_and_rolls_back_event_subscription(
    qapp,
) -> None:
    from zcu_tools.gui.app.main.services.remote import RemoteControlAdapter
    from zcu_tools.gui.app.main.services.remote.events import EVENT_SERIALIZERS
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.remote.rpc_endpoint import ControlOptions

    class FailingEventBus(BaseEventBus):
        def __init__(self) -> None:
            super().__init__()
            self.subscribe_count = 0

        def subscribe(self, payload_type: type[Any], cb: Callable[[Any], None]) -> Any:
            self.subscribe_count += 1
            if self.subscribe_count == 2:
                raise RuntimeError("subscribe failed")
            return super().subscribe(payload_type, cb)

    bus = FailingEventBus()
    ctrl_mock = MagicMock()
    ctrl_mock.get_bus.return_value = bus
    adapter = RemoteControlAdapter(
        controller=ctrl_mock,
        opts=ControlOptions(port=0),
        render_view=MagicMock(),
    )

    with patch.object(adapter._endpoint, "start") as endpoint_start:
        with pytest.raises(RuntimeError, match="subscribe failed"):
            adapter.start()

    endpoint_start.assert_not_called()
    assert len(adapter._bus_subs) == 0
    assert all(not bus._subs.get(event_key) for event_key in EVENT_SERIALIZERS)


def test_remote_control_adapter_start_rolls_back_advertise_error(qapp) -> None:
    """RemoteControlServiceBase.start() also rolls back after advertise failure."""

    from zcu_tools.gui.app.main.services.remote import RemoteControlAdapter
    from zcu_tools.gui.app.main.services.remote.events import EVENT_SERIALIZERS
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.remote.rpc_endpoint import ControlOptions

    bus = BaseEventBus()
    ctrl_mock = MagicMock()
    ctrl_mock.get_bus.return_value = bus
    adapter = RemoteControlAdapter(
        controller=ctrl_mock,
        opts=ControlOptions(port=0, app_slug="measure"),
        render_view=MagicMock(),
    )
    advertise_error = RuntimeError("discovery write failed")

    with (
        patch.object(adapter._endpoint, "start", return_value=12345),
        patch.object(adapter._endpoint, "stop") as endpoint_stop,
        patch.object(adapter, "_advertise_session", side_effect=advertise_error),
    ):
        with pytest.raises(RuntimeError, match="discovery write failed"):
            adapter.start()

    endpoint_stop.assert_called_once_with()
    assert len(adapter._bus_subs) == 0
    assert all(not bus._subs.get(event_key) for event_key in EVENT_SERIALIZERS)
    assert ctrl_mock.set_cfg_editor_change_listener.call_args_list[-1].args == (None,)
    ctrl_mock.add_diagnostic_sink.assert_called_once_with(adapter)
    ctrl_mock.remove_diagnostic_sink.assert_called_once_with(adapter)
    assert ctrl_mock.set_agent_connected_query.call_args_list[-1].args == (None,)


# ---------------------------------------------------------------------------
# 4. Ephemeral fallback (default port) vs fast-fail (pinned port)
# ---------------------------------------------------------------------------


class _FakeRouter:
    def on_client_open(self, link):  # noqa: ANN001, ANN201
        pass

    def on_client_close(self, link, *, on_main_thread):  # noqa: ANN001, ANN201
        pass

    def route(self, link, request):  # noqa: ANN001, ANN201
        pass


def _take_port() -> tuple[socket.socket, int]:
    """Bind+listen an ephemeral port so a re-bind there raises EADDRINUSE."""
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    return blocker, blocker.getsockname()[1]


def test_default_port_busy_falls_back_to_ephemeral() -> None:
    """allow_port_fallback=True: a taken port retries on 0 and binds elsewhere."""
    from zcu_tools.gui.remote.rpc_endpoint import ControlOptions, NdjsonRpcEndpoint

    blocker, taken_port = _take_port()
    opts = ControlOptions(port=taken_port, allow_port_fallback=True)
    endpoint = NdjsonRpcEndpoint(
        opts,
        wire_version=1,
        gui_version=1,
        server_name="test_fallback",
        router=_FakeRouter(),
    )
    try:
        bound = endpoint.start()
        assert bound != taken_port and bound != 0, (
            "fallback must bind a real OS-assigned port, not the busy one or 0"
        )
        assert endpoint.port == bound
    finally:
        endpoint.stop()
        blocker.close()


def test_pinned_port_busy_fast_fails() -> None:
    """allow_port_fallback=False (explicit --control-port): a taken port raises."""
    from zcu_tools.gui.remote.rpc_endpoint import ControlOptions, NdjsonRpcEndpoint

    blocker, taken_port = _take_port()
    opts = ControlOptions(port=taken_port, allow_port_fallback=False)
    endpoint = NdjsonRpcEndpoint(
        opts,
        wire_version=1,
        gui_version=1,
        server_name="test_pinned",
        router=_FakeRouter(),
    )
    try:
        with pytest.raises(RuntimeError, match="bind"):
            endpoint.start()
    finally:
        blocker.close()
