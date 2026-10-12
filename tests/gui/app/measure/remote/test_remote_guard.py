"""RemoteControlAdapter transport / dispatch / query tests.

Each test spins up a real TCP socket on an ephemeral loopback port and drives
a fixture Controller (already wired up with a fake adapter). The Qt event loop
is not freely running under pytest, so a helper interleaves
``QApplication.processEvents()`` between socket reads to give marshalled
handlers a chance to execute.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
    CfgPreconditionError,
    CfgPreconditionReason,
)
from zcu_tools.gui.remote.framing import MAX_LINE_BYTES
from zcu_tools.gui.session.context_types import EntryRequest
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.resources.entry.registry import component_registry

from tests.gui.app.measure._components import QubitSchema, ResonatorSchema
from tests.gui.app.measure._load_support import historical_metadata
from tests.gui.app.measure._synthetic import SampleAdapter

from ._helpers import call as _raw_call
from ._helpers import make_png, observe_run_inputs
from ._helpers import mcp_client as _mcp_client
from ._remote_core_support import (
    RemoteCoreFixture as _Fixture,
)
from ._remote_core_support import open_client as _open_client
from ._remote_core_support import recv_response as _recv_response
from ._remote_core_support import send as _send


@pytest.fixture()
def fx(qapp):
    before = component_registry.kinds()
    added = []
    for kind, model in (
        ("qubit/transmon", QubitSchema),
        ("resonator", ResonatorSchema),
    ):
        if kind not in before:
            component_registry.register(kind, model)
            added.append(kind)
    f = _Fixture()
    f.start()
    try:
        yield f
    finally:
        f.stop()
        for kind in reversed(added):
            component_registry.unregister(kind)
        assert component_registry.kinds() == before


pytestmark = pytest.mark.uses_wall_clock


def _prepare_guarded_context(fx: _Fixture) -> None:
    fx.state.set_context(replace(fx.state.session_env, soccfg=make_mock_soccfg()))
    fx.state.version.bump("soc")


def _await_completed_run(
    call: Callable[[str, dict[str, Any]], dict[str, Any]], handle: int
) -> None:
    assert call("wait", {"op": handle, "timeout": 3})["status"] == "finished"


def _edit_context_as_gui(fx: _Fixture, key: str, value: str | int) -> None:
    parameters = fx.ctrl.read_context()
    assert parameters is not None
    fx.ctrl.context_control.set_parameter(
        parameters.target, f"general.ext.{key}", value
    )


@pytest.mark.parametrize("with_device", [False, True])
def test_mcp_created_tab_can_start_a_guarded_run_on_real_gui_state(
    fx, tmp_path: Path, with_device: bool
) -> None:
    _prepare_guarded_context(fx)
    port = fx.service.port
    bridge, call = _mcp_client(port, tmp_path)
    try:
        assert call("connect", {"port": port})["port"] == port
        if with_device:
            connected = call(
                "rpc_call",
                {
                    "method": "device.connect",
                    "params": {
                        "name": "bias",
                        "type_name": "FakeDevice",
                        "address": "none",
                    },
                },
            )
            assert (
                call("wait", {"op": connected["handle"], "timeout": 5})["status"]
                == "finished"
            )
            # A fresh connection must observe the already registered device itself.
            bridge.disconnect()
            call("connect", {"port": port})
        tab_id = call(
            "rpc_call", {"method": "tab.new", "params": {"adapter_name": "fake"}}
        )["tab_id"]
        assert (
            call("rpc_call", {"method": "context.snapshot"})["point"]["general"]["ext"]
            == {}
        )
        assert "cfg" in call(
            "rpc_call", {"method": "soc.info", "params": {"include_cfg": True}}
        )
        observe_run_inputs(
            fx,
            tab_id,
            lambda method, params: call(
                "rpc_call", {"method": method, "params": params}
            ),
        )
        started = call(
            "rpc_call",
            {
                "method": "tab.run_start",
                "params": {
                    "tab_id": tab_id,
                },
            },
        )
        assert started["handle"] > 0
        _await_completed_run(call, started["handle"])
    finally:
        try:
            if with_device:
                disconnected = call(
                    "rpc_call",
                    {
                        "method": "device.disconnect",
                        "params": {"name": "bias", "remember": False},
                    },
                )
                assert (
                    call("wait", {"op": disconnected["handle"], "timeout": 5})["status"]
                    == "finished"
                )
        finally:
            bridge.disconnect()


def test_attached_gui_tab_runs_after_explicit_full_reads(fx, tmp_path: Path) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")  # Created by GUI before MCP attaches.
    bridge, call = _mcp_client(fx.service.port, tmp_path)
    try:
        call("connect", {"port": fx.service.port})
        assert (
            call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}})[
                "tabs"
            ][0]["tab_id"]
            == tab_id
        )
        call("rpc_call", {"method": "context.snapshot"})
        call("rpc_call", {"method": "soc.info", "params": {"include_cfg": True}})
        observe_run_inputs(
            fx,
            tab_id,
            lambda method, params: call(
                "rpc_call", {"method": method, "params": params}
            ),
        )
        handle = call(
            "rpc_call",
            {
                "method": "tab.run_start",
                "params": {
                    "tab_id": tab_id,
                },
            },
        )["handle"]
        assert handle > 0
        _await_completed_run(call, handle)
    finally:
        bridge.disconnect()


def test_mcp_run_analyze_writeback_save_close_on_one_connection(
    fx, tmp_path: Path
) -> None:
    _prepare_guarded_context(fx)
    # The fixture View supplies a valid PNG; artifact export below is real.
    fx.view.take_figure_screenshot_for_subtab.return_value = make_png()
    bridge, call = _mcp_client(fx.service.port, tmp_path)
    try:
        call("connect", {"port": fx.service.port})
        tab_id = call(
            "rpc_call", {"method": "tab.new", "params": {"adapter_name": "fake"}}
        )["tab_id"]
        call("rpc_call", {"method": "context.snapshot"})
        call("rpc_call", {"method": "soc.info", "params": {"include_cfg": True}})
        observe_run_inputs(
            fx,
            tab_id,
            lambda method, params: call(
                "rpc_call", {"method": method, "params": params}
            ),
        )
        started = call(
            "rpc_call",
            {
                "method": "tab.run_start",
                "params": {
                    "tab_id": tab_id,
                },
            },
        )
        _await_completed_run(call, started["handle"])
        assert call(
            "rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}}
        )["tabs"][0]["interaction"]["has_run_result"]
        analyzed = call("tab_analyze", {"tab": tab_id})
        if "op" in analyzed:
            assert (
                call("wait", {"op": analyzed["op"], "timeout": 5})["status"]
                == "finished"
            )
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}})[
            "tabs"
        ][0]
        preview = call(
            "rpc_call",
            {
                "method": "tab.writeback_preview",
                "params": {"tab_id": tab_id, "subtab_id": "analysis"},
            },
        )
        assert len(preview["items"]) == 1
        item = preview["items"][0]
        assert item["target_name"] == "general.ext.fake_peak"
        written = call(
            "rpc_call",
            {
                "method": "tab.writeback_write",
                "params": {
                    "write": [{"id": item["id"]}],
                    "tab_id": tab_id,
                    "subtab_id": "analysis",
                },
            },
        )
        assert written["written"][0]["after"] == item["proposed"]
        assert (
            call("rpc_call", {"method": "context.snapshot"})["point"]["general"]["ext"][
                "fake_peak"
            ]
            == item["proposed"]
        )
        assert call("tab_close", {"tab": tab_id, "discard_unsaved": True}) == {
            "closed": tab_id
        }
        assert not fx.ctrl.run_analyze_control.has_tab(tab_id)
    finally:
        bridge.disconnect()


def test_tab_run_uses_the_attached_gui_draft_and_returns_a_waitable_handle(
    fx, tmp_path: Path
) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    bridge, call = _mcp_client(fx.service.port, tmp_path)
    try:
        call("connect", {"port": fx.service.port})
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}})
        call("rpc_call", {"method": "context.snapshot"})
        call("rpc_call", {"method": "soc.info", "params": {"include_cfg": True}})

        observe_run_inputs(
            fx,
            tab_id,
            lambda method, params: call(
                "rpc_call", {"method": method, "params": params}
            ),
        )
        started = call(
            "rpc_call",
            {
                "method": "tab.run_start",
                "params": {
                    "tab_id": tab_id,
                },
            },
        )
        assert set(started) == {"handle"}
        assert isinstance(started["handle"], int) and started["handle"] > 0
        _await_completed_run(call, started["handle"])
        assert fx.state.get_tab(tab_id).run.result is not None
    finally:
        bridge.disconnect()


def test_tab_run_rejects_missing_active_context_without_starting(
    fx, tmp_path: Path
) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    fx.ctrl.setup_control.open_entry(EntryRequest("fake_qubit"))
    bridge, call = _mcp_client(fx.service.port, tmp_path)
    try:
        call("connect", {"port": fx.service.port})
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}})
        call("rpc_call", {"method": "soc.info", "params": {"include_cfg": True}})
        observe_run_inputs(
            fx,
            tab_id,
            lambda method, params: call(
                "rpc_call", {"method": method, "params": params}
            ),
        )
        with pytest.raises(RuntimeError) as exc:
            call(
                "rpc_call",
                {
                    "method": "tab.run_start",
                    "params": {
                        "tab_id": tab_id,
                    },
                },
            )
        assert getattr(exc.value, "reason", None) == "no_active_context"
        assert fx.state.get_tab(tab_id).run.result is None
    finally:
        bridge.disconnect()


@pytest.mark.parametrize("terminal", ["finished", "cancelled"])
def test_tab_run_busy_close_and_terminal_keep_the_gui_result(
    fx, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal: str
) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    entered = threading.Event()
    release = threading.Event()
    original_run = SampleAdapter.run

    def held_run(self, request, schema, *, context):
        result = original_run(self, request, schema, context=context)
        entered.set()
        if not release.wait(4):
            raise TimeoutError("fake run release was not signalled")
        return result

    monkeypatch.setattr(SampleAdapter, "run", held_run)
    bridge, call = _mcp_client(fx.service.port, tmp_path)
    try:
        call("connect", {"port": fx.service.port})
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}})
        call("rpc_call", {"method": "context.snapshot"})
        call("rpc_call", {"method": "soc.info", "params": {"include_cfg": True}})
        observe_run_inputs(
            fx,
            tab_id,
            lambda method, params: call(
                "rpc_call", {"method": method, "params": params}
            ),
        )
        op = call(
            "rpc_call",
            {
                "method": "tab.run_start",
                "params": {
                    "tab_id": tab_id,
                },
            },
        )["handle"]
        assert entered.wait(1)
        with pytest.raises(RuntimeError, match="busy"):
            call(
                "rpc_call",
                {
                    "method": "tab.run_start",
                    "params": {
                        "tab_id": tab_id,
                    },
                },
            )
        editor = fx.ctrl.cfg_resources.lookup(tab_id)
        editor.observe()
        for command in (
            lambda: editor.edit((CfgEdit(("gain",), 0.2),)),
            lambda: editor.reset(),
        ):
            with pytest.raises(CfgPreconditionError) as blocked:
                command()
            assert blocked.value.reason is CfgPreconditionReason.MUTATION_BLOCKED
        with pytest.raises(RuntimeError, match="busy"):
            fx.ctrl.close_tab(tab_id)
        other = fx.ctrl.new_tab("fake")
        other_editor = fx.ctrl.cfg_resources.lookup(other)
        other_editor.edit((CfgEdit(("gain",), 0.3),))
        _edit_context_as_gui(fx, "run_source", 17)
        if terminal == "cancelled":
            assert call("cancel", {"op": op})["status"] in {"cancelling", "cancelled"}
        release.set()
        assert call("wait", {"op": op, "timeout": 3})["status"] == terminal
        assert fx.state.get_tab(tab_id).run.result is not None
        fx.ctrl.close_tab(tab_id)
        with pytest.raises(CfgPreconditionError) as gone:
            editor.observe()
        assert gone.value.reason is CfgPreconditionReason.RESOURCE_GONE
    finally:
        release.set()
        fx.ctrl._background_svc.quiesce()
        bridge.disconnect()


@pytest.mark.parametrize("value_bytes", [2 << 20, MAX_LINE_BYTES - 2048])
def test_large_context_roundtrip_restores_load_guard(
    fx, tmp_path: Path, value_bytes: int
) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    bridge, call = _mcp_client(fx.service.port, tmp_path)
    try:
        call("connect", {"port": fx.service.port})
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}})
        call("rpc_call", {"method": "context.snapshot"})
        value = "x" * value_bytes
        _edit_context_as_gui(fx, "large", value)
        _edit_context_as_gui(fx, "changed", 17)
        args = {
            "method": "tab.load_data",
            "params": {"tab_id": tab_id, "data_path": "missing.h5"},
        }
        observed = call("rpc_call", {"method": "context.snapshot"})
        assert observed["point"]["general"]["ext"]["large"] == value
        assert observed["point"]["general"]["ext"]["changed"] == 17
        with pytest.raises(RuntimeError) as missing_file:
            call("rpc_call", args)
        assert getattr(missing_file.value, "reason", None) == "unsupported_load"
    finally:
        bridge.disconnect()


def test_oversized_context_read_returns_error_without_advancing_guard(
    fx, tmp_path: Path
) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    bridge, call = _mcp_client(fx.service.port, tmp_path)
    try:
        call("connect", {"port": fx.service.port})
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}})
        call("rpc_call", {"method": "context.snapshot"})
        _edit_context_as_gui(fx, "large", "x" * (MAX_LINE_BYTES // 2))
        _edit_context_as_gui(fx, "other_large", "x" * (MAX_LINE_BYTES // 2))

        with pytest.raises(RuntimeError) as oversized:
            call("rpc_call", {"method": "context.snapshot"})
        assert getattr(oversized.value, "reason", None) == "response_encoding_failed"
        assert "may have executed" in str(oversized.value)
        args = {
            "method": "tab.load_data",
            "params": {"tab_id": tab_id, "data_path": "missing.h5"},
        }

        _edit_context_as_gui(fx, "large", "small")
        assert (
            call("rpc_call", {"method": "context.snapshot"})["point"]["general"]["ext"][
                "large"
            ]
            == "small"
        )
        with pytest.raises(RuntimeError) as missing_file:
            call("rpc_call", args)
        assert getattr(missing_file.value, "reason", None) == "unsupported_load"
    finally:
        bridge.disconnect()


def test_current_cfg_runs_from_another_holder_connection(fx) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    source = _open_client(fx.service.port)
    runner = _open_client(fx.service.port)
    try:
        assert _raw_call(source, "tab.get_cfg", {"tab_id": tab_id})["ok"]
        args = {"tab_id": tab_id}
        for method, params in (
            ("tab.snapshot", {"tab_id": tab_id}),
            ("soc.info", {"include_cfg": True}),
        ):
            assert _raw_call(runner, method, params)["ok"]
        devices = _raw_call(runner, "device.list")["result"]["devices"]
        for device in devices:
            assert _raw_call(runner, "device.snapshot", {"name": device["name"]})["ok"]
        # Read cfg on one holder connection and run current inputs on the other.
        started = _raw_call(runner, "tab.run_start", args)
        assert started["ok"]
        result = _raw_call(
            runner,
            "operation.await",
            {"operation_id": started["result"]["operation_id"], "timeout": 3.0},
        )
        assert result["result"]["status"] == "finished"
    finally:
        source.close()
        runner.close()


def test_socket_snapshot_exposes_result_replacement(fx) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    sock = _open_client(fx.service.port)
    try:

        def rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
            _send(sock, {"id": method, "method": method, "params": params})
            return _recv_response(sock)

        assert rpc("context.snapshot", {})["ok"] is True
        initial = rpc("tab.snapshot", {"tab_id": tab_id})["result"]["tabs"][0]
        assert initial["result_state"] == {
            "revision": 0,
            "available": False,
            "source_path": None,
            "source_operation_id": None,
        }
        assert initial["analysis_state"]["available"] is False
        assert initial["post_analysis_state"]["available"] is False
        # A non-serializable result stays in State; the operational projection
        # identifies replacements without requiring its payload on the wire.
        fx.state.update_tab_loaded_result(
            tab_id, object(), "loaded.h5", metadata=historical_metadata()
        )
        observed = rpc("tab.snapshot", {"tab_id": tab_id})["result"]["tabs"][0]
        assert observed["result_state"] == {
            "revision": 1,
            "available": True,
            "source_path": "loaded.h5",
            "source_operation_id": None,
        }
        fx.state.update_tab_loaded_result(
            tab_id, object(), "loaded.h5", metadata=historical_metadata()
        )
        replacement = rpc("tab.snapshot", {"tab_id": tab_id})["result"]["tabs"][0]
        assert replacement["result_state"]["revision"] == 2
        assert replacement["result_state"]["source_path"] == "loaded.h5"
    finally:
        sock.close()


def test_frozen_run_does_not_require_large_context_export(fx, tmp_path: Path) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    bridge, call = _mcp_client(fx.service.port, tmp_path)
    try:
        call("connect", {"port": fx.service.port})
        _edit_context_as_gui(fx, "large", "x" * (MAX_LINE_BYTES // 2))
        _edit_context_as_gui(fx, "other_large", "x" * (MAX_LINE_BYTES // 2))
        with pytest.raises(RuntimeError) as oversized:
            call("rpc_call", {"method": "context.snapshot"})
        assert getattr(oversized.value, "reason", None) == "response_encoding_failed"
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}})
        call("rpc_call", {"method": "soc.info", "params": {"include_cfg": True}})
        _edit_context_as_gui(fx, "unrelated", 17)
        started = call(
            "rpc_call",
            {
                "method": "tab.run_start",
                "params": {"tab_id": tab_id},
            },
        )
        _await_completed_run(call, started["handle"])
    finally:
        bridge.disconnect()
