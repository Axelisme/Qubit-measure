"""RemoteControlAdapter transport / dispatch / query tests.

Each test spins up a real TCP socket on an ephemeral loopback port and drives
a fixture Controller (already wired up with a fake adapter). The Qt event loop
is not freely running under pytest, so a helper interleaves
``QApplication.processEvents()`` between socket reads to give marshalled
handlers a chance to execute.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from qick.asm_v2 import QickParam
from zcu_tools.gui.app.measure.adapter import ContextReadiness
from zcu_tools.gui.app.measure.remote import (
    ControlOptions,
)
from zcu_tools.gui.cfg.edit_codec import encode_ref
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
    CfgPreconditionError,
    CfgPreconditionReason,
)
from zcu_tools.gui.remote.framing import MAX_LINE_BYTES
from zcu_tools.program.v2 import WaveformCfgFactory
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from zcu_lab.recipes import RECIPES
from zcu_lab.v2.fake.stub.gui import FakeAdapter

from ._helpers import call as _raw_call
from ._helpers import make_png, observe_run_inputs
from ._helpers import mcp_client as _mcp_client
from ._remote_core_support import (
    RemoteCoreFixture as _Fixture,
)
from ._remote_core_support import (
    open_client as _open_client,
)
from ._remote_core_support import (
    recv_response as _recv_response,
)
from ._remote_core_support import (
    send as _send,
)


@pytest.fixture()
def fx(qapp):
    f = _Fixture()
    f.start()
    try:
        yield f
    finally:
        f.stop()
        # tests/README.md requires joining Controller workers before fixture GC.
        f.ctrl._background_svc.quiesce()  # pyright: ignore[reportPrivateUsage]


pytestmark = pytest.mark.uses_wall_clock


def _prepare_guarded_context(fx: _Fixture, ml: ModuleLibrary | None = None) -> None:
    fx.state.set_context(
        replace(
            fx.state.session_env,
            md=MetaDict(),
            ml=ml if ml is not None else ModuleLibrary(),
            soccfg=make_mock_soccfg(),
        )
    )
    fx.state.version.bump("context")
    fx.state.version.bump("soc")


def _await_completed_run(
    call: Callable[[str, dict[str, Any]], dict[str, Any]], handle: int
) -> None:
    assert call("wait", {"op": handle, "timeout": 3})["status"] == "finished"


def _edit_context_as_gui(fx: _Fixture, key: str, value: object) -> None:
    sock = _open_client(fx.service.port)
    try:
        _send(
            sock,
            {
                "id": "edit",
                "method": "context.md_set_attr",
                "params": {"key": key, "value": value},
            },
        )
        assert _recv_response(sock)["ok"] is True
    finally:
        sock.close()


def test_socket_write_requires_a_full_read_on_that_connection(fx) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    sock = _open_client(fx.service.port)
    try:
        args = {"tab_id": tab_id, "data_path": "missing.h5"}
        _send(sock, {"id": "unread", "method": "tab.load_data", "params": args})
        unread = _recv_response(sock)
        assert unread["error"]["reason"] == "stale_version"
        assert "context" in unread["error"]["data"]["stale"]
        assert f"tab:{tab_id}:result" in unread["error"]["data"]["stale"]
        assert f"tab:{tab_id}:analyze" in unread["error"]["data"]["stale"]

        for method, params in (
            ("tab.snapshot", {"tab_id": tab_id}),
            ("context.snapshot", {}),
        ):
            _send(sock, {"id": method, "method": method, "params": params})
            assert _recv_response(sock)["ok"] is True
        _send(sock, {"id": "read", "method": "tab.load_data", "params": args})
        after_read = _recv_response(sock)
        assert after_read["ok"] is False
        assert after_read["error"].get("reason") != "stale_version"
    finally:
        sock.close()


def test_socket_snapshot_exposes_result_replacement_and_restores_guard(fx) -> None:
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
        fx.state.update_tab_loaded_result(tab_id, object(), "loaded.h5")
        args = {"tab_id": tab_id, "data_path": "missing.h5"}
        assert rpc("tab.load_data", args)["error"]["reason"] == "stale_version"
        observed = rpc("tab.snapshot", {"tab_id": tab_id})["result"]["tabs"][0]
        assert observed["result_state"] == {
            "revision": 1,
            "available": True,
            "source_path": "loaded.h5",
            "source_operation_id": None,
        }
        assert rpc("tab.load_data", args)["error"].get("reason") != "stale_version"
        fx.state.update_tab_loaded_result(tab_id, object(), "loaded.h5")
        assert rpc("tab.load_data", args)["error"]["reason"] == "stale_version"
        replacement = rpc("tab.snapshot", {"tab_id": tab_id})["result"]["tabs"][0]
        assert replacement["result_state"]["revision"] == 2
        assert replacement["result_state"]["source_path"] == "loaded.h5"
        assert rpc("tab.load_data", args)["error"].get("reason") != "stale_version"
    finally:
        sock.close()


def test_socket_self_write_advances_only_its_prior_seen_context(fx) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    first = _open_client(fx.service.port)
    second = _open_client(fx.service.port)
    try:
        for sock in (first, second):
            for method, params in (
                ("tab.snapshot", {"tab_id": tab_id}),
                ("context.snapshot", {}),
            ):
                _send(sock, {"id": method, "method": method, "params": params})
                assert _recv_response(sock)["ok"] is True

        _send(
            first,
            {
                "id": "write",
                "method": "context.md_set_attr",
                "params": {"key": "r_f", "value": 6000.0},
            },
        )
        assert _recv_response(first)["ok"] is True
        args = {"tab_id": tab_id, "data_path": "missing.h5"}
        _send(first, {"id": "self", "method": "tab.load_data", "params": args})
        assert _recv_response(first)["error"].get("reason") != "stale_version"
        _send(second, {"id": "other", "method": "tab.load_data", "params": args})
        assert _recv_response(second)["error"]["reason"] == "stale_version"

        _send(second, {"id": "refresh", "method": "context.snapshot", "params": {}})
        assert _recv_response(second)["result"]["md"]["r_f"] == 6000.0
        _send(second, {"id": "retry", "method": "tab.load_data", "params": args})
        assert _recv_response(second)["error"].get("reason") != "stale_version"
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("stale_cfg", [False, True], ids=["current-ref", "old-ref"])
def test_source_write_requires_the_explicit_current_cfg_ref(fx, stale_cfg) -> None:
    _prepare_guarded_context(fx)
    fx.ctrl.context_control.create_md_attr("count", 4)
    tab_id = fx.ctrl.new_tab("fake")
    from zcu_tools.gui.cfg import EvalValue
    from zcu_tools.gui.cfg.resource import CfgEdit

    cfg = fx.ctrl.cfg_resources.lookup(tab_id)
    cfg.edit(cfg.observe().ref.revision, (CfgEdit(("reps",), EvalValue("count")),))
    sock = _open_client(fx.service.port)
    try:
        for method, params in (
            ("tab.snapshot", {"tab_id": tab_id}),
            ("context.snapshot", {}),
            ("soc.info", {"include_cfg": True}),
            ("device.list", {}),
        ):
            assert _raw_call(sock, method, params)["ok"] is True
        expected = encode_ref(cfg.observe().ref)
        before = cfg.observe().ref
        assert (
            _raw_call(sock, "context.md_set_attr", {"key": "count", "value": 7})["ok"]
            is True
        )
        assert cfg.observe().ref != before
        started = _raw_call(
            sock,
            "tab.run_start",
            {
                "tab_id": tab_id,
                "expected": expected if stale_cfg else encode_ref(cfg.observe().ref),
            },
        )
        if stale_cfg:
            assert started["error"]["reason"] == "stale_revision"
            assert started["error"]["data"] == {
                "expected": expected,
                "actual": encode_ref(cfg.observe().ref),
            }
        else:
            assert started["ok"] is True
            result = _raw_call(
                sock,
                "operation.await",
                {"operation_id": started["result"]["operation_id"], "timeout": 3.0},
            )
            assert result["result"]["status"] == "finished"
    finally:
        sock.close()


def test_cross_connection_cfg_ref_runs_only_after_other_full_reads(fx) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    source = _open_client(fx.service.port)
    runner = _open_client(fx.service.port)
    try:
        publication = _raw_call(source, "tab.get_cfg", {"tab_id": tab_id})["result"]
        args = {"tab_id": tab_id, "expected": publication["cfg_ref"]}
        unread = _raw_call(runner, "tab.run_start", args)
        assert unread["error"]["reason"] == "stale_version"
        assert "soc" in unread["error"]["data"]["stale"]
        for method, params in (
            ("tab.snapshot", {"tab_id": tab_id}),
            ("soc.info", {"include_cfg": True}),
        ):
            assert _raw_call(runner, method, params)["ok"]
        devices = _raw_call(runner, "device.list")["result"]["devices"]
        for device in devices:
            assert _raw_call(runner, "device.snapshot", {"name": device["name"]})["ok"]
        # Only the other connection read cfg. This request supplies that ref.
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


def test_editor_consecutive_self_writes_preserve_commit_observation(fx) -> None:
    ml = ModuleLibrary()
    ml.waveforms["seed"] = WaveformCfgFactory.from_raw(
        {"style": "const", "length": 0.1}
    )
    _prepare_guarded_context(fx, ml)
    first = _open_client(fx.service.port)
    second = _open_client(fx.service.port)
    try:
        editor_id = _raw_call(
            first, "editor.new", {"item_kind": "waveform", "from_name": "seed"}
        )["result"]["editor_id"]
        for sock in (first, second):
            assert _raw_call(sock, "editor.get", {"editor_id": editor_id})["ok"] is True
            assert _raw_call(sock, "context.snapshot", {})["ok"] is True
        for length in (0.2, 0.3):
            assert (
                _raw_call(
                    first,
                    "editor.set_field",
                    {"editor_id": editor_id, "path": "length", "value": length},
                )["ok"]
                is True
            )
        stale = _raw_call(
            second, "editor.commit", {"editor_id": editor_id, "name": "other"}
        )
        assert stale["error"]["reason"] == "stale_version"
        assert (
            _raw_call(first, "editor.commit", {"editor_id": editor_id, "name": "copy"})[
                "ok"
            ]
            is True
        )
        assert ml.waveforms["copy"].to_dict()["length"] == 0.3
        assert "other" not in ml.waveforms
    finally:
        first.close()
        second.close()


def test_timed_out_socket_read_does_not_establish_guard_baseline(fx) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    original = fx.service._method_registry["context.snapshot"]

    def slow_snapshot(adapter, params):
        time.sleep(0.15)
        return original.handler(adapter, params)

    fx.service._method_registry = {
        **fx.service._method_registry,
        "context.snapshot": replace(
            original,
            handler=slow_snapshot,
            spec=replace(original.spec, timeout_seconds=0.01),
        ),
    }
    sock = _open_client(fx.service.port)
    try:
        _send(
            sock,
            {"id": "tab", "method": "tab.snapshot", "params": {"tab_id": tab_id}},
        )
        assert _recv_response(sock)["ok"] is True
        _send(sock, {"id": "slow", "method": "context.snapshot", "params": {}})
        timed_out = _recv_response(sock)
        assert timed_out["ok"] is False
        assert timed_out["error"]["code"] == "timeout"
        _send(
            sock,
            {
                "id": "write",
                "method": "tab.load_data",
                "params": {"tab_id": tab_id, "data_path": "missing.h5"},
            },
        )
        assert _recv_response(sock)["error"]["reason"] == "stale_version"
    finally:
        sock.close()


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
        assert call("rpc_call", {"method": "context.snapshot"})["md"] == {}
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
                    "expected": encode_ref(
                        fx.ctrl.cfg_resources.lookup(tab_id).observe().ref
                    ),
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
                    "expected": encode_ref(
                        fx.ctrl.cfg_resources.lookup(tab_id).observe().ref
                    ),
                },
            },
        )["handle"]
        assert handle > 0
        _await_completed_run(call, handle)
    finally:
        bridge.disconnect()


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize(
    ("recipe", "adapter", "disabled_slots"),
    [
        ("lookback", "lookback", ("reset", "init_pulse")),
        ("twotone_spectrum", "twotone/freq", ("reset",)),
        ("time_rabi", "twotone/rabi/len_rabi", ("reset",)),
        ("amplitude_rabi", "twotone/rabi/amp_rabi", ("reset",)),
        ("t1", "twotone/t1", ("reset",)),
        ("t2ramsey", "twotone/t2ramsey", ("reset",)),
        ("t2echo", "twotone/t2echo", ("reset",)),
        ("singleshot_ge", "singleshot/ge", ("reset", "init_pulse")),
    ],
)
def test_recipe_optional_modules_reach_missing_calibration_on_real_gui(
    fx, tmp_path, request, recipe, adapter, disabled_slots, reuse
):
    _prepare_guarded_context(fx)
    bridge, call = _mcp_client(
        fx.service.port, tmp_path, request=request, recipes=RECIPES
    )
    try:
        call("connect", {"port": fx.service.port})
        arguments = {}
        if reuse:
            arguments["reuse_tab_id"] = call(
                "rpc_call", {"method": "tab.new", "params": {"adapter_name": adapter}}
            )["tab_id"]
        result = call(recipe, arguments)
        assert result["status"] == "needs_parameters", result
        assert result["missing"]
        assert result["run_op"] is None
        tab = result["tab"]
        if reuse:
            assert tab == arguments["reuse_tab_id"]
        cfg = call("rpc_call", {"method": "tab.get_cfg", "params": {"tab_id": tab}})
        modules = cfg["tree"]["children"]["modules"]["children"]
        for slot in disabled_slots:
            assert modules[slot]["ref"] is None
            assert modules[slot]["valid"]
        snapshot = call(
            "rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab}}
        )["tabs"][0]
        assert not snapshot["interaction"]["has_run_result"]
    finally:
        bridge.disconnect()


@pytest.mark.parametrize(
    ("frequency", "length", "offset"),
    [(6000, 2, 0), (6000.25, 2.5, 0.05)],
)
def test_lookback_numbers_prepare_real_cfg_without_connected_soc(
    fx, tmp_path, request, frequency, length, offset
):
    _prepare_guarded_context(fx)
    fx.state.set_context(replace(fx.state.session_env, soc=None))
    bridge, call = _mcp_client(
        fx.service.port, tmp_path, request=request, recipes=RECIPES
    )
    try:
        call("connect", {"port": fx.service.port})
        result = call(
            "lookback",
            {
                "frequency_mhz": frequency,
                "readout_length_us": length,
                "trigger_offset_us": offset,
                "rounds": 1,
            },
        )
        parameters = result["actual"]["parameters"]
        for parameter, expected in (
            ("frequency_mhz", frequency),
            ("ro_frequency_mhz", frequency),
            ("readout_length_us", length),
            ("trigger_offset_us", offset),
        ):
            assert parameters[parameter]["value"] == expected
            assert type(parameters[parameter]["value"]) is float
        assert parameters["rounds"]["value"] == 1
        assert type(parameters["rounds"]["value"]) is int
        assert result["status"] == "failed"
        assert result["run_op"] is None
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
                    "expected": encode_ref(
                        fx.ctrl.cfg_resources.lookup(tab_id).observe().ref
                    ),
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
        assert item["target_name"] == "fake_peak"
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
            call("rpc_call", {"method": "context.snapshot"})["md"]["fake_peak"]
            == item["proposed"]
        )
        image = tmp_path / "fit.png"
        saved = call(
            "rpc_call",
            {
                "method": "tab.save_artifacts",
                "params": {
                    "artifacts": ["analysis:fit"],
                    "paths": {"analysis:fit": str(image)},
                    "tab_id": tab_id,
                },
            },
        )
        assert (
            call("wait", {"op": saved["handle"], "timeout": 5})["status"] == "finished"
        )
        assert image.read_bytes().startswith(b"\x89PNG")
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
                    "expected": encode_ref(
                        fx.ctrl.cfg_resources.lookup(tab_id).observe().ref
                    ),
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
    fx.state.set_context(
        replace(fx.state.session_env, readiness=ContextReadiness.DRAFT)
    )
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
                        "expected": encode_ref(
                            fx.ctrl.cfg_resources.lookup(tab_id).observe().ref
                        ),
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
    original_run = FakeAdapter.run

    def held_run(self, request, schema, *, context):
        result = original_run(self, request, schema, context=context)
        entered.set()
        if not release.wait(4):
            raise TimeoutError("fake run release was not signalled")
        return result

    monkeypatch.setattr(FakeAdapter, "run", held_run)
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
                    "expected": encode_ref(
                        fx.ctrl.cfg_resources.lookup(tab_id).observe().ref
                    ),
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
                        "expected": encode_ref(
                            fx.ctrl.cfg_resources.lookup(tab_id).observe().ref
                        ),
                        "tab_id": tab_id,
                    },
                },
            )
        editor = fx.ctrl.cfg_resources.lookup(tab_id)
        accepted_ref = editor.observe().ref
        for command in (
            lambda: editor.edit(accepted_ref.revision, (CfgEdit(("gain",), 0.2),)),
            lambda: editor.reset(accepted_ref.revision),
        ):
            with pytest.raises(CfgPreconditionError) as blocked:
                command()
            assert blocked.value.reason is CfgPreconditionReason.MUTATION_BLOCKED
        with pytest.raises(RuntimeError, match="busy"):
            fx.ctrl.close_tab(tab_id)
        other = fx.ctrl.new_tab("fake")
        other_editor = fx.ctrl.cfg_resources.lookup(other)
        other_editor.edit(
            other_editor.observe().ref.revision, (CfgEdit(("gain",), 0.3),)
        )
        _edit_context_as_gui(fx, "run_source", 17)
        assert editor.observe().ref != accepted_ref
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


def test_restarted_gui_requires_new_full_reads_before_running(
    qapp, tmp_path: Path
) -> None:
    first = _Fixture()
    port = first.start()
    _prepare_guarded_context(first)
    first_tab = first.ctrl.new_tab("fake")
    bridge, call = _mcp_client(port, tmp_path)
    second: _Fixture | None = None
    try:
        call("connect", {"port": port})
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": first_tab}})
        call("rpc_call", {"method": "context.snapshot"})
        call("rpc_call", {"method": "soc.info", "params": {"include_cfg": True}})
        observe_run_inputs(
            first,
            first_tab,
            lambda method, params: call(
                "rpc_call", {"method": method, "params": params}
            ),
        )
        old_handle = call(
            "rpc_call",
            {
                "method": "tab.run_start",
                "params": {
                    "tab_id": first_tab,
                    "expected": encode_ref(
                        first.ctrl.cfg_resources.lookup(first_tab).observe().ref
                    ),
                },
            },
        )["handle"]
        _await_completed_run(call, old_handle)

        first.stop()
        second = _Fixture(ControlOptions(port=port))
        assert second.start() == port
        _prepare_guarded_context(second)
        second_tab = second.ctrl.new_tab("fake")
        deadline = time.monotonic() + 2
        while bridge.is_connected and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not bridge.is_connected
        call("rpc_list", {"domain": "context"})  # Lazily reconnect to the new GUI.
        with pytest.raises(RuntimeError) as expired:
            call("wait", {"op": old_handle, "timeout": 0.01})
        assert getattr(expired.value, "reason", None) == "unknown_op"
        # The old snapshots must not supply the new GUI's nonzero tab, SoC or
        # context guard baseline. Explicit reads restore access to this tab.
        with pytest.raises(RuntimeError) as stale:
            call(
                "rpc_call",
                {
                    "method": "tab.run_start",
                    "params": {
                        "tab_id": second_tab,
                        "expected": encode_ref(
                            second.ctrl.cfg_resources.lookup(second_tab).observe().ref
                        ),
                    },
                },
            )
        assert getattr(stale.value, "reason", None) == "stale_version"
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": second_tab}})
        call("rpc_call", {"method": "context.snapshot"})
        call("rpc_call", {"method": "soc.info", "params": {"include_cfg": True}})
        observe_run_inputs(
            second,
            second_tab,
            lambda method, params: call(
                "rpc_call", {"method": method, "params": params}
            ),
        )
        new_handle = call(
            "rpc_call",
            {
                "method": "tab.run_start",
                "params": {
                    "tab_id": second_tab,
                    "expected": encode_ref(
                        second.ctrl.cfg_resources.lookup(second_tab).observe().ref
                    ),
                },
            },
        )["handle"]
        assert new_handle > old_handle
        _await_completed_run(call, new_handle)
    finally:
        bridge.disconnect()
        first.stop()
        if second is not None:
            second.stop()


def test_load_after_gui_context_edit_requires_a_new_full_read(
    fx, tmp_path: Path
) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    bridge, call = _mcp_client(fx.service.port, tmp_path)
    try:
        call("connect", {"port": fx.service.port})
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}})
        call("rpc_call", {"method": "context.snapshot"})
        _edit_context_as_gui(fx, "r_f", 6000.0)
        args = {
            "method": "tab.load_data",
            "params": {"tab_id": tab_id, "data_path": "missing.h5"},
        }
        with pytest.raises(RuntimeError) as stale:
            call("rpc_call", args)
        assert getattr(stale.value, "reason", None) == "stale_version"

        assert call("rpc_call", {"method": "context.snapshot"})["md"]["r_f"] == 6000.0
        with pytest.raises(RuntimeError) as missing_file:
            call("rpc_call", args)
        assert getattr(missing_file.value, "reason", None) != "stale_version"
    finally:
        bridge.disconnect()


@pytest.mark.parametrize(
    "lossy",
    [{"1": "first", 1: "second"}, QickParam(start=1.0, spans={"pulse": 0.1})],
    ids=["nested-key-collision", "qick-param"],
)
def test_failed_complete_context_read_does_not_advance_mcp_load_guard(
    fx, tmp_path: Path, lossy: object
) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    bridge, call = _mcp_client(fx.service.port, tmp_path)
    try:
        call("connect", {"port": fx.service.port})
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}})
        call("rpc_call", {"method": "context.snapshot"})
        md = fx.state.session_env.md
        md.update({"nested": lossy})
        fx.state.version.bump("context")
        with pytest.raises(RuntimeError) as unreadable:
            call("rpc_call", {"method": "context.snapshot"})
        assert getattr(unreadable.value, "reason", None) == "unserializable_context"

        args = {
            "method": "tab.load_data",
            "params": {"tab_id": tab_id, "data_path": "missing.h5"},
        }
        with pytest.raises(RuntimeError) as stale:
            call("rpc_call", args)
        assert getattr(stale.value, "reason", None) == "stale_version"

        md.update({"nested": {"first": "a", "second": "b"}})
        fx.state.version.bump("context")
        snapshot = call("rpc_call", {"method": "context.snapshot"})
        assert snapshot["md"]["nested"] == {"first": "a", "second": "b"}
        with pytest.raises(RuntimeError) as missing_file:
            call("rpc_call", args)
        assert getattr(missing_file.value, "reason", None) != "stale_version"
    finally:
        bridge.disconnect()


@pytest.mark.parametrize("mutate_cfg", [False, True])
def test_frozen_run_needs_cfg_observation_not_large_context_export(
    fx, tmp_path: Path, mutate_cfg: bool
) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    from zcu_tools.gui.cfg.resource import CfgEdit

    cfg = fx.ctrl.cfg_resources.lookup(tab_id)
    bridge, call = _mcp_client(fx.service.port, tmp_path)
    try:
        call("connect", {"port": fx.service.port})
        _edit_context_as_gui(fx, "large", "x" * (MAX_LINE_BYTES // 2))
        _edit_context_as_gui(fx, "other_large", "x" * (MAX_LINE_BYTES // 2))
        with pytest.raises(RuntimeError) as oversized:
            call("rpc_call", {"method": "context.snapshot"})
        assert getattr(oversized.value, "reason", None) == "response_encoding_failed"
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}})
        call("rpc_call", {"method": "tab.get_cfg", "params": {"tab_id": tab_id}})
        call("rpc_call", {"method": "soc.info", "params": {"include_cfg": True}})
        _edit_context_as_gui(fx, "unrelated", 17)
        # Every source publication advances cfg; a small complete cfg read
        # establishes the new baseline without exporting the large context.
        call("rpc_call", {"method": "tab.get_cfg", "params": {"tab_id": tab_id}})
        expected = encode_ref(cfg.observe().ref)
        if mutate_cfg:
            cfg.edit(cfg.observe().ref.revision, (CfgEdit(("reps",), 42),))
        args = {
            "method": "tab.run_start",
            "params": {"tab_id": tab_id, "expected": expected},
        }
        if mutate_cfg:
            with pytest.raises(RuntimeError) as stale:
                call("rpc_call", args)
            assert getattr(stale.value, "reason", None) == "stale_revision"
        else:
            started = call("rpc_call", args)
            _await_completed_run(call, started["handle"])
    finally:
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
        call(
            "rpc_call",
            {
                "method": "context.md_set_attr",
                "params": {"key": "large", "value": value},
            },
        )
        _edit_context_as_gui(fx, "changed", 17)
        args = {
            "method": "tab.load_data",
            "params": {"tab_id": tab_id, "data_path": "missing.h5"},
        }
        with pytest.raises(RuntimeError) as stale:
            call("rpc_call", args)
        assert getattr(stale.value, "reason", None) == "stale_version"

        observed = call("rpc_call", {"method": "context.snapshot"})
        assert observed["md"]["large"] == value
        assert observed["md"]["changed"] == 17
        with pytest.raises(RuntimeError) as missing_file:
            call("rpc_call", args)
        assert getattr(missing_file.value, "reason", None) != "stale_version"
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
        with pytest.raises(RuntimeError) as stale:
            call("rpc_call", args)
        assert getattr(stale.value, "reason", None) == "stale_version"

        _edit_context_as_gui(fx, "large", "small")
        assert (
            call("rpc_call", {"method": "context.snapshot"})["md"]["large"] == "small"
        )
        with pytest.raises(RuntimeError) as missing_file:
            call("rpc_call", args)
        assert getattr(missing_file.value, "reason", None) != "stale_version"
    finally:
        bridge.disconnect()


def test_editor_commit_rejects_unseen_gui_edit_then_accepts_full_read(
    fx, tmp_path: Path
) -> None:
    ml = ModuleLibrary()
    ml.waveforms["seed"] = WaveformCfgFactory.from_raw(
        {"style": "const", "length": 0.1}
    )
    _prepare_guarded_context(fx, ml)
    bridge, call = _mcp_client(fx.service.port, tmp_path)
    try:
        call("connect", {"port": fx.service.port})
        editor_id = call(
            "rpc_call",
            {
                "method": "editor.new",
                "params": {"item_kind": "waveform", "from_name": "seed"},
            },
        )["editor_id"]
        call("rpc_call", {"method": "editor.get", "params": {"editor_id": editor_id}})
        call("rpc_call", {"method": "context.snapshot"})
        _edit_context_as_gui(fx, "r_f", 6000.0)
        args = {
            "method": "editor.commit",
            "params": {"editor_id": editor_id, "name": "copy"},
        }
        with pytest.raises(RuntimeError) as stale:
            call("rpc_call", args)
        assert getattr(stale.value, "reason", None) == "stale_version"
        call("rpc_call", {"method": "context.snapshot"})
        assert call("rpc_call", args) == {}
        assert ml.waveforms["copy"] == ml.waveforms["seed"]
    finally:
        bridge.disconnect()
