from __future__ import annotations

import socket

import pytest
from zcu_tools.gui.app.measure.remote.method_entries._registry import (
    RemoteMethodEntry,
    build_dispatch_registry,
    build_method_specs,
    method_entry,
)
from zcu_tools.gui.remote.method_spec import MethodSpec

from ._helpers import Fixture, call, open_client


@pytest.fixture
def remote(tmp_path, qapp):
    fixture = Fixture(project_root=str(tmp_path / "project"))
    fixture.start()
    try:
        yield fixture
    finally:
        fixture.stop()


def _entry(method: str, handler_ref: str) -> RemoteMethodEntry:
    return method_entry(method, handler_ref, MethodSpec(1.0, "test"))


def test_build_method_specs_rejects_duplicate_methods() -> None:
    entries = (
        _entry("dup.method", "operation:h_operation_active"),
        _entry("dup.method", "operation:h_operation_await"),
    )

    with pytest.raises(
        RuntimeError, match="duplicate remote method entries: dup.method"
    ):
        build_method_specs(entries)


@pytest.mark.parametrize(
    ("handler_ref", "message"),
    [
        ("operation", "<handler_module>:<function_name>"),
        ("missing_module:h_missing", "cannot import remote handler module"),
        ("operation:h_missing", "does not name an attribute"),
        ("operation:logger", "is not callable"),
    ],
)
def test_build_dispatch_registry_rejects_bad_handler_refs(
    handler_ref: str, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        build_dispatch_registry((_entry("test.method", handler_ref),))


@pytest.mark.requires_loopback
def test_discovered_methods_are_dispatchable(remote) -> None:
    # An observer reaches validation/admission without mutating application state.
    with socket.create_connection(("127.0.0.1", remote.service.port)) as client:
        assert call(
            client, "control.connect", {"label": "catalog observer", "mode": "readonly"}
        )["ok"]
        catalog = call(client, "rpc.catalog")
        assert catalog["ok"] is True
        methods = catalog["result"]["methods"]
        assert methods
        for entry in methods:
            reply = call(client, entry["method"], {})
            if not reply["ok"]:
                assert reply["error"]["code"] in {
                    "invalid_params",
                    "precondition_failed",
                }, (entry["method"], reply)


@pytest.mark.requires_loopback
def test_dispatch_operation_await_returns_terminal_reply(remote) -> None:
    with open_client(remote.service.port) as client:
        created = call(client, "tab.new", {"adapter_name": "fake"})
        tab_id = created["result"]["tab_id"]
        started = call(client, "tab.run_start", {"tab_id": tab_id})
        assert started["ok"] is True
        reply = call(
            client,
            "operation.await",
            {"operation_id": started["result"]["operation_id"], "timeout": 30},
        )
        assert reply["ok"] is True
        assert reply["result"] == {"reason": "completed", "status": "finished"}
