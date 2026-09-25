from __future__ import annotations

import pytest
from zcu_tools.gui.app.main.services.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.app.main.services.remote.handlers.notify import _h_notify_await
from zcu_tools.gui.app.main.services.remote.method_entries import METHOD_ENTRIES
from zcu_tools.gui.app.main.services.remote.method_entries._registry import (
    RemoteMethodEntry,
    build_dispatch_registry,
    build_method_specs,
    method_entry,
)
from zcu_tools.gui.app.main.services.remote.method_specs import METHOD_SPECS
from zcu_tools.gui.remote.method_spec import MethodSpec


def _entry(method: str, handler_ref: str) -> RemoteMethodEntry:
    return method_entry(method, handler_ref, MethodSpec(1.0, "test"))


def test_build_method_specs_rejects_duplicate_methods() -> None:
    entries = (
        _entry("dup.method", "notify:_h_notify_open"),
        _entry("dup.method", "notify:_h_notify_await"),
    )

    with pytest.raises(
        RuntimeError, match="duplicate remote method entries: dup.method"
    ):
        build_method_specs(entries)


@pytest.mark.parametrize(
    ("handler_ref", "message"),
    [
        ("notify", "<handler_module>:<function_name>"),
        ("missing_module:_h_missing", "cannot import remote handler module"),
        ("notify:_h_missing", "does not name an attribute"),
        ("notify:logger", "is not callable"),
    ],
)
def test_build_dispatch_registry_rejects_bad_handler_refs(
    handler_ref: str, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        build_dispatch_registry((_entry("test.method", handler_ref),))


def test_method_entries_are_single_registry_source() -> None:
    entry_methods = {entry.method for entry in METHOD_ENTRIES}

    assert entry_methods == set(METHOD_SPECS)
    assert entry_methods == set(METHOD_REGISTRY)


def test_handler_refs_follow_wire_method_name_convention() -> None:
    for entry in METHOD_ENTRIES:
        _, function_name = entry.handler_ref.split(":", 1)
        assert function_name == "_h_" + entry.method.replace(".", "_")


def test_dispatch_registry_resolves_real_handler_identity() -> None:
    assert METHOD_REGISTRY["notify.await"].handler is _h_notify_await
