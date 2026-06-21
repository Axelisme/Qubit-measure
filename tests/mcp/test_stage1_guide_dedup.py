"""Tests for gui_run_stage1 guide deduplication — Phase 167 commit 2.

The guide is a static, per-adapter classmethod result (~7.8 KB). Sending it on
every stage1 call within the same MCP server session wastes tokens. The policy
(implemented in tool_gui_run_stage1 + the module-level _GUIDE_SENT set):
  - First call for an adapter_name in this session -> reply includes full 'guide'.
  - Subsequent calls for the SAME adapter -> 'guide' is absent, 'guide_omitted'
    is True.
  - Different adapter names are tracked independently (each gets its own first
    guide send).

Isolation strategy: the measure server imports Qt and GUI layers at module level.
Tests stub those heavy dependencies into sys.modules before importing the server,
then import the server module once per fixture scope. Between tests _GUIDE_SENT
is cleared via monkeypatch to prevent state leakage.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# One-time fixture: stub heavy dependencies and import the server module.
# The server module is imported once per session; _GUIDE_SENT is cleared per
# test via the module-level ``guide_dedup_env`` fixture.
# ---------------------------------------------------------------------------


def _stub_heavy_deps() -> None:
    """Inject stubs for Qt and GUI layers so the server module can be imported
    without a live Qt installation or GUI process."""
    # Qt stubs — need __spec__ (non-None) to pass the server's find_spec guard,
    # and __path__ so sub-module imports (qtpy.QtCore, etc.) don't explode.
    for dep in (
        "qtpy",
        "qtpy.QtCore",
        "qtpy.QtWidgets",
        "qtpy.QtGui",
        "qtpy.QtSvg",
        "qtpy.QtOpenGL",
        "PyQt6",
        "PyQt6.QtCore",
        "PyQt6.QtWidgets",
        "PyQt6.QtGui",
    ):
        if dep not in sys.modules:
            m = types.ModuleType(dep)
            m.__spec__ = object()  # type: ignore[attr-defined]
            m.__path__ = []  # type: ignore[attr-defined]
            sys.modules[dep] = m

    # GUI service stubs — only the two names the server's top-level imports
    # reference directly (method_specs.METHOD_SPECS, wire_version.WIRE_VERSION).
    method_specs_mod = types.ModuleType(
        "zcu_tools.gui.app.main.services.remote.method_specs"
    )
    method_specs_mod.METHOD_SPECS = {}  # type: ignore[attr-defined]
    wire_version_mod = types.ModuleType(
        "zcu_tools.gui.app.main.services.remote.wire_version"
    )
    wire_version_mod.WIRE_VERSION = 99  # type: ignore[attr-defined]
    for name, mod in [
        (
            "zcu_tools.gui.app.main.services.remote.method_specs",
            method_specs_mod,
        ),
        (
            "zcu_tools.gui.app.main.services.remote.wire_version",
            wire_version_mod,
        ),
    ]:
        sys.modules.setdefault(name, mod)


# Stub before any test collection touches the server module.
_stub_heavy_deps()

import zcu_tools.mcp.measure.server as _srv  # noqa: E402  (after stubs)

# ---------------------------------------------------------------------------
# Per-test fixture: clear _GUIDE_SENT and patch send_gui_rpc + _fold_tab_editing_context
# ---------------------------------------------------------------------------


@pytest.fixture()
def guide_dedup_env(monkeypatch: pytest.MonkeyPatch):
    """Reset _GUIDE_SENT and patch the two I/O functions stage1 calls.

    send_gui_rpc is monkeypatched to a fake that:
      - tab.new        -> {"tab_id": "fake-tab-id"}
      - adapter.guide  -> {"guide": "GUIDE_TEXT_FOR_<adapter_name>"}
      - (any other method) -> {}

    _fold_tab_editing_context is patched to a no-op that adds predictable keys
    ({editor_id, tree}) without making real RPC calls.
    """
    monkeypatch.setattr(_srv, "_GUIDE_SENT", set())

    def _fake_send_gui_rpc(
        method: str, params: dict[str, Any], *args: Any
    ) -> dict[str, Any]:
        if method == "tab.new":
            return {"tab_id": "fake-tab-id"}
        if method == "adapter.guide":
            name = params.get("adapter_name", "unknown")
            return {"guide": f"GUIDE_TEXT_FOR_{name}"}
        return {}

    monkeypatch.setattr(_srv, "send_gui_rpc", _fake_send_gui_rpc)

    def _fake_fold(tab_id: str, reply: dict[str, Any]) -> dict[str, Any]:
        # Simulate what _fold_tab_editing_context adds without real RPCs.
        reply["editor_id"] = "fake-editor-id"
        reply["tree"] = {}
        return reply

    monkeypatch.setattr(_srv, "_fold_tab_editing_context", _fake_fold)

    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_first_call_returns_guide(guide_dedup_env: None) -> None:
    """First stage1 call for an adapter includes the full 'guide' in the reply."""
    result = _srv.tool_gui_run_stage1({"adapter_name": "onetone"})
    assert "guide" in result
    assert result["guide"] == "GUIDE_TEXT_FOR_onetone"
    assert "guide_omitted" not in result


def test_first_call_guide_sent_adds_to_set(guide_dedup_env: None) -> None:
    """After the first call, the adapter name is recorded in _GUIDE_SENT."""
    _srv.tool_gui_run_stage1({"adapter_name": "onetone"})
    assert "onetone" in _srv._GUIDE_SENT


def test_second_call_omits_guide(guide_dedup_env: None) -> None:
    """Second stage1 call for the same adapter: no 'guide', 'guide_omitted' is True."""
    _srv.tool_gui_run_stage1({"adapter_name": "onetone"})
    result2 = _srv.tool_gui_run_stage1({"adapter_name": "onetone"})
    assert "guide" not in result2
    assert result2.get("guide_omitted") is True


def test_repeated_call_still_has_other_fields(guide_dedup_env: None) -> None:
    """Even when guide is omitted, the standard fields (tab_id, adapter,
    editor_id, tree) are still present in the reply."""
    _srv.tool_gui_run_stage1({"adapter_name": "twotone"})
    result2 = _srv.tool_gui_run_stage1({"adapter_name": "twotone"})
    assert result2["tab_id"] == "fake-tab-id"
    assert result2["adapter"] == "twotone"
    assert "editor_id" in result2
    assert "tree" in result2


def test_different_adapters_each_get_guide(guide_dedup_env: None) -> None:
    """Two different adapter names are tracked independently; each receives its
    own first-use guide."""
    r1 = _srv.tool_gui_run_stage1({"adapter_name": "onetone"})
    r2 = _srv.tool_gui_run_stage1({"adapter_name": "rabi"})
    assert "guide" in r1
    assert "guide" in r2
    assert r1["guide"] == "GUIDE_TEXT_FOR_onetone"
    assert r2["guide"] == "GUIDE_TEXT_FOR_rabi"


def test_different_adapter_repeat_still_omits_for_first(guide_dedup_env: None) -> None:
    """After sending guide for both adapters, repeating either one gives guide_omitted."""
    _srv.tool_gui_run_stage1({"adapter_name": "onetone"})
    _srv.tool_gui_run_stage1({"adapter_name": "rabi"})
    r3 = _srv.tool_gui_run_stage1({"adapter_name": "onetone"})
    r4 = _srv.tool_gui_run_stage1({"adapter_name": "rabi"})
    assert "guide" not in r3 and r3.get("guide_omitted") is True
    assert "guide" not in r4 and r4.get("guide_omitted") is True


def test_guide_dedup_isolated_between_tests(guide_dedup_env: None) -> None:
    """Confirm that _GUIDE_SENT is empty at test start (fixture reset works)."""
    # No prior calls in this test; the set must be empty.
    assert len(_srv._GUIDE_SENT) == 0
