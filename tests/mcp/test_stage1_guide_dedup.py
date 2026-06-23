"""Tests for gui_tab_open guide semantics — Phase 171 skip_guide flip.

The guide policy (implemented in tool_gui_tab_open):
  - DEFAULT (skip_guide omitted or False) -> reply always includes full 'guide'.
    This ensures fresh contexts, sub-agents, and context-reset sessions always
    receive the orientation text without having to remember a flag.
  - skip_guide=True -> guide fetch is suppressed; reply carries 'guide_omitted: True'.
    Only the caller (the agent) knows whether its context already has the guide.

There is no longer a server-side _GUIDE_SENT set — the dedup decision is the
caller's responsibility.

Isolation strategy: the measure server imports Qt and GUI layers at module level.
Tests stub those heavy dependencies into sys.modules before importing the server,
then import the server module once per session scope. Between tests the patched
send_gui_rpc / _fold_tab_editing_context are reset via monkeypatch.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# One-time fixture: stub heavy dependencies and import the server module.
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
# Per-test fixture: patch send_gui_rpc + _fold_tab_editing_context
# ---------------------------------------------------------------------------


@pytest.fixture()
def tab_open_env(monkeypatch: pytest.MonkeyPatch):
    """Patch the two I/O functions that tool_gui_tab_open calls.

    send_gui_rpc is monkeypatched to a fake that:
      - tab.new        -> {"tab_id": "fake-tab-id"}
      - adapter.guide  -> {"guide": "GUIDE_TEXT_FOR_<adapter_name>"}
      - (any other method) -> {}

    _fold_tab_editing_context is patched to a no-op that adds predictable keys
    ({editor_id, tree}) without making real RPC calls.
    """

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
        reply["editor_id"] = "fake-editor-id"
        reply["tree"] = {}
        return reply

    monkeypatch.setattr(_srv, "_fold_tab_editing_context", _fake_fold)

    yield


# ---------------------------------------------------------------------------
# Tests — default behaviour (guide always sent)
# ---------------------------------------------------------------------------


def test_default_call_returns_guide(tab_open_env: None) -> None:
    """Default gui_tab_open (no skip_guide) always includes the full guide."""
    result = _srv.tool_gui_tab_open({"adapter_name": "onetone"})
    assert "guide" in result
    assert result["guide"] == "GUIDE_TEXT_FOR_onetone"
    assert "guide_omitted" not in result


def test_second_default_call_still_returns_guide(tab_open_env: None) -> None:
    """Repeated gui_tab_open without skip_guide still includes the guide — there
    is no server-side dedup that would suppress it on the second call."""
    _srv.tool_gui_tab_open({"adapter_name": "onetone"})
    result2 = _srv.tool_gui_tab_open({"adapter_name": "onetone"})
    assert "guide" in result2
    assert result2["guide"] == "GUIDE_TEXT_FOR_onetone"
    assert "guide_omitted" not in result2


def test_default_call_has_standard_fields(tab_open_env: None) -> None:
    """Reply always contains tab_id, adapter, editor_id, tree alongside guide."""
    result = _srv.tool_gui_tab_open({"adapter_name": "twotone"})
    assert result["tab_id"] == "fake-tab-id"
    assert result["adapter"] == "twotone"
    assert "editor_id" in result
    assert "tree" in result
    assert "guide" in result


def test_different_adapters_each_get_guide(tab_open_env: None) -> None:
    """Each adapter name returns its own guide — no cross-adapter interference."""
    r1 = _srv.tool_gui_tab_open({"adapter_name": "onetone"})
    r2 = _srv.tool_gui_tab_open({"adapter_name": "rabi"})
    assert r1["guide"] == "GUIDE_TEXT_FOR_onetone"
    assert r2["guide"] == "GUIDE_TEXT_FOR_rabi"


# ---------------------------------------------------------------------------
# Tests — skip_guide=True behaviour
# ---------------------------------------------------------------------------


def test_skip_guide_omits_guide(tab_open_env: None) -> None:
    """skip_guide=True suppresses the guide fetch; reply carries guide_omitted."""
    result = _srv.tool_gui_tab_open({"adapter_name": "onetone", "skip_guide": True})
    assert "guide" not in result
    assert result.get("guide_omitted") is True


def test_skip_guide_still_has_standard_fields(tab_open_env: None) -> None:
    """Even with skip_guide=True the standard fields (tab_id, adapter,
    editor_id, tree) are present."""
    result = _srv.tool_gui_tab_open({"adapter_name": "rabi", "skip_guide": True})
    assert result["tab_id"] == "fake-tab-id"
    assert result["adapter"] == "rabi"
    assert "editor_id" in result
    assert "tree" in result


def test_skip_guide_false_explicit_sends_guide(tab_open_env: None) -> None:
    """Explicitly passing skip_guide=False is identical to the default."""
    result = _srv.tool_gui_tab_open({"adapter_name": "t1", "skip_guide": False})
    assert "guide" in result
    assert "guide_omitted" not in result


# ---------------------------------------------------------------------------
# Tests — fresh-context / sub-agent scenario
# ---------------------------------------------------------------------------


def test_fresh_context_no_prior_state_gets_guide(tab_open_env: None) -> None:
    """Simulates a fresh agent context (no prior server state for this adapter):
    a call without skip_guide always returns the guide regardless of whether
    another agent has previously opened the same adapter in this server process.
    Because the server holds no _GUIDE_SENT set, the fresh agent is not starved."""
    # Simulate "another agent already opened onetone" — but since there is no
    # _GUIDE_SENT set, the new context simply calls without skip_guide and gets
    # the guide unconditionally.
    _ = _srv.tool_gui_tab_open({"adapter_name": "onetone"})  # "prior agent"
    # "Fresh context / new sub-agent" opens the same adapter without skip_guide.
    fresh_result = _srv.tool_gui_tab_open({"adapter_name": "onetone"})
    assert "guide" in fresh_result
    assert fresh_result["guide"] == "GUIDE_TEXT_FOR_onetone"
    assert "guide_omitted" not in fresh_result
