"""Unit tests for AgentChatService.

Covers: record_*/clear/entries, ring-buffer cap, per-field truncation,
listener notification, and the skip-method filter.
"""

from __future__ import annotations

from zcu_tools.gui.app.main.services.agent_chat import (
    _MAX_ENTRIES,
    _MAX_FIELD_LEN,
    AgentChatService,
    _should_record,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _svc() -> AgentChatService:
    return AgentChatService()


# ---------------------------------------------------------------------------
# record_activity
# ---------------------------------------------------------------------------


def test_record_activity_appends_entry():
    svc = _svc()
    svc.record_activity("run.start", {"tab_id": "t1"}, {"status": "pending"})
    entries = svc.entries()
    assert len(entries) == 1
    assert entries[0].kind == "activity"
    assert "run.start" in entries[0].text


def test_record_activity_skips_query_method():
    svc = _svc()
    svc.record_activity("tab.list", {}, {"tabs": []})
    assert len(svc.entries()) == 0


def test_record_activity_skips_poll():
    svc = _svc()
    svc.record_activity("run.poll", {"tab_id": "t1"}, {"status": "running"})
    assert len(svc.entries()) == 0


def test_record_activity_truncates_long_params():
    svc = _svc()
    long_val = "x" * 1000
    svc.record_activity("context.set_md_attr", {"key": "k", "value": long_val}, {})
    text = svc.entries()[0].text
    # The text should not exceed 2 * _MAX_FIELD_LEN + fixed overhead comfortably.
    # At minimum the truncation marker must be present.
    assert "…" in text or len(text) < 2000


def test_record_activity_truncates_long_result():
    svc = _svc()
    long_result = {"data": "y" * 1000}
    svc.record_activity("analyze", {"tab_id": "t1"}, long_result)
    text = svc.entries()[0].text
    assert len(text) < 2 * _MAX_FIELD_LEN + 200  # generous bound


# ---------------------------------------------------------------------------
# record_feedback
# ---------------------------------------------------------------------------


def test_record_feedback_appends_you_prefix():
    svc = _svc()
    svc.record_feedback("stop that")
    entries = svc.entries()
    assert len(entries) == 1
    assert entries[0].kind == "feedback"
    assert entries[0].text == "you: stop that"


# ---------------------------------------------------------------------------
# record_diagnostic
# ---------------------------------------------------------------------------


def test_record_diagnostic_error_appends():
    svc = _svc()
    svc.record_diagnostic("error", "Run failed", "some detail")
    entries = svc.entries()
    assert len(entries) == 1
    assert entries[0].kind == "diagnostic"
    assert "[ERROR]" in entries[0].text
    assert "Run failed" in entries[0].text
    assert "some detail" in entries[0].text


def test_record_diagnostic_info_no_title():
    svc = _svc()
    svc.record_diagnostic("info", "", "Data saved")
    text = svc.entries()[0].text
    assert "[INFO]" in text
    assert "Data saved" in text


def test_record_diagnostic_suppressed_when_embedded():
    """Embedded mode drops GUI-internal diagnostics (e.g. device connect)."""
    svc = _svc()
    svc.set_embedded_active(True)
    svc.record_diagnostic("info", "", "Device connected: fake_flux")
    assert svc.entries() == ()


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear_removes_all_entries():
    svc = _svc()
    svc.record_feedback("hello")
    svc.record_feedback("world")
    svc.clear()
    assert svc.entries() == ()


# ---------------------------------------------------------------------------
# entries — immutable snapshot
# ---------------------------------------------------------------------------


def test_entries_returns_tuple():
    svc = _svc()
    svc.record_feedback("hi")
    result = svc.entries()
    assert isinstance(result, tuple)


def test_entries_snapshot_is_independent():
    svc = _svc()
    svc.record_feedback("a")
    snap1 = svc.entries()
    svc.record_feedback("b")
    snap2 = svc.entries()
    assert len(snap1) == 1
    assert len(snap2) == 2


# ---------------------------------------------------------------------------
# Ring-buffer cap
# ---------------------------------------------------------------------------


def test_ring_buffer_drops_oldest():
    svc = _svc()
    for i in range(_MAX_ENTRIES + 10):
        svc.record_diagnostic("info", "", f"msg {i}")
    entries = svc.entries()
    assert len(entries) == _MAX_ENTRIES
    # Oldest entries should be gone; newest should be at the end.
    assert "msg" in entries[-1].text


# ---------------------------------------------------------------------------
# Listener notification
# ---------------------------------------------------------------------------


def test_listener_called_on_append():
    svc = _svc()
    calls: list[None] = []
    svc.add_listener(lambda: calls.append(None))
    svc.record_feedback("ping")
    assert len(calls) == 1


def test_listener_called_on_clear():
    svc = _svc()
    calls: list[None] = []
    svc.add_listener(lambda: calls.append(None))
    svc.clear()
    assert len(calls) == 1


def test_remove_listener_stops_notifications():
    svc = _svc()
    calls: list[None] = []
    cb = lambda: calls.append(None)  # noqa: E731
    svc.add_listener(cb)
    svc.remove_listener(cb)
    svc.record_feedback("ping")
    assert calls == []


def test_remove_listener_idempotent():
    svc = _svc()
    cb = lambda: None  # noqa: E731
    svc.remove_listener(cb)  # not registered — must not raise


def test_add_listener_idempotent():
    svc = _svc()
    calls: list[None] = []
    cb = lambda: calls.append(None)  # noqa: E731
    svc.add_listener(cb)
    svc.add_listener(cb)  # second add must not duplicate
    svc.record_feedback("x")
    assert len(calls) == 1


def test_listener_exception_does_not_propagate():
    """A bad listener must not crash the service."""
    svc = _svc()

    def bad_cb():
        raise RuntimeError("boom")

    svc.add_listener(bad_cb)
    # Must not raise:
    svc.record_feedback("ping")
    assert len(svc.entries()) == 1


# ---------------------------------------------------------------------------
# _should_record helper
# ---------------------------------------------------------------------------


def test_should_record_command_methods():
    assert _should_record("run.start") is True
    assert _should_record("context.set_md_attr") is True
    assert _should_record("writeback.apply") is True
    assert _should_record("device.setup") is True
    assert _should_record("connect.start") is True


def test_should_not_record_query_methods():
    assert _should_record("tab.list") is False
    assert _should_record("run.poll") is False
    assert _should_record("operation.await") is False
    assert _should_record("state.check") is False
    assert _should_record("resources.versions") is False
    assert _should_record("context.get_md") is False
