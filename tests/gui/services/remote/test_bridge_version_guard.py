"""mcp_server-side optimistic-concurrency bookkeeping.

Drives send_gui_rpc over a synchronous FakeTransport injected into the bridge:
each send_line echoes a per-method crafted reply, so we assert the mcp policy
(attach expected_versions for guarded ops, translate a stale rejection, refresh
_LAST_SEEN after every successful RPC) without a real GUI.

Post-E4/F: socket I/O lives behind the McpBridge transport seam — tests inject a
synchronous ``FakeTransport`` (no socket, no thread) and run the bridge's REAL
``send_rpc_raw``; the mcp policy (``send_gui_rpc`` / ``_LAST_SEEN`` / guard) stays
on ``mcp_server`` and is asserted there. No socket internals are patched.
"""

from __future__ import annotations

from typing import Any

import pytest
from zcu_tools.mcp.measure import server as mcp_server

from ._helpers import FakeTransport


@pytest.fixture()
def wired(monkeypatch):
    """Inject a FakeTransport into the bridge; reset the guard baseline.

    Returns a dict you populate as ``{method: reply_envelope}``; ``["sent"]``
    records every outgoing ``(method, params)`` so tests can assert what the guard
    attached.
    """
    fake = FakeTransport()
    mcp_server._BRIDGE.set_transport(fake)
    monkeypatch.setattr(mcp_server, "_LAST_SEEN", {}, raising=False)
    replies: dict[str, dict[str, Any]] = fake.replies
    replies["sent"] = fake.sent  # type: ignore[assignment]
    yield replies
    mcp_server._BRIDGE.set_transport(None)


def _versions_reply(table: dict[str, int]) -> dict[str, Any]:
    return {"ok": True, "result": {"versions": table}}


def test_guarded_op_attaches_expected_versions(wired):
    sent = wired["sent"]
    # Baseline the agent has observed.
    mcp_server._LAST_SEEN.update(
        {
            "tab:t:cfg": 3,
            "tab:t": 1,
            "soc": 2,
            "context": 4,
            "device:yoko": 5,
            "devices:__set__": 6,
        }
    )
    wired["tab.run_start"] = {"ok": True, "result": {}}
    wired["resources.versions"] = _versions_reply(dict(mcp_server._LAST_SEEN))

    mcp_server.send_gui_rpc("tab.run_start", {"tab_id": "t"})

    run_params = next(p for (m, p) in sent if m == "tab.run_start")
    assert run_params["expected_versions"] == {
        "tab:t:cfg": 3,
        "tab:t": 1,
        "soc": 2,
        "context": 4,
        "device:yoko": 5,
        "devices:__set__": 6,
    }


def test_run_start_declares_device_set_cardinality_key(wired):
    """tab.run_start must declare devices:__set__ so a concurrently-added device
    (which device:* glob cannot reveal) is caught by the guard."""
    sent = wired["sent"]
    # Agent observed an empty device set (cardinality key unseen → 0).
    mcp_server._LAST_SEEN.update({"tab:t:cfg": 1, "tab:t": 1, "soc": 1, "context": 1})
    wired["tab.run_start"] = {"ok": True, "result": {}}
    wired["resources.versions"] = _versions_reply(dict(mcp_server._LAST_SEEN))

    mcp_server.send_gui_rpc("tab.run_start", {"tab_id": "t"})

    expected = next(p for (m, p) in sent if m == "tab.run_start")["expected_versions"]
    # Declared at its last-seen baseline of 0; the server rejects if a device was
    # added since (cardinality now ≥ 1).
    assert expected["devices:__set__"] == 0


def test_save_depends_on_result_and_save_path_not_cfg(wired):
    sent = wired["sent"]
    mcp_server._LAST_SEEN.update(
        {"tab:t:result": 7, "tab:t:save_path": 2, "tab:t:cfg": 9}
    )
    wired["tab.save_data"] = {"ok": True, "result": {}}
    wired["resources.versions"] = _versions_reply(dict(mcp_server._LAST_SEEN))

    mcp_server.send_gui_rpc("tab.save_data", {"tab_id": "t"})

    params = next(p for (m, p) in sent if m == "tab.save_data")
    assert params["expected_versions"] == {"tab:t:result": 7, "tab:t:save_path": 2}
    assert "tab:t:cfg" not in params["expected_versions"]


def test_writeback_apply_depends_on_result_analyze_and_context(wired):
    sent = wired["sent"]
    mcp_server._LAST_SEEN.update(
        {"tab:t:result": 7, "tab:t:analyze": 4, "context": 9, "tab:t:save_path": 2}
    )
    wired["tab.writeback_apply"] = {"ok": True, "result": {}}
    wired["resources.versions"] = _versions_reply(dict(mcp_server._LAST_SEEN))

    mcp_server.send_gui_rpc("tab.writeback_apply", {"tab_id": "t", "selections": []})

    params = next(p for (m, p) in sent if m == "tab.writeback_apply")
    assert params["expected_versions"] == {
        "tab:t:result": 7,
        "tab:t:analyze": 4,
        "context": 9,
    }
    # save_path is irrelevant to tab.writeback_apply.
    assert "tab:t:save_path" not in params["expected_versions"]


def test_unguarded_op_attaches_nothing(wired):
    sent = wired["sent"]
    wired["tab.snapshot"] = {"ok": True, "result": {"x": 1}}
    wired["resources.versions"] = _versions_reply({})

    mcp_server.send_gui_rpc("tab.snapshot", {"tab_id": "t"})

    params = next(p for (m, p) in sent if m == "tab.snapshot")
    assert "expected_versions" not in params


def test_stale_rejection_translated_and_refreshes(wired):
    mcp_server._LAST_SEEN.update({"tab:t:cfg": 3, "tab:t": 1, "soc": 1, "context": 1})
    wired["tab.run_start"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "stale_version",
            "message": "stale",
        },
    }
    # After rejection the bridge re-reads the table; the human's edit bumped cfg.
    wired["resources.versions"] = _versions_reply({"tab:t:cfg": 4})

    with pytest.raises(RuntimeError) as ei:
        mcp_server.send_gui_rpc("tab.run_start", {"tab_id": "t"})

    msg = str(ei.value)
    assert "PRECONDITION_FAILED" in msg
    # No raw version numbers leak into the agent-facing message.
    assert "4" not in msg and "tab:t:cfg" not in msg
    # _LAST_SEEN was resynced from the post-rejection read.
    assert mcp_server._LAST_SEEN == {"tab:t:cfg": 4}


def test_successful_rpc_refreshes_last_seen(wired):
    wired["state.has_soc"] = {"ok": True, "result": {"value": True}}
    wired["resources.versions"] = _versions_reply({"soc": 9, "context": 1})

    mcp_server.send_gui_rpc("state.has_soc", {})

    assert mcp_server._LAST_SEEN == {"soc": 9, "context": 1}


def test_device_glob_expands_to_all_device_keys(wired):
    sent = wired["sent"]
    mcp_server._LAST_SEEN.update(
        {
            "tab:t:cfg": 1,
            "tab:t": 1,
            "soc": 1,
            "context": 1,
            "device:yoko": 2,
            "device:sgs": 3,
        }
    )
    wired["tab.run_start"] = {"ok": True, "result": {}}
    wired["resources.versions"] = _versions_reply(dict(mcp_server._LAST_SEEN))

    mcp_server.send_gui_rpc("tab.run_start", {"tab_id": "t"})

    expected = next(p for (m, p) in sent if m == "tab.run_start")["expected_versions"]
    assert expected["device:yoko"] == 2
    assert expected["device:sgs"] == 3


# ---------------------------------------------------------------------------
# Read-reveal guard refresh (MCP_VERSION 62)
#
# These pin the read/write split: a pure read in ``_READ_REVEALS`` refreshes
# ONLY the keys it revealed; a write (or unclassified read) refreshes the whole
# table. The mcp side's job is to build the right ``expected_versions``; a write
# whose ``expected_versions[X]`` lags the GUI's current X version is what the
# server then rejects (the GUI-side guard atomicity is covered by
# test_stale_guard.py). So here "would be rejected" == the attached baseline for
# X is BELOW the current GUI version of X; "would pass" == it equals it.
#
# The FakeTransport returns ``wired["resources.versions"]`` for every refresh
# read, so a concurrent (human) bump is simulated by swapping that reply between
# two send_gui_rpc calls.
# ---------------------------------------------------------------------------


def _last_run_expected(sent) -> dict:
    return next(p for (m, p) in reversed(sent) if m == "tab.run_start")[
        "expected_versions"
    ]


def test_read_reveal_no_concurrency_write_not_rejected(wired):
    """Scenario 1 — no concurrency: read a mapped resource (tab cfg), then write
    the same tab. The write's cfg baseline must equal the current version, so no
    false positive."""
    sent = wired["sent"]
    table = {"tab:t:cfg": 3, "tab:t": 1, "soc": 1, "context": 1}
    wired["tab.get_cfg"] = {"ok": True, "result": {"raw": {}}}
    wired["tab.run_start"] = {"ok": True, "result": {}}
    wired["resources.versions"] = _versions_reply(dict(table))

    mcp_server.send_gui_rpc("tab.get_cfg", {"tab_id": "t"})
    mcp_server.send_gui_rpc("tab.run_start", {"tab_id": "t"})

    # Baseline kept up with the current cfg version → server would not reject.
    assert _last_run_expected(sent)["tab:t:cfg"] == 3


def test_read_reveal_adjacent_conflict_write_rejected(wired):
    """Scenario 2 — adjacent conflict: read tab cfg, then a concurrent edit bumps
    that very cfg, then write the same tab. The write's cfg baseline must lag the
    current version (the guard still bites)."""
    sent = wired["sent"]
    before = {"tab:t:cfg": 3, "tab:t": 1, "soc": 1, "context": 1}
    wired["tab.get_cfg"] = {"ok": True, "result": {"raw": {}}}
    wired["tab.run_start"] = {"ok": True, "result": {}}
    wired["resources.versions"] = _versions_reply(dict(before))

    mcp_server.send_gui_rpc("tab.get_cfg", {"tab_id": "t"})
    # Concurrent human edit bumps the tab's cfg AFTER the read observed v3.
    after = dict(before)
    after["tab:t:cfg"] = 4
    wired["resources.versions"] = _versions_reply(after)

    mcp_server.send_gui_rpc("tab.run_start", {"tab_id": "t"})

    # The read pinned cfg at v3; the run attaches v3 while the GUI is now at v4 →
    # the server rejects (baseline below current).
    assert _last_run_expected(sent)["tab:t:cfg"] == 3


def test_read_reveal_absorption_fixed(wired):
    """Scenario 3 — the read-absorption bug is fixed: read X (tab cfg), a
    concurrent edit bumps X, then read an UNRELATED mapped resource (a device
    snapshot), then write X. The unrelated read's narrow refresh must NOT touch
    X's baseline, so the write to X is still rejected."""
    sent = wired["sent"]
    before = {
        "tab:t:cfg": 3,
        "tab:t": 1,
        "soc": 1,
        "context": 1,
        "device:yoko": 5,
    }
    wired["tab.get_cfg"] = {"ok": True, "result": {"raw": {}}}
    wired["device.snapshot"] = {"ok": True, "result": {"snapshot": {}}}
    wired["tab.run_start"] = {"ok": True, "result": {}}
    wired["resources.versions"] = _versions_reply(dict(before))

    mcp_server.send_gui_rpc("tab.get_cfg", {"tab_id": "t"})  # observes cfg v3
    # Concurrent human edit bumps the tab's cfg to v4.
    after = dict(before)
    after["tab:t:cfg"] = 4
    wired["resources.versions"] = _versions_reply(after)
    # An unrelated mapped read (device snapshot) — its narrow refresh reveals only
    # device:yoko, so it must leave the (now stale) cfg baseline at v3.
    mcp_server.send_gui_rpc("device.snapshot", {"name": "yoko"})
    # Direct evidence the device read did NOT absorb the cfg bump.
    assert mcp_server._LAST_SEEN["tab:t:cfg"] == 3

    mcp_server.send_gui_rpc("tab.run_start", {"tab_id": "t"})

    # cfg baseline NOT absorbed by the device read → still v3 while GUI is v4 →
    # server rejects. This is the bug the read-reveal split fixes.
    assert _last_run_expected(sent)["tab:t:cfg"] == 3
    # And the device read DID refresh its own revealed key.
    assert mcp_server._LAST_SEEN["device:yoko"] == 5


def test_write_whole_table_refresh_self_block_safe(wired):
    """Scenario 4 — self-block safety: a WRITE that bumps context (editor.commit)
    refreshes the whole table, so a follow-up op depending on context is NOT
    falsely rejected by the agent's own bump.

    The commit wrote md/ml, so by the time it lands the GUI has already bumped
    context 7 → 8; the post-commit whole-table refresh reads that 8 into the
    baseline. The next run depending on context must therefore attach 8 (its own
    bump absorbed), not a value below it."""
    sent = wired["sent"]
    # The table the post-commit refresh sees: context already at its bumped value
    # (the commit's own write). A write whole-table refreshes, so the baseline
    # absorbs this 8.
    post_commit = {"editor:e": 2, "context": 8, "tab:t:cfg": 1, "tab:t": 1, "soc": 1}
    wired["editor.commit"] = {"ok": True, "result": {}}
    wired["tab.run_start"] = {"ok": True, "result": {}}
    wired["resources.versions"] = _versions_reply(dict(post_commit))

    mcp_server.send_gui_rpc("editor.commit", {"editor_id": "e", "name": "m"})
    mcp_server.send_gui_rpc("tab.run_start", {"tab_id": "t"})

    # The run depends on context; its baseline is the post-commit value the
    # whole-table refresh absorbed (8) → equals current → no false reject.
    assert _last_run_expected(sent)["context"] == 8


def test_unmapped_read_still_whole_table(wired):
    """Scenario 5 — an unmapped read keeps the old whole-table behaviour: reading
    a method NOT in _READ_REVEALS replaces the entire baseline, so a follow-up op
    is not falsely rejected (and unrelated keys are refreshed too)."""
    sent = wired["sent"]
    assert "soc.info" not in mcp_server._READ_REVEALS  # guards the premise
    wired["soc.info"] = {"ok": True, "result": {"description": "x", "is_mock": True}}
    wired["tab.run_start"] = {"ok": True, "result": {}}
    table = {"tab:t:cfg": 9, "tab:t": 2, "soc": 4, "context": 6}
    wired["resources.versions"] = _versions_reply(dict(table))

    mcp_server.send_gui_rpc("soc.info", {})
    # Unmapped read replaced the WHOLE baseline (old behaviour preserved).
    assert mcp_server._LAST_SEEN == table

    mcp_server.send_gui_rpc("tab.run_start", {"tab_id": "t"})
    expected = _last_run_expected(sent)
    assert expected["soc"] == 4
    assert expected["tab:t:cfg"] == 9


# ---------------------------------------------------------------------------
# Counter-examples: these tests would FAIL (assert wrong) if the removed /
# narrowed entries were still present, proving they were over-broad.
# "Over-broad" = a read that masks a concurrent edit (false-ACCEPT / masking),
# NOT a false-reject — the guard FAILS TO FIRE when it should.
# ---------------------------------------------------------------------------


def test_device_list_does_not_mask_device_info_change(wired):
    """Counter-example (a) — device.list is narrow (devices:__set__ only).

    If device.list were mapped to device:* it would advance device:yoko's baseline
    to current, masking a concurrent device-info bump. After the fix device.list
    only reveals devices:__set__, so a concurrent device:yoko bump is still caught
    by the next tab.run_start (which guards on device:*)."""
    sent = wired["sent"]
    assert "device.list" in mcp_server._READ_REVEALS
    # Sanity: the fixed entry must NOT include device:*.
    assert "device:*" not in mcp_server._READ_REVEALS["device.list"]

    before = {
        "tab:t:cfg": 1,
        "tab:t": 1,
        "soc": 1,
        "context": 1,
        "device:yoko": 5,
        "devices:__set__": 2,
    }
    wired["device.list"] = {"ok": True, "result": {"devices": []}}
    wired["tab.run_start"] = {"ok": True, "result": {}}
    # Prime the baseline with device:yoko=5 (agent previously observed it).
    # device.list's narrow refresh must leave device:yoko at 5, NOT advance it.
    mcp_server._LAST_SEEN.update(before)
    wired["resources.versions"] = _versions_reply(dict(before))

    mcp_server.send_gui_rpc("device.list", {})
    # Concurrent device info/value change bumps device:yoko 5 → 6.
    after = dict(before)
    after["device:yoko"] = 6
    wired["resources.versions"] = _versions_reply(after)

    mcp_server.send_gui_rpc("tab.run_start", {"tab_id": "t"})

    expected = _last_run_expected(sent)
    # device.list narrow refresh only updated devices:__set__; device:yoko must
    # still carry the stale baseline (5), which lags the current GUI value (6) →
    # server rejects. This would be wrong (mask) if device.list had device:* entry.
    assert expected["device:yoko"] == 5
    # devices:__set__ WAS refreshed (it IS what device.list reveals).
    assert mcp_server._LAST_SEEN["devices:__set__"] == 2


def test_context_md_get_does_not_mask_context_change(wired):
    """Counter-example (b) — context.md_get is NOT in _READ_REVEALS (whole-table).

    If context.md_get were mapped to ``context`` it would advance the context
    baseline to current, masking a concurrent ml-entry edit. After the fix
    context.md_get falls back to whole-table, so a concurrent context bump IS
    absorbed by that whole-table refresh — which means the context baseline is
    current and the guard does NOT fire.

    Wait — whole-table absorbs the bump, so the guard would also not fire for the
    whole-table path. The key distinction is: the INTENDED behaviour for an unmapped
    read is whole-table (accepted, self-block safe), whereas a MAPPED read that is
    too narrow would advance ONLY the revealed keys and leave others stale. What we
    are pinning here is that context.md_get is NOT in _READ_REVEALS (it cannot
    safely claim to reveal the full ``context`` state)."""
    # The crucial assertion: context.md_get must NOT be in _READ_REVEALS.
    # If it were, a projection read would claim to reveal all of context, which is
    # over-broad and would mask concurrent ml/md edits that the read didn't see.
    assert "context.md_get" not in mcp_server._READ_REVEALS
    assert "context.md_get_attr" not in mcp_server._READ_REVEALS
    assert "context.ml_get" not in mcp_server._READ_REVEALS
    assert "context.ml_list_roles" not in mcp_server._READ_REVEALS

    # Verify that context.md_get falls back to whole-table (not narrow) by
    # confirming it behaves like an unmapped read: it replaces all of _LAST_SEEN.
    sent = wired["sent"]
    before = {"context": 7, "tab:t:cfg": 1, "tab:t": 1, "soc": 1}
    wired["context.md_get"] = {"ok": True, "result": {"keys": []}}
    wired["tab.writeback_apply"] = {"ok": True, "result": {}}
    wired["resources.versions"] = _versions_reply(dict(before))

    mcp_server.send_gui_rpc("context.md_get", {})
    # Whole-table refresh: baseline is now the full table (including context 7).
    assert mcp_server._LAST_SEEN["context"] == 7

    # Concurrent ml edit bumps context 7 → 8.
    after = dict(before)
    after["context"] = 8
    wired["resources.versions"] = _versions_reply(after)

    mcp_server.send_gui_rpc("tab.writeback_apply", {"tab_id": "t", "selections": []})
    expected = next(p for (m, p) in reversed(sent) if m == "tab.writeback_apply")[
        "expected_versions"
    ]

    # context.md_get uses whole-table, so it absorbed context v7 into the baseline.
    # After the concurrent bump to v8 the write attaches the stale v7 → server
    # rejects (baseline below current). This confirms that the whole-table fallback
    # for context.md_get does NOT mask the concurrent edit (the guard still fires).
    assert expected["context"] == 7
