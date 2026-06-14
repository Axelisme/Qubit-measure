"""Tests for taskboard store pure functions and file I/O layer.

Pure-function tests cover:
  - path normalisation and overlap (file, directory prefix, glob, resource token)
  - conflict matrix (read+read safe; write+anything overlapping = conflict)
  - add_claim grant / pending
  - release_claim promotion (including chain promotion)
  - touch_claim heartbeat
  - reclaim_stale TTL reclaim + cascade promote
  - render_markdown content
  - fast-fail cases (unknown id, empty paths, invalid mode)

File-I/O tests cover:
  - full claim→check→list→release round-trip via TaskboardStore
  - flock atomicity: two sequential RMW calls produce a consistent final state
"""

from __future__ import annotations

import time

import pytest
from zcu_tools.mcp.taskboard.store import (
    TaskboardStore,
    add_claim,
    claims_conflict,
    empty_state,
    normalize_path,
    paths_overlap,
    reclaim_stale,
    release_claim,
    render_markdown,
    touch_claim,
)

# ---------------------------------------------------------------------------
# normalize_path
# ---------------------------------------------------------------------------


def test_normalize_strips_leading_slash():
    assert normalize_path("/lib/foo") == "lib/foo"


def test_normalize_strips_trailing_slash():
    assert normalize_path("lib/foo/") == "lib/foo"


def test_normalize_resource_token_lowercase():
    assert normalize_path("@HW/ZCU216") == "@hw/zcu216"


def test_normalize_empty_raises():
    with pytest.raises(ValueError, match="empty path"):
        normalize_path("")


def test_normalize_bare_at_raises():
    with pytest.raises(ValueError, match="invalid resource token"):
        normalize_path("@")


# ---------------------------------------------------------------------------
# paths_overlap
# ---------------------------------------------------------------------------


def test_exact_match_overlaps():
    assert paths_overlap("lib/foo.py", "lib/foo.py")


def test_ancestor_dir_overlaps():
    assert paths_overlap("lib/zcu_tools", "lib/zcu_tools/mcp/server.py")


def test_child_to_ancestor_overlaps():
    assert paths_overlap("lib/zcu_tools/mcp/server.py", "lib/zcu_tools")


def test_sibling_dirs_no_overlap():
    assert not paths_overlap("lib/zcu_tools/mcp", "lib/zcu_tools/gui")


def test_glob_star_overlaps():
    assert paths_overlap("lib/zcu_tools/*.py", "lib/zcu_tools/server.py")


def test_glob_doublestar_overlaps():
    assert paths_overlap("lib/zcu_tools/**/*.py", "lib/zcu_tools/mcp/server.py")


def test_resource_token_exact():
    assert paths_overlap("@gui/measure", "@gui/measure")


def test_resource_token_ancestor():
    assert paths_overlap("@gui", "@gui/measure")


def test_resource_token_no_overlap():
    assert not paths_overlap("@gui/measure", "@gui/fluxdep")


# ---------------------------------------------------------------------------
# claims_conflict
# ---------------------------------------------------------------------------


def _make_claim(paths, mode):
    return {"paths": paths, "mode": mode, "status": "granted", "claim_id": "x"}


def test_read_read_no_conflict():
    c1 = _make_claim(["lib/foo"], "read")
    c2 = _make_claim(["lib/foo"], "read")
    assert not claims_conflict(c1, c2)


def test_write_write_conflict():
    c1 = _make_claim(["lib/foo"], "write")
    c2 = _make_claim(["lib/foo"], "write")
    assert claims_conflict(c1, c2)


def test_read_write_conflict():
    c1 = _make_claim(["lib/foo"], "read")
    c2 = _make_claim(["lib/foo"], "write")
    assert claims_conflict(c1, c2)


def test_write_read_conflict():
    c1 = _make_claim(["lib/foo"], "write")
    c2 = _make_claim(["lib/foo"], "read")
    assert claims_conflict(c1, c2)


def test_non_overlapping_paths_no_conflict():
    c1 = _make_claim(["lib/a"], "write")
    c2 = _make_claim(["lib/b"], "write")
    assert not claims_conflict(c1, c2)


def test_token_write_write_conflict():
    c1 = _make_claim(["@gui/measure"], "write")
    c2 = _make_claim(["@gui/measure"], "write")
    assert claims_conflict(c1, c2)


# ---------------------------------------------------------------------------
# add_claim + compute_conflicts
# ---------------------------------------------------------------------------


def test_add_claim_granted_when_no_conflicts():
    state = empty_state()
    new_state, claim = add_claim(state, "alice", ["lib/foo"], "task A", "write")
    assert claim["status"] == "granted"
    assert len(new_state["claims"]) == 1


def test_add_claim_pending_when_conflict():
    state = empty_state()
    state, c1 = add_claim(state, "alice", ["lib/foo"], "task A", "write")
    state, c2 = add_claim(state, "bob", ["lib/foo"], "task B", "write")
    assert c2["status"] == "pending"
    assert c1["claim_id"] in c2["blockers"]


def test_read_read_both_granted():
    state = empty_state()
    state, c1 = add_claim(state, "alice", ["lib/foo"], "read A", "read")
    state, c2 = add_claim(state, "bob", ["lib/foo"], "read B", "read")
    assert c1["status"] == "granted"
    assert c2["status"] == "granted"


def test_add_claim_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        add_claim(empty_state(), "alice", ["lib/foo"], "t", "bogus")


def test_add_claim_empty_paths_raises():
    with pytest.raises(ValueError, match="paths must be non-empty"):
        add_claim(empty_state(), "alice", [], "t", "write")


# ---------------------------------------------------------------------------
# release_claim + promotion
# ---------------------------------------------------------------------------


def test_release_promotes_pending():
    state = empty_state()
    state, c1 = add_claim(state, "alice", ["lib/foo"], "A", "write")
    state, c2 = add_claim(state, "bob", ["lib/foo"], "B", "write")
    assert c2["status"] == "pending"

    state, promoted = release_claim(state, c1["claim_id"])
    assert any(p["claim_id"] == c2["claim_id"] for p in promoted)
    granted_ids = {c["claim_id"] for c in state["claims"] if c["status"] == "granted"}
    assert c2["claim_id"] in granted_ids


def test_release_chain_promotion():
    """Three sequential claims on same path — release first promotes second, which
    then enables third to be checked (but third is blocked by second, so only one
    promotion per release call)."""
    state = empty_state()
    state, c1 = add_claim(state, "a", ["lib/foo"], "A", "write")
    state, c2 = add_claim(state, "b", ["lib/foo"], "B", "write")
    state, c3 = add_claim(state, "c", ["lib/foo"], "C", "write")
    assert c2["status"] == "pending"
    assert c3["status"] == "pending"

    # Release c1 → c2 promoted, c3 still pending (blocked by c2)
    state, promoted = release_claim(state, c1["claim_id"])
    assert len(promoted) == 1
    assert promoted[0]["claim_id"] == c2["claim_id"]

    # Release c2 → c3 promoted
    state, promoted2 = release_claim(state, c2["claim_id"])
    assert len(promoted2) == 1
    assert promoted2[0]["claim_id"] == c3["claim_id"]


def test_release_unknown_id_raises():
    with pytest.raises(ValueError, match="unknown claim_id"):
        release_claim(empty_state(), "deadbeef")


def test_release_moves_to_history():
    state = empty_state()
    state, c1 = add_claim(state, "alice", ["lib/foo"], "A", "write")
    state, _ = release_claim(state, c1["claim_id"])
    assert any(c["claim_id"] == c1["claim_id"] for c in state["released"])
    assert not any(c["claim_id"] == c1["claim_id"] for c in state["claims"])


# ---------------------------------------------------------------------------
# touch_claim
# ---------------------------------------------------------------------------


def test_touch_updates_timestamp():
    state = empty_state()
    state, c1 = add_claim(state, "alice", ["lib/x"], "A", "write", now=1000.0)
    state, touched = touch_claim(state, c1["claim_id"])
    assert touched["touched"] >= 1000.0


def test_touch_unknown_raises():
    with pytest.raises(ValueError, match="unknown claim_id"):
        touch_claim(empty_state(), "deadbeef")


# ---------------------------------------------------------------------------
# reclaim_stale
# ---------------------------------------------------------------------------


def test_reclaim_stale_removes_old_claim():
    state = empty_state()
    state, c1 = add_claim(state, "alice", ["lib/x"], "A", "write", now=0.0)
    state, stale = reclaim_stale(state, now=9999.0, ttl=3600.0)
    assert any(s["claim_id"] == c1["claim_id"] for s in stale)
    assert not any(c["claim_id"] == c1["claim_id"] for c in state["claims"])


def test_reclaim_stale_promotes_pending():
    # c1 is stale (touched at t=0, now=9999, ttl=3600 → expired).
    # c2 touched at t=8000 (still within TTL=3600 of now=9999 → fresh, not stale).
    # After c1 is reclaimed, c2 should be promoted to granted.
    state = empty_state()
    state, c1 = add_claim(state, "alice", ["lib/x"], "A", "write", now=0.0)
    state, c2 = add_claim(state, "bob", ["lib/x"], "B", "write", now=8000.0)
    state, stale = reclaim_stale(state, now=9999.0, ttl=3600.0)
    assert any(s["claim_id"] == c1["claim_id"] for s in stale)
    promoted = [c for c in state["claims"] if c["status"] == "granted"]
    assert any(p["claim_id"] == c2["claim_id"] for p in promoted)


def test_reclaim_stale_fresh_claim_kept():
    state = empty_state()
    now = time.time()
    state, c1 = add_claim(state, "alice", ["lib/x"], "A", "write", now=now)
    state, stale = reclaim_stale(state, now=now + 60.0, ttl=3600.0)
    assert not stale
    assert any(c["claim_id"] == c1["claim_id"] for c in state["claims"])


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------


def test_render_markdown_has_required_sections():
    state = empty_state()
    state, c1 = add_claim(state, "alice", ["lib/foo"], "task A", "write")
    md = render_markdown(state)
    assert "## Active claims" in md
    assert "## Pending queue" in md
    assert "## Recent released" in md
    assert "alice" in md
    assert "task A" in md


def test_render_markdown_empty_state():
    md = render_markdown(empty_state())
    assert "_No active claims._" in md
    assert "_No pending claims._" in md
    assert "_No recent history._" in md


# ---------------------------------------------------------------------------
# TaskboardStore file I/O integration
# ---------------------------------------------------------------------------


def test_store_claim_and_release(tmp_path):
    store = TaskboardStore(
        json_path=tmp_path / "taskboard.json",
        md_path=tmp_path / "taskboard.md",
    )
    result = store.claim("alice", ["lib/foo"], "task A", mode="write")
    assert result["status"] == "granted"
    claim_id = result["claim_id"]

    # check — zero side effects
    chk = store.check(["lib/foo"], mode="write")
    assert len(chk["conflicts"]) == 1
    assert chk["conflicts"][0]["owner"] == "alice"

    # list
    lst = store.list_claims()
    assert any(c["claim_id"] == claim_id for c in lst["active"])

    # release
    rel = store.release(claim_id)
    assert rel["released_id"] == claim_id

    # after release: no conflicts
    chk2 = store.check(["lib/foo"], mode="write")
    assert chk2["conflicts"] == []

    # markdown was written (file is UTF-8; read it as such, not via locale codec)
    md_content = (tmp_path / "taskboard.md").read_text(encoding="utf-8")
    assert "Taskboard" in md_content


def test_store_pending_promoted_on_release(tmp_path):
    store = TaskboardStore(
        json_path=tmp_path / "taskboard.json",
        md_path=tmp_path / "taskboard.md",
    )
    r1 = store.claim("alice", ["lib/foo"], "A")
    r2 = store.claim("bob", ["lib/foo"], "B")
    assert r1["status"] == "granted"
    assert r2["status"] == "pending"

    rel = store.release(r1["claim_id"])
    assert any(p["claim_id"] == r2["claim_id"] for p in rel["promoted"])

    lst = store.list_claims()
    assert any(c["claim_id"] == r2["claim_id"] for c in lst["active"])


def test_store_wait_grants(tmp_path):
    store = TaskboardStore(
        json_path=tmp_path / "taskboard.json",
        md_path=tmp_path / "taskboard.md",
    )
    r1 = store.claim("alice", ["lib/foo"], "A")
    assert r1["status"] == "granted"
    # Waiting on a granted claim returns immediately.
    w = store.wait(r1["claim_id"], timeout_s=1.0)
    assert w["status"] == "granted"


def test_store_wait_timeout(tmp_path):
    store = TaskboardStore(
        json_path=tmp_path / "taskboard.json",
        md_path=tmp_path / "taskboard.md",
    )
    store.claim("alice", ["lib/foo"], "A")
    r2 = store.claim("bob", ["lib/foo"], "B")
    assert r2["status"] == "pending"
    # Wait with very short timeout — should time out.
    w = store.wait(r2["claim_id"], timeout_s=0.6)
    assert w["status"] == "timeout"


def test_store_wait_unknown_id_raises(tmp_path):
    store = TaskboardStore(
        json_path=tmp_path / "taskboard.json",
        md_path=tmp_path / "taskboard.md",
    )
    with pytest.raises(ValueError, match="unknown claim_id"):
        store.wait("deadbeef", timeout_s=1.0)


def test_store_touch(tmp_path):
    store = TaskboardStore(
        json_path=tmp_path / "taskboard.json",
        md_path=tmp_path / "taskboard.md",
    )
    r = store.claim("alice", ["lib/foo"], "A")
    t = store.touch(r["claim_id"])
    assert t["claim_id"] == r["claim_id"]
    assert "touched" in t


def test_store_force_release(tmp_path):
    store = TaskboardStore(
        json_path=tmp_path / "taskboard.json",
        md_path=tmp_path / "taskboard.md",
    )
    r = store.claim("alice", ["lib/foo"], "A")
    fr = store.force_release(r["claim_id"])
    assert fr["released_id"] == r["claim_id"]


def test_store_check_read_read_no_conflict(tmp_path):
    store = TaskboardStore(
        json_path=tmp_path / "taskboard.json",
        md_path=tmp_path / "taskboard.md",
    )
    store.claim("alice", ["lib/foo"], "read A", mode="read")
    chk = store.check(["lib/foo"], mode="read")
    assert chk["conflicts"] == []


def test_store_sequential_rmw_consistent(tmp_path):
    """Two sequential claim calls produce two distinct granted claims with no overlap."""
    store = TaskboardStore(
        json_path=tmp_path / "taskboard.json",
        md_path=tmp_path / "taskboard.md",
    )
    r1 = store.claim("alice", ["lib/a"], "A")
    r2 = store.claim("bob", ["lib/b"], "B")
    assert r1["status"] == "granted"
    assert r2["status"] == "granted"
    assert r1["claim_id"] != r2["claim_id"]
    lst = store.list_claims()
    active_ids = {c["claim_id"] for c in lst["active"]}
    assert {r1["claim_id"], r2["claim_id"]} == active_ids


# ---------------------------------------------------------------------------
# F. Dotfile normalisation (A-fix)
# ---------------------------------------------------------------------------


def test_normalize_dotfile_preserved():
    """'.github/x' must keep the leading dot — lstrip charset would have eaten it."""
    assert normalize_path(".github/x") == ".github/x"


def test_normalize_dotfile_no_overlap_with_plain():
    """.github/x and github/x are distinct paths — dotfile lock must not leak."""
    assert not paths_overlap(".github/x", "github/x")


def test_normalize_dotfile_same_overlaps():
    """.claude/skills/a and itself must overlap (self-lock)."""
    assert paths_overlap(".claude/skills/a", ".claude/skills/a")


def test_normalize_dot_slash_stripped():
    """./lib/x should normalise to lib/x."""
    assert normalize_path("./lib/x") == "lib/x"


def test_normalize_dotdot_folded():
    """a/../b normalises to b, so it must overlap with b."""
    normed = normalize_path("a/../b")
    assert normed == "b"
    assert paths_overlap(normed, "b")


def test_normalize_dotdot_overlap_with_b():
    """paths_overlap('a/../b', 'b') after normalisation — caller normalises before storing."""
    # The store normalises paths before storing, so test via add_claim round-trip.
    state = empty_state()
    # Manually normalise as the store would.
    state, c = add_claim(state, "alice", [normalize_path("a/../b")], "T", "write")
    assert c["paths"][0] == "b"


# ---------------------------------------------------------------------------
# F. Atomic write + corrupt-file fast-fail (B-fix)
# ---------------------------------------------------------------------------


def test_store_write_readable_roundtrip(tmp_path):
    """State written by _write_state must be readable back without corruption."""
    store = TaskboardStore(
        json_path=tmp_path / "taskboard.json",
        md_path=tmp_path / "taskboard.md",
    )
    store.claim("alice", ["lib/foo"], "A")
    # Re-read via a second store instance pointing at the same file.
    store2 = TaskboardStore(
        json_path=tmp_path / "taskboard.json",
        md_path=tmp_path / "taskboard.md",
    )
    lst = store2.list_claims()
    assert len(lst["active"]) == 1
    assert lst["active"][0]["owner"] == "alice"


def test_store_corrupt_json_raises(tmp_path):
    """An existing but invalid JSON file must raise, not silently return empty state."""
    json_path = tmp_path / "taskboard.json"
    json_path.write_text("{ this is not json }", encoding="utf-8")
    store = TaskboardStore(
        json_path=json_path,
        md_path=tmp_path / "taskboard.md",
    )
    # Any operation that reads state should propagate the corruption error.
    with pytest.raises(
        (RuntimeError, Exception), match="corrupt|invalid JSON|Expecting"
    ):
        store.list_claims()


# ---------------------------------------------------------------------------
# F. Mode validation at all public entry points (C-fix)
# ---------------------------------------------------------------------------


def test_claim_invalid_mode_raises(tmp_path):
    store = TaskboardStore(
        json_path=tmp_path / "taskboard.json",
        md_path=tmp_path / "taskboard.md",
    )
    with pytest.raises(ValueError, match="mode"):
        store.claim("alice", ["lib/foo"], "T", mode="bogus")


def test_check_invalid_mode_raises(tmp_path):
    store = TaskboardStore(
        json_path=tmp_path / "taskboard.json",
        md_path=tmp_path / "taskboard.md",
    )
    with pytest.raises(ValueError, match="mode"):
        store.check(["lib/foo"], mode="bogus")


def test_check_valid_mode_no_side_effects(tmp_path):
    """check with a valid mode must not mutate state (zero side effects)."""
    store = TaskboardStore(
        json_path=tmp_path / "taskboard.json",
        md_path=tmp_path / "taskboard.md",
    )
    store.claim("alice", ["lib/foo"], "A", mode="write")
    before = store.list_claims()

    # check with mode="read" — valid, no side effects.
    store.check(["lib/foo"], mode="read")
    after = store.list_claims()
    assert before["active"] == after["active"]
    assert before["pending"] == after["pending"]


# ---------------------------------------------------------------------------
# F. Glob vs literal paths (D-fix)
# ---------------------------------------------------------------------------


def test_literal_bracket_no_false_positive():
    """'file[1].py' is a literal filename, must NOT overlap with 'file1.py'."""
    assert not paths_overlap("file[1].py", "file1.py")


def test_glob_doublestar_overlaps_nested():
    """'lib/**' is a glob and must overlap with 'lib/a/b.py'."""
    assert paths_overlap("lib/**", "lib/a/b.py")


def test_resource_token_exact_overlaps():
    """Two identical resource tokens overlap."""
    assert paths_overlap("@hw/zcu216", "@hw/zcu216")


def test_resource_token_different_no_overlap():
    """Different resource tokens must not overlap."""
    assert not paths_overlap("@hw/zcu216", "@hw/other")


def test_resource_token_parent_child_overlaps():
    """@hw overlaps @hw/zcu216 (ancestor rule)."""
    assert paths_overlap("@hw", "@hw/zcu216")


# ---------------------------------------------------------------------------
# F. Read lock: multiple readers all granted; after writer release all promote
# ---------------------------------------------------------------------------


def test_multiple_readers_all_granted():
    """Read claims on the same path must all be granted simultaneously."""
    state = empty_state()
    state, c1 = add_claim(state, "alice", ["lib/foo"], "R1", "read")
    state, c2 = add_claim(state, "bob", ["lib/foo"], "R2", "read")
    state, c3 = add_claim(state, "carol", ["lib/foo"], "R3", "read")
    assert c1["status"] == "granted"
    assert c2["status"] == "granted"
    assert c3["status"] == "granted"


def test_writer_release_promotes_multiple_pending_readers():
    """After a writer releases, all pending readers should be promoted together."""
    state = empty_state()
    # Writer holds the lock.
    state, writer = add_claim(state, "alice", ["lib/foo"], "W", "write")
    assert writer["status"] == "granted"
    # Two readers queue up.
    state, r1 = add_claim(state, "bob", ["lib/foo"], "R1", "read")
    state, r2 = add_claim(state, "carol", ["lib/foo"], "R2", "read")
    assert r1["status"] == "pending"
    assert r2["status"] == "pending"

    # Release the writer — both readers should be promoted in one pass.
    state, promoted = release_claim(state, writer["claim_id"])
    promoted_ids = {p["claim_id"] for p in promoted}
    assert r1["claim_id"] in promoted_ids
    assert r2["claim_id"] in promoted_ids

    # Confirm both are granted in the final state.
    granted_ids = {c["claim_id"] for c in state["claims"] if c["status"] == "granted"}
    assert r1["claim_id"] in granted_ids
    assert r2["claim_id"] in granted_ids
