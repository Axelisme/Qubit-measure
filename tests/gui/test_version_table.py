"""Unit tests for zcu_tools.gui.app.main.state.VersionTable."""

from __future__ import annotations

from zcu_tools.gui.app.main.state import VersionTable


def test_unbumped_key_reads_zero():
    vt = VersionTable()
    assert vt.get("context") == 0


def test_bump_is_monotonic_and_returns_new_value():
    vt = VersionTable()
    assert vt.bump("soc") == 1
    assert vt.bump("soc") == 2
    assert vt.get("soc") == 2


def test_bumps_are_independent_per_key():
    vt = VersionTable()
    vt.bump("context")
    vt.bump("context")
    vt.bump("device:yoko")
    assert vt.get("context") == 2
    assert vt.get("device:yoko") == 1


def test_snapshot_is_a_copy():
    vt = VersionTable()
    vt.bump("soc")
    snap = vt.snapshot()
    assert snap == {"soc": 1}
    snap["soc"] = 999  # mutating the copy must not affect the table
    assert vt.get("soc") == 1


def test_drop_prefix_forgets_matching_keys_only():
    vt = VersionTable()
    vt.bump("tab:abc")
    vt.bump("tab:abc:cfg")
    vt.bump("tab:abc:result")
    vt.bump("tab:xyz")
    vt.bump("context")

    vt.drop_prefix("tab:abc")

    # Dropped keys read as 0 (gone -> guard treats as stale).
    assert vt.get("tab:abc") == 0
    assert vt.get("tab:abc:cfg") == 0
    assert vt.get("tab:abc:result") == 0
    # A different tab and global resources are untouched.
    assert vt.get("tab:xyz") == 1
    assert vt.get("context") == 1


def test_bump_after_drop_starts_from_zero_again():
    vt = VersionTable()
    vt.bump("tab:abc:cfg")
    vt.bump("tab:abc:cfg")
    vt.drop_prefix("tab:abc")
    # A re-created resource (same key) restarts at 1. tab_ids are uuid4 so this
    # never actually collides in practice, but the table makes no assumption.
    assert vt.bump("tab:abc:cfg") == 1
