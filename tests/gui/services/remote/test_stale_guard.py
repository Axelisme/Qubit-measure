"""Optimistic-concurrency version guard.

A guarded op (run / save / editor.commit) carries an ``expected_versions`` map
(filled by the mcp layer) of the resource versions it depends on. The server
compares them atomically against the current VersionTable; any mismatch — a
dependency moved since the caller read it, or the resource was dropped (now 0) —
raises PRECONDITION_FAILED. Absent/empty means no check.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.services.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.main.services.remote.errors import ErrorCode, RemoteError


@pytest.fixture(autouse=True)
def _qt(qapp):  # noqa: ARG001
    yield


def _service(versions=None):
    """A service whose Controller reports the given resource version table."""
    ctrl = MagicMock()
    ctrl.get_bus.return_value = None
    ctrl.resources_versions.return_value = dict(versions or {})
    return RemoteControlAdapter(controller=ctrl, opts=ControlOptions(port=0))


# ---------------------------------------------------------------------------
# No expected_versions -> never blocks (plain RPC behaviour)
# ---------------------------------------------------------------------------


def test_no_expected_versions_passes():
    svc = _service({"tab:t:cfg": 3})
    svc._guard_versions({"tab_id": "t"})  # no raise
    svc._guard_versions({"tab_id": "t", "expected_versions": {}})  # no raise


# ---------------------------------------------------------------------------
# Matching versions pass; mismatches block
# ---------------------------------------------------------------------------


def test_matching_versions_pass():
    svc = _service({"tab:t:cfg": 3, "soc": 1, "context": 2})
    svc._guard_versions(
        {"expected_versions": {"tab:t:cfg": 3, "soc": 1, "context": 2}}
    )  # no raise


def test_stale_dependency_blocks():
    svc = _service({"tab:t:cfg": 4})  # current is 4
    with pytest.raises(RemoteError) as ei:
        svc._guard_versions({"expected_versions": {"tab:t:cfg": 3}})  # saw 3
    assert ei.value.code == ErrorCode.PRECONDITION_FAILED
    assert ei.value.reason == "stale_version"
    # The error names the resource identities that moved (no version numbers),
    # so mcp can translate them into agent language (Phase 120c-3).
    assert ei.value.data == {"stale": ["tab:t:cfg"]}


def test_stale_data_lists_only_mismatched_keys():
    svc = _service({"tab:t:cfg": 4, "soc": 1, "context": 2})
    with pytest.raises(RemoteError) as ei:
        svc._guard_versions(
            {"expected_versions": {"tab:t:cfg": 3, "soc": 1, "context": 9}}
        )
    # Only the moved keys (cfg, context) appear — soc matched, so it is omitted.
    assert ei.value.data == {"stale": ["context", "tab:t:cfg"]}


def test_one_mismatch_among_matches_blocks():
    svc = _service({"tab:t:cfg": 3, "soc": 2, "context": 2})
    with pytest.raises(RemoteError):
        # soc moved 1 -> 2; the rest match, but one mismatch is enough.
        svc._guard_versions(
            {"expected_versions": {"tab:t:cfg": 3, "soc": 1, "context": 2}}
        )


def test_dropped_dependency_reads_zero_and_blocks():
    # A closed tab's key is gone from the table -> current reads 0. A caller that
    # depended on a non-zero version is stale (the resource it relied on is gone).
    svc = _service({})  # tab:t:cfg dropped
    with pytest.raises(RemoteError) as ei:
        svc._guard_versions({"expected_versions": {"tab:t:cfg": 2}})
    assert ei.value.reason == "stale_version"


def test_expected_zero_on_absent_key_passes():
    # Depending on version 0 of an absent key matches (never-bumped == 0).
    svc = _service({})
    svc._guard_versions({"expected_versions": {"soc": 0}})  # no raise


# ---------------------------------------------------------------------------
# Bad input
# ---------------------------------------------------------------------------


def test_non_dict_expected_versions_is_invalid_params():
    svc = _service({})
    with pytest.raises(RemoteError) as ei:
        svc._guard_versions({"expected_versions": ["not", "a", "dict"]})
    assert ei.value.code == ErrorCode.INVALID_PARAMS
