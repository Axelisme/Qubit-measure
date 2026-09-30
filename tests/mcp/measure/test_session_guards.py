"""MCP projects GUI stale errors; concurrency behavior belongs to socket tests."""

from pathlib import Path

import pytest

from ._support import make_client


def test_stale_error_identifies_changed_resources_through_the_rpc_boundary(
    tmp_path: Path,
) -> None:
    client = make_client(tmp_path)
    client.transport.replies["tab.run_start"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "stale_version",
            "message": "stale",
            "data": {
                "stale": [
                    "context",
                    "soc",
                    "tab:abc123:cfg",
                    "device:flux",
                    "devices:__set__",
                    "arb_waveforms",
                ]
            },
        },
    }
    with pytest.raises(RuntimeError) as error:
        client.call(
            "rpc_call",
            {
                "method": "tab.run_start",
                "params": {
                    "tab_id": "abc123",
                    "expected": {"cfg_id": "cfg-abc", "revision": "0"},
                },
            },
        )
    message = str(error.value)
    for resource in [
        "the active context (md/ml)",
        "the SoC connection",
        "this tab's cfg",
        "device 'flux'",
        "the set of devices (one added/removed)",
        "the arbitrary waveform asset store",
    ]:
        assert resource in message


def test_tab_run_forwards_the_supplied_ref_once(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    client.context.session.ensure_connected()
    client.transport.sent.clear()
    client.transport.replies["tab.run_start"] = {
        "ok": True,
        "result": {"operation_id": 17},
    }
    expected = {"cfg_id": "cfg-observed", "revision": "3"}

    result = client.call("tab_run", {"tab": "tab-1", "expected": expected})

    assert isinstance(result["op"], int)
    assert client.transport.sent == [
        ("tab.run_start", {"tab_id": "tab-1", "expected": expected})
    ]


def test_tab_run_stale_is_not_retried_or_refreshed(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    client.context.session.ensure_connected()
    client.transport.sent.clear()
    client.transport.replies["tab.run_start"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "stale_revision",
            "message": "stale cfg",
        },
    }
    expected = {"cfg_id": "cfg-observed", "revision": "3"}

    with pytest.raises(RuntimeError):
        client.call("tab_run", {"tab": "tab-1", "expected": expected})

    assert client.transport.sent == [
        ("tab.run_start", {"tab_id": "tab-1", "expected": expected})
    ]
