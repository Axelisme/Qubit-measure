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
            "rpc_call", {"method": "tab.run_start", "params": {"tab_id": "abc123"}}
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
