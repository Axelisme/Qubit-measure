"""Export and Restore expose sampled checkpoints, not per-point payloads."""

import json
from io import StringIO

import pytest
from zcu_tools.mcp.core.stdio_server import run_stdio_loop

from tests.mcp.fluxdep._support import Client, GuiResponse


@pytest.mark.parametrize("restore", [False, True])
def test_checkpoint_tools_return_native_path_and_spectrum_summaries(
    client: Client, restore: bool
):
    rows = [
        {
            "name": "sampled",
            "spec_type": "TwoTone",
            "aligned": True,
            "points_completed": True,
            "point_count": 3,
            "pending": None,
        },
        {
            "name": "empty",
            "spec_type": "OneTone",
            "aligned": True,
            "points_completed": True,
            "point_count": 0,
            "pending": None,
        },
    ]
    receipt = (
        {"names": ["sampled", "empty"]}
        if restore
        else {"filepath": "/resolved/checkpoint.hdf5"}
    )
    client.transport.replies.extend(
        [GuiResponse(result=receipt), GuiResponse(result={"spectrums": rows})]
    )
    tool = "fluxdep_spectrum_load_processed" if restore else "fluxdep_export_spectrums"
    args = (
        {"filepath": "/checkpoint.hdf5"}
        if restore
        else {"filepath": "/checkpoint.hdf5", "overwrite": True}
    )
    output = StringIO()
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": args},
    }
    run_stdio_loop(
        client.server.bridge.config,
        client.server.tools,
        input_stream=StringIO(json.dumps(request) + "\n"),
        output_stream=output,
    )
    response = json.loads(output.getvalue())
    assert "result" in response, response
    assert not response["result"].get("isError", False), response
    summary = json.loads(response["result"]["content"][0]["text"])
    if restore:
        assert summary == {
            "spectrums": [
                {
                    "name": "sampled",
                    "type": "TwoTone",
                    "stage": "sampled",
                    "point_count": 3,
                    "pending": None,
                },
                {
                    "name": "empty",
                    "type": "OneTone",
                    "stage": "sampled",
                    "point_count": 0,
                    "pending": None,
                },
            ]
        }
    else:
        assert summary == {
            "filepath": "/resolved/checkpoint.hdf5",
            "spectrums": [
                {"name": "sampled", "stage": "sampled", "point_count": 3},
                {"name": "empty", "stage": "sampled", "point_count": 0},
            ],
            "note": "filter mask not saved",
        }
    assert [
        (request["method"], request["params"]) for request in client.transport.requests
    ] == [
        ("spectrum.load_processed" if restore else "export.spectrums", args),
        ("spectrum.list", {}),
    ]
