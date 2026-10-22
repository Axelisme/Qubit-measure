"""Standalone analysis Save reports and interaction images through MCP."""

import base64
from copy import deepcopy

import pytest
from zcu_tools.mcp.core.reply import PngImage

from ._analyze_support import (
    PNG,
    analysis_client,
    analysis_result_reply,
    analysis_start_reply,
)


@pytest.mark.parametrize("stage", ["primary", "post"])
@pytest.mark.parametrize("has_figure", [False, True])
def test_analysis_saves_whole_artifact_and_queries_only_local_capture(
    tmp_path, clients, stage, has_figure
):
    pane = "analysis" if stage == "primary" else "post_analysis"
    report = {
        "run_id": "run-a",
        "artifacts": {
            pane: {
                "results": {
                    "status": "Partially saved",
                    "members": {
                        f"analysis:{stage}": [
                            {
                                "format": "json",
                                "path": f"/runs/run-a/{pane}/results.json",
                                "status": "saved",
                                "error": None,
                            }
                        ],
                        "figure:fit": [
                            {
                                "format": "png",
                                "path": f"/runs/run-a/{pane}/fit.png",
                                "status": "saved",
                                "error": None,
                            },
                            {
                                "format": "png",
                                "path": "/copies/fit.png",
                                "status": "failed",
                                "error": "copy refused",
                            },
                        ]
                        if has_figure
                        else [],
                    },
                }
            }
        },
    }
    summary = {
        "frequency": 5.0,
        "frequency_stderr": 0.2,
        "fit_quality": {"domain-specific": {"custom_score": None}},
        "warnings": ["domain warning"],
    }
    invalid = [
        {
            "path": "summary.fit_quality.domain-specific.custom_score",
            "reason": "domain_reason",
        }
    ]

    def respond(method, params):
        if method in {"tab.analyze", "tab.post_analyze"}:
            return analysis_start_reply({"skip": 1}, [])
        if method == "operation.await":
            assert params["operation_id"] == 71
            return {"status": "finished", "reason": "completed"}
        if method in {"tab.get_analyze_result", "tab.get_post_analyze_result"}:
            reply = analysis_result_reply(
                pane, ["fit"] if has_figure else [], {"skip": 1}
            )
            reply.update(summary=deepcopy(summary), invalid=deepcopy(invalid))
            return reply
        if method == "tab.save_artifacts":
            assert params == {
                "tab_id": "t",
                "artifacts": [{"section": pane, "name": "results"}],
                "operation_id": 71,
            }
            return deepcopy(report)
        if method == "tab.get_figure":
            assert params == {"tab_id": "t", "subtab_id": pane, "operation_id": 71}
            return {"png_b64": base64.b64encode(PNG).decode()}
        raise AssertionError(method)

    client = analysis_client(tmp_path, clients, respond)
    reply = client.call("tab_analyze", {"tab": "t", "stage": stage})
    assert reply.data["status"] == "finished", reply.data
    assert reply.data["run_id"] == report["run_id"]
    assert reply.data["artifacts"] == report["artifacts"]
    assert reply.data["analysis"][stage]["summary"] == summary
    assert reply.data["analysis"][stage]["invalid"] == invalid
    assert reply.images == ((PngImage(PNG),) if has_figure else ())
    before = list(client.transport.sent)
    for name, arguments in [
        ("status", {"execution": reply.data["execution"]}),
        ("wait", {"execution": reply.data["execution"], "timeout": 0}),
    ]:
        observed = client.call(name, arguments)
        data = observed.data if name == "wait" else observed
        assert data["artifacts"] == report["artifacts"]
        assert data["analysis"][stage]["summary"] == summary
    full = client.call(
        "status", {"execution": reply.data["execution"], "detail": "full"}
    )
    assert (full["tab"], full["stage"], full["op"]) == ("t", stage, reply.data["op"])
    assert full["save_report"] == report
    assert full["result"]["summary"] == summary
    assert full["result"]["invalid"] == invalid
    assert client.transport.sent == before


def test_interaction_delivers_committed_state_and_inline_png(tmp_path, clients):
    state = {"chosen": 2}

    def respond(method, params):
        assert method == "tab.interact"
        assert params == {"tab_id": "t"}
        return {
            "operation_id": 71,
            "plugin": "picker",
            "state": state,
            "info": {"label": "active"},
            "commands": [{"name": "done"}],
            "preview_active": True,
            "figure": {"png_b64": base64.b64encode(PNG).decode()},
        }

    client = analysis_client(tmp_path, clients, respond)
    client.context.session.bind()
    before = len(client.transport.sent)
    reply = client.call("tab_interact", {"tab": "t"})
    assert reply.data["state"] == state
    assert reply.data["commands"] == [{"name": "done"}]
    assert reply.data["preview_active"] is True
    assert reply.images == (PngImage(PNG),)
    assert client.transport.sent[before:] == [("tab.interact", {"tab_id": "t"})]
