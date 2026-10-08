"""Native Save reports and inline images through public MCP/author seams."""

import base64
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict

import pytest
from zcu_tools.mcp.core.reply import PngImage
from zcu_tools.mcp.measure.recipe import RecipeGenerator, RecipeRun, RecipeSession
from zcu_tools.mcp.measure.session import GuiRpcError

from ._analyze_support import (
    PNG,
    analysis_client,
    analysis_result_reply,
    analysis_start_reply,
)
from ._recipe_execution_support import definition
from ._recipe_support import LookbackGui, recipe_client
from ._support import make_client


@pytest.mark.parametrize(
    "states", [("saved", "saved"), ("saved", "failed"), ("failed", "failed")]
)
def test_raw_save_captures_every_output_from_the_synchronous_owner(tmp_path, states):
    gui = LookbackGui()
    labber = {
        "format": "labber",
        "path": "/labber/first.hdf5",
        "status": states[1],
        "error": "Labber write failed" if states[1] == "failed" else None,
    }
    report = {
        "run_id": "run-a" if "saved" in states else None,
        "artifacts": {
            "raw": {
                "data": {
                    "status": "Saved"
                    if states == ("saved", "saved")
                    else "Partially saved"
                    if "saved" in states
                    else "Not saved",
                    "members": {
                        "data": [
                            {
                                "format": "data_h5",
                                "path": "/runs/run-a/data.h5",
                                "status": states[0],
                                "error": "h5 write failed"
                                if states[0] == "failed"
                                else None,
                            },
                            labber,
                        ]
                    },
                }
            }
        },
    }

    def respond(method, params):
        if method == "tab.save_data":
            assert params == {"tab_id": "t", "run_operation_id": 71}
            return deepcopy(report)
        return gui(method, params)

    with recipe_client(tmp_path, respond) as client:
        operation = RecipeSession(client.context).open_tab("lookback").run()
        run, status = operation.complete_in_current_worker()
        assert isinstance(run, RecipeRun) and status == "completed"
        before = len(client.transport.sent)
        saved = run.save_raw()
        assert asdict(saved) == report
        assert run.snapshot().raw_save == saved
        assert client.transport.sent[before:] == [
            ("tab.save_data", {"tab_id": "t", "run_operation_id": 71})
        ]

        labber["path"] = "/labber/second.hdf5"
        repeated = run.save_raw()
        assert asdict(repeated) == report
        assert run.snapshot().raw_save == repeated
        assert (
            asdict(saved)["artifacts"]["raw"]["data"]["members"]["data"][1]["path"]
            == "/labber/first.hdf5"
        )


def test_raw_save_rejection_clears_the_admitted_attempt_and_propagates(tmp_path):
    gui = LookbackGui()

    with recipe_client(tmp_path, gui) as client:
        client.transport.replies["tab.save_data"] = {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": "run_source_superseded",
                "message": "source replaced",
            },
        }
        operation = RecipeSession(client.context).open_tab("lookback").run()
        run, _ = operation.complete_in_current_worker()
        assert isinstance(run, RecipeRun)
        with pytest.raises(GuiRpcError, match="source replaced"):
            run.save_raw()
        assert run.snapshot().raw_save is None


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
    assert full["save_report"] == report
    assert full["result"]["summary"] == summary
    assert full["result"]["invalid"] == invalid
    assert client.transport.sent == before


def test_recipe_reply_projects_raw_report_and_delivers_run_bytes(tmp_path):
    gui = LookbackGui()
    report = {
        "run_id": "run-a",
        "artifacts": {
            "raw": {
                "data": {
                    "status": "Saved",
                    "members": {
                        "data": [
                            {
                                "format": "data_h5",
                                "path": "/runs/run-a/data.h5",
                                "status": "saved",
                                "error": None,
                            }
                        ]
                    },
                }
            }
        },
    }

    def sample(session: RecipeSession) -> RecipeGenerator:
        run, status = yield session.open_tab("lookback").run()
        assert isinstance(run, RecipeRun) and status == "completed"
        run.save_raw()
        run.preview()

    def respond(method, params):
        if method == "tab.save_data":
            gui.raw_saved = True
            return deepcopy(report)
        if method == "tab.get_figure":
            assert params == {"tab_id": "t", "subtab_id": "run", "run_operation_id": 71}
            return {"png_b64": base64.b64encode(PNG).decode()}
        return gui(method, params)

    client = make_client(tmp_path, respond, recipes=(definition(sample),))
    with closing(client.context.session):
        reply = client.call("sample", {})
        assert reply.data["status"] == "finished", reply.data
        assert reply.data["run_id"] == "run-a"
        assert reply.data["artifacts"] == report["artifacts"]
        assert reply.images == (PngImage(PNG),)
        before = list(client.transport.sent)
        full = client.call(
            "status", {"execution": reply.data["execution"], "detail": "full"}
        )
        assert full["raw_save"] == report
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


@pytest.mark.parametrize(
    "report",
    [
        {"run_id": "run-a"},
        {
            "run_id": "run-a",
            "artifacts": {"raw": {"data": {"status": "invalid", "members": {}}}},
        },
    ],
)
def test_raw_save_rejects_malformed_native_report_at_the_wire_boundary(
    tmp_path, report
):
    with recipe_client(tmp_path, LookbackGui()) as client:
        run, _ = (
            RecipeSession(client.context)
            .open_tab("lookback")
            .run()
            .complete_in_current_worker()
        )
        assert isinstance(run, RecipeRun)
        client.transport.replies["tab.save_data"] = {"ok": True, "result": report}
        with pytest.raises(GuiRpcError) as error:
            run.save_raw()
        assert error.value.reason == "incompatible_wire"
        assert run.snapshot().raw_save is None
