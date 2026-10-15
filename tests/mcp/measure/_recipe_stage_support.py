"""Recording primary/post replies for same-Run stage contracts."""

import base64
from copy import deepcopy
from threading import Event
from typing import Any

from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.measure.raw_save import SaveArtifact, SaveOutput, SaveReport

from ._recipe_support import PNG, ExecutionGui
from ._recipe_sync_support import start_recipe_at
from ._support import MeasureClient

RAW_REPORT = SaveReport(
    "run-a",
    {
        "raw": {
            "data": SaveArtifact(
                "Saved",
                {
                    "data": [
                        SaveOutput("data_h5", "/runs/run-a/data.h5", "saved"),
                        SaveOutput("labber", "/labber/raw.hdf5", "saved"),
                    ]
                },
            )
        }
    },
)
PRIMARY_REPORT = SaveReport(
    "run-a",
    {
        "analysis": {
            "results": SaveArtifact(
                "Saved",
                {
                    "analysis:primary": [
                        SaveOutput("json", "/runs/run-a/analysis/results.json", "saved")
                    ],
                    "figure:trace": [
                        SaveOutput("png", "/runs/run-a/analysis/trace.png", "saved")
                    ],
                },
            )
        }
    },
)
POST_REPORT = SaveReport(
    "run-a",
    {
        "post_analysis": {
            "results": SaveArtifact(
                "Saved",
                {
                    "analysis:post": [
                        SaveOutput(
                            "json", "/runs/run-a/post_analysis/results.json", "saved"
                        )
                    ],
                    "figure:cloud": [
                        SaveOutput(
                            "png", "/runs/run-a/post_analysis/cloud.png", "saved"
                        )
                    ],
                },
            )
        }
    },
)


def start_pipeline(client: MeasureClient, arrived: Event) -> ToolReply:
    """Admit the synthetic stages and poll only after the fake's target RPC arrives."""
    return start_recipe_at(client, arrived, {"frequency_mhz": 5000.0, "post": True})


def skip_writeback(client: MeasureClient, question: ToolReply) -> ToolReply:
    """Observe both captured stages and answer without draft writes."""
    assert question.data["status"] == "awaiting_answer", question.data
    assert tuple(question.data["question_items"]) == ("predict_offset", "classifier")
    before = list(client.transport.sent)
    reply = client.call(
        "answer", {"recipe": question.data["execution"], "decision": "skipped"}
    )
    assert reply.data["status"] == "finished", reply.data
    assert client.transport.sent == before
    return reply


class StageGui(ExecutionGui):
    def __init__(self, *, interactive=False):
        super().__init__()
        self.interactive = interactive
        self.current_stage = "primary"
        self.done = {"primary": Event(), "post": Event()}
        self.writes: list[dict[str, object]] = []
        self.calls = []

    def __call__(self, method, params) -> dict[str, Any]:
        self.calls.append((method, deepcopy(params)))
        if method == "tab.interact":
            if params.get("payload", {}).get("command") == "done":
                self.done[self.current_stage].set()
            return {
                "operation_id": 93 if self.current_stage == "primary" else 104,
                "plugin": "stage-picker",
                "state": {"stage": self.current_stage},
                "info": {},
                "commands": [{"name": "done"}],
                "preview_active": True,
                "figure": {"png_b64": base64.b64encode(PNG).decode()}
                if params.get("include_figure", True)
                else None,
            }
        if method == "operation.await" and params["operation_id"] in (93, 104):
            stage = "primary" if params["operation_id"] == 93 else "post"
            if self.interactive and not self.done[stage].is_set():
                return {"reason": "timeout", "status": "interactive"}
        if method == "tab.writeback_write":
            self.writes.append(deepcopy(params))
            return {
                "written": [
                    {
                        "id": item["id"],
                        "kind": "parameter",
                        "target": item["id"],
                        "before": {"value": 0.0},
                        "after": {"value": 1.0},
                    }
                    for item in params["write"]
                ]
            }
        if "operation_id" not in params and method in (
            "tab.get_analyze_result",
            "tab.get_post_analyze_result",
            "tab.writeback_preview",
        ):
            return self._current_draft(method, params)
        return self._native_reply(method, params)

    def _current_draft(self, method, params):
        if method in ("tab.get_analyze_result", "tab.get_post_analyze_result"):
            return {"summary": {}}
        if method == "tab.writeback_preview":
            if params["subtab_id"] == "post_analysis":
                return self._post_result(method, params)
            return {
                "has_draft": True,
                "items": [
                    {
                        "id": "param-1",
                        "kind": "parameter",
                        "target_name": "predict_offset",
                        "proposed": 0.24,
                        "current": 0.0,
                        "selected": False,
                    }
                ],
                "destination_context": {"active_label": "sample"},
            }
        raise AssertionError(method)

    def _native_reply(self, method, params):
        if method == "tab.post_analyze":
            self.current_stage = "post"
            assert params == {
                "tab_id": "t",
                "updates": {},
                "operation_id": 93,
                "run_operation_id": 71,
            }
            return {
                "operation_id": 104,
                "interactive": self.interactive,
                "params": {"bins": 64},
                "invalidated_on_success": [],
            }
        if method == "operation.await" and params["operation_id"] == 104:
            return {"reason": "completed", "status": "finished"}
        if method == "tab.get_post_analyze_result":
            assert params == {"tab_id": "t", "operation_id": 104}
            return {
                "summary": {"fidelity": 0.98},
                "invalid": [],
                "params": {"bins": 64},
                "operation_state": {"post_analysis_state": {"figure_names": ["cloud"]}},
            }
        if params.get("operation_id") == 104:
            return self._post_result(method, params)
        reply = super().__call__(method, params)
        if method == "tab.analyze":
            self.current_stage = "primary"
            reply["interactive"] = self.interactive
        if method == "tab.writeback_preview":
            reply["items"][0].update(
                kind="parameter",
                target_name="predict_offset",
                current=0.0,
                selected=False,
            )
        return reply

    def _post_result(self, method, params):
        if method == "tab.save_artifacts":
            assert params == {
                "tab_id": "t",
                "artifacts": [{"section": "post_analysis", "name": "results"}],
                "operation_id": 104,
            }
            return {
                "run_id": "run-a",
                "artifacts": {
                    "post_analysis": {
                        "results": {
                            "status": "Saved",
                            "members": {
                                "analysis:post": [
                                    {
                                        "format": "json",
                                        "path": "/runs/run-a/post_analysis/results.json",
                                        "status": "saved",
                                        "error": None,
                                    }
                                ],
                                "figure:cloud": [
                                    {
                                        "format": "png",
                                        "path": "/runs/run-a/post_analysis/cloud.png",
                                        "status": "saved",
                                        "error": None,
                                    }
                                ],
                            },
                        }
                    }
                },
            }
        assert params["subtab_id"] == "post_analysis"
        if method == "tab.get_figure":
            return {"png_b64": base64.b64encode(PNG).decode()}
        if method == "tab.writeback_preview":
            return {
                "has_draft": True,
                "items": [
                    {
                        "id": "classifier",
                        "kind": "parameter",
                        "target_name": "classifier",
                        "proposed": 0.98,
                        "current": 0.0,
                        "selected": False,
                    }
                ],
                "destination_context": {"active_label": "sample"},
            }
        raise AssertionError(method)
