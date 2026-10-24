"""Synthetic GUI wire collaborator and declaration startup for recipe contracts."""

from threading import Event

from pydantic import BaseModel, Field
from zcu_tools.gui.form import FormModel
from zcu_tools.mcp.measure.script_recipe_execution import ScriptRecipeExecution

from ._support import MeasureClient


class Inputs(FormModel):
    count: int = Field(description="Synthetic repetition count")


class Report(BaseModel):
    count: int


class Fit(BaseModel):
    estimate: float


class SyntheticGui:
    """A wire collaborator; no production recipe or instrument dependencies."""

    def __init__(self, *, hold: bool = False, partial_first: bool = False) -> None:
        self.hold = hold
        self.partial_first = partial_first
        self.started = Event()
        self.cancelled = Event()
        self.runs = 0
        self.submissions = 0

    def respond(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.read_recipe_field":
            return {
                "description": {"path": [], "target": None, "children": []},
                "input_present": True,
                "input": {"count": 9},
                "value": {"count": 9},
                "errors": [],
                "value_status": "available",
                "selector": None,
                "source": None,
                "readonly": False,
            }
        if method == "tab.check_recipe_ready":
            return {"errors": []}
        if method == "tab.writeback_begin":
            self.submissions += 1
            return {"submission_id": f"wb-{self.submissions}", "status": "pending"}
        if method == "tab.new":
            return {"tab_id": "tab-1"}
        if method in {
            "tab.reset_cfg",
            "tab.edit_recipe_fields",
            "tab.select_recipe_module",
            "tab.select_recipe_option",
            "tab.run_recipe_command",
        }:
            return {"errors": [], "changed": True}
        return self._operation(method, params)

    def _operation(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.run_start":
            self.runs += 1
            self.started.set()
            return {"operation_id": self.runs}
        if method in {"tab.analyze", "tab.post_analyze"}:
            return {"operation_id": 100}
        if method == "operation.await":
            if (
                self.hold
                and params["operation_id"] != 100
                and not self.cancelled.is_set()
            ):
                return {"reason": "timeout"}
            return {
                "reason": "completed",
                "status": "cancelled"
                if self.cancelled.is_set() and params["operation_id"] != 100
                else "finished",
            }
        if method in {"tab.cancel", "tab.analyze_cancel", "operation.cancel"}:
            self.cancelled.set()
            return {"status": "requested"}
        return self._snapshots(method, params)

    def _snapshots(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.snapshot":
            return {
                "tabs": [
                    {
                        "tab_id": "tab-1",
                        "adapter_name": "synthetic",
                        "interaction": {"is_running": False, "is_analyzing": False},
                        "raw_data": {"run_id": f"native-{self.runs}", "saved": False},
                        "result_state": {
                            "available": True,
                            "source_operation_id": self.runs,
                            "partial_result": self.cancelled.is_set()
                            or (self.partial_first and self.runs == 1),
                        },
                    }
                ]
            }
        return self._artifacts(method, params)

    def _artifacts(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.writeback_preview":
            return {
                "has_draft": True,
                "items": [
                    {
                        "id": "param-0",
                        "target_name": "Q1.freq",
                        "actual_path": "Q1.freq",
                        "proposed": 3.5,
                        "current": 2.0,
                        "stderr": None,
                        "note": None,
                        "kind": "parameter",
                    }
                ],
                "destination_context": {"active_label": "preview-context"},
            }
        if method == "tab.writeback_accept":
            return {
                "status": "completed",
                "written": [{"subtab_id": "analysis", "ids": ["param-0"]}],
                "error": None,
            }
        if method == "tab.writeback_write":
            assert set(params) == {"tab_id", "subtab_id", "write"}
            return {"written": [{"id": "param-0", "target": "Q1.freq"}]}
        if method == "tab.get_analyze_result":
            return {"summary": {"estimate": 3.5}, "invalid": []}
        if method in {"tab.save_data", "tab.save_artifacts"}:
            sections = (
                ("measurement_data",)
                if method == "tab.save_data"
                else ("measurement_data", "analysis")
            )
            return {
                "run_id": f"native-{self.runs}",
                "artifacts": {
                    section: {
                        "results": {
                            "status": "Saved",
                            "members": {
                                "data": [
                                    {
                                        "format": "data_h5"
                                        if section == "measurement_data"
                                        else "json",
                                        "path": "/synthetic/output",
                                        "status": "saved",
                                        "error": None,
                                    }
                                ]
                            },
                        }
                    }
                    for section in sections
                },
            }
        raise AssertionError(f"Unexpected GUI request: {method}")


class MappingWritebackGui(SyntheticGui):
    """A wire owner offering a structured module candidate and counting writes."""

    def __init__(self) -> None:
        super().__init__()
        self.writes = 0

    def respond(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.writeback_accept":
            self.writes += 1
        reply = super().respond(method, params)
        if method == "tab.writeback_preview":
            return {
                **reply,
                "items": [
                    {
                        "id": "param-0",
                        "kind": "module",
                        "target_name": "Q1.pulses.pi",
                        "actual_path": "library:opaque.key",
                        "proposed": {"pulse": {"gain": 0.2}},
                    }
                ],
            }
        return reply


def answer_writeback(execution: ScriptRecipeExecution) -> None:
    """Accept the current public writeback question and await its continuation."""
    execution.wait(3)
    pending = execution.snapshot().pending_question
    assert pending is not None
    execution.answer(
        {
            "request_id": pending.request_id,
            "answer": {
                "accepted": True,
                "items": pending.question.model_dump(mode="json")["proposal"]["items"],
            },
        }
    )
    execution.wait(3)


def start(client: MeasureClient, name: str, count: int = 9) -> ScriptRecipeExecution:
    """Start through the shared registry, asserting the declaration route."""
    execution = client.context.session.recipes.start(
        client.context, name, {"count": count}
    )
    assert isinstance(execution, ScriptRecipeExecution)
    return execution
