"""MCP-only projections of captured execution facts; never read the live GUI."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

if TYPE_CHECKING:
    from recipes import RecipeDefinition

ReplyDetail = Literal["summary", "full"]


@dataclass(frozen=True)
class SummaryParameter:
    name: str
    field: str
    unit: str | None = None


@dataclass(frozen=True)
class SummaryEstimate:
    name: str
    value_key: str
    error_key: str | None = None
    unit: str | None = None


class StepReply(TypedDict):
    status: str
    reason: NotRequired[str]


class ParameterReply(TypedDict):
    value: Any
    source: Any
    unit: NotRequired[str]


class EstimateReply(TypedDict):
    value: Any
    stderr: Any
    unit: str | None
    quality: Any


def _step(status: str, outcome: dict[str, Any] | None = None) -> StepReply:
    step: StepReply = {"status": status}
    if outcome and outcome.get("reason") is not None:
        step["reason"] = outcome["reason"]
    return step


def _actual(
    captured: dict[str, Any] | None, definition: RecipeDefinition
) -> dict[str, Any]:
    captured = captured or {}
    fields = captured.get("fields", {})
    parameters: dict[str, ParameterReply] = {}
    modules: dict[str, ParameterReply] = {}
    for parameter in definition.summary_parameters:
        fact = fields.get(parameter.field)
        if fact is None:
            continue
        projected: ParameterReply = {
            "value": deepcopy(fact["value"]),
            "source": deepcopy(fact.get("source")),
        }
        unit = fact.get("unit", parameter.unit)
        if unit is not None:
            projected["unit"] = unit
        if parameter.field.startswith("modules.") and parameter.field.count(".") == 1:
            modules[parameter.field.split(".", 1)[1]] = projected
        else:
            parameters[parameter.name] = projected
    return {
        "cfg_ref": captured.get("cfg_ref"),
        "parameters": parameters,
        "modules": modules,
    }


def _pane(
    execution: dict[str, Any] | None, definition: RecipeDefinition
) -> tuple[dict[str, Any] | None, dict[str, str]]:
    if execution is None:
        return None, {}
    result = execution.get("result") or {}
    native = result.get("summary")
    details = deepcopy(native) if isinstance(native, dict) else {"result": native}
    estimates: dict[str, EstimateReply] = {}
    paths = {}
    for estimate in definition.summary_estimates:
        if estimate.value_key not in details:
            continue
        estimates[estimate.name] = {
            "value": details.pop(estimate.value_key),
            "stderr": details.pop(estimate.error_key, None)
            if estimate.error_key
            else None,
            "unit": estimate.unit,
            "quality": None,
        }
        paths[f"summary.{estimate.value_key}"] = f"estimates.{estimate.name}.value"
        if estimate.error_key:
            paths[f"summary.{estimate.error_key}"] = f"estimates.{estimate.name}.stderr"
    return {
        "params": result.get("params", execution.get("params")),
        "estimates": estimates,
        "details": details,
        "warnings": result.get("warnings", []),
    }, paths


def _image_artifacts(execution: dict[str, Any] | None) -> dict[str, Any]:
    if execution is None:
        return {}
    artifacts = {}
    for image in execution.get("saved_images", []):
        artifacts[image["figure_name"]] = {
            "status": "saved",
            "lifetime": "persistent",
            "members": {"image": [{"path": image["image_path"], "status": "saved"}]},
        }
    return artifacts


def _candidate(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "kind": "parameter" if item.get("kind") == "metadict" else item.get("kind"),
        "target": item.get("target_name"),
        "resolved_target": item.get("resolved_target"),
        "proposed": deepcopy(item.get("proposed", item.get("proposed_value"))),
        "current": deepcopy(item.get("current")),
        "selected": item.get("selected"),
    }


def _writeback(snapshot: dict[str, Any]) -> dict[str, Any]:
    primary = snapshot.get("writeback") or {}
    post = snapshot.get("post_writeback") or {}
    return {
        "destination": {
            "context": deepcopy(
                primary.get("destination_context") or post.get("destination_context")
            )
        }
        if primary.get("destination_context") or post.get("destination_context")
        else {},
        "stages": {
            "primary": [_candidate(item) for item in primary.get("items", [])],
            "post": [_candidate(item) for item in post.get("items", [])],
        },
        "requires": primary.get("requires", []) + post.get("requires", []),
    }


def _analysis_step(execution: dict[str, Any] | None) -> StepReply:
    if execution is None:
        return _step("not_started")
    outcome = execution.get("operation_outcome")
    return _step(outcome["status"] if outcome else execution["status"], outcome)


def _analysis_save_step(execution: dict[str, Any] | None) -> StepReply:
    return _step(execution["save_status"] if execution else "not_started")


def project_execution(
    snapshot: dict[str, Any],
    *,
    definition: RecipeDefinition,
    detail: ReplyDetail = "summary",
) -> dict[str, Any]:
    """Project one captured recipe snapshot without refreshing observations."""
    if detail == "full":
        return deepcopy(snapshot)
    primary_execution = snapshot.get("analysis")
    post_execution = snapshot.get("post_analysis")
    primary, primary_paths = _pane(primary_execution, definition)
    post, post_paths = _pane(post_execution, definition)
    invalid = []
    for stage, execution, paths in (
        ("primary", primary_execution, primary_paths),
        ("post", post_execution, post_paths),
    ):
        for item in ((execution or {}).get("result") or {}).get("invalid", []):
            path = item["path"]
            projected_path = paths.get(path, path.replace("summary.", "details.", 1))
            invalid.append({**item, "path": f"analysis.{stage}.{projected_path}"})
    raw = snapshot["raw_save"]
    paths = []
    if raw["path"] is not None:
        paths.append({"path": raw["path"], "status": "saved"})
    if raw["reserved_path"] is not None and raw["reserved_path"] != raw["path"]:
        paths.append({"path": raw["reserved_path"], "status": "reserved"})
    outcome = snapshot.get("run_outcome")
    run = _step(
        outcome["status"]
        if outcome
        else "running"
        if snapshot["run_op"] is not None
        else "not_started",
        outcome,
    )
    return deepcopy(
        {
            "execution": snapshot["execution"],
            "run_id": None,
            "recipe": snapshot["recipe"],
            "tab": snapshot["tab"],
            "op": snapshot["op"],
            "run_op": snapshot["run_op"],
            "status": snapshot["status"],
            "phase": snapshot["phase"],
            "cancel_requested": snapshot["cancel_requested"],
            "finish_early_requested": snapshot["finish_early_requested"],
            "steps": {
                "run": run,
                "raw_save": _step(raw["status"], raw.get("operation_outcome")),
                "analysis": {
                    "primary": _analysis_step(primary_execution),
                    "post": _analysis_step(post_execution),
                },
                "analysis_save": {
                    "primary": _analysis_save_step(primary_execution),
                    "post": _analysis_save_step(post_execution),
                },
            },
            "actual": _actual(snapshot.get("actual"), definition),
            "analysis": {
                "stage": snapshot.get("analysis_stage"),
                "primary": primary,
                "post": post,
            },
            "artifacts": {
                "raw": {
                    "data": {
                        "status": raw["status"],
                        "lifetime": "persistent",
                        "members": {"data": paths},
                    }
                },
                "analysis": _image_artifacts(primary_execution),
                "post_analysis": _image_artifacts(post_execution),
            },
            "writeback": _writeback(snapshot),
            "interaction": (primary_execution or {}).get("interaction")
            or (post_execution or {}).get("interaction"),
            "missing": snapshot["missing"],
            "invalid": invalid,
            "error": snapshot["error"],
        }
    )
