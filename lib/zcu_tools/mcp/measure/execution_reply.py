"""MCP projections of captured facts; queries never read the live GUI."""

from copy import deepcopy
from typing import Any, Literal

from pydantic import TypeAdapter
from pydantic_core import to_jsonable_python

from zcu_tools.core.element import ElementIdentity
from zcu_tools.format_version import YamlValue

ReplyDetail = Literal["summary", "full"]


def project_control(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Return stop acknowledgement, not proof of operation completion."""
    result = {
        key: snapshot[key]
        for key in (
            "execution",
            "status",
            "phase",
            "cancel_requested",
            "op",
            "gui_cancel",
            "cancelled",
            "reason",
            "active_path",
            "candidates",
        )
        if key in snapshot
    }
    result["run_op"] = snapshot.get("run_op")
    return deepcopy(result)


def _pane(execution: dict[str, Any] | None) -> dict[str, Any] | None:
    if execution is None:
        return None
    result = execution.get("result") or {}
    return {
        "params": result.get("params", execution.get("params")),
        "summary": result.get("summary"),
        "invalid": result.get("invalid", []),
    }


def _step(status: str, outcome: dict[str, Any] | None = None) -> dict[str, Any]:
    step: dict[str, Any] = {"status": status}
    if outcome and outcome.get("reason") is not None:
        step["reason"] = outcome["reason"]
    return step


def _module_summary(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("A module proposal must be an object")
    summary = {
        key: deepcopy(value[key])
        for key in (
            "type",
            "freq",
            "gain",
            "phase",
            "cloned_from",
            "ro_length",
            "trig_offset",
        )
        if key in value
    }
    waveform = value.get("waveform")
    if isinstance(waveform, dict):
        summary["waveform"] = {
            key: item
            for key, item in waveform.items()
            if not isinstance(item, (dict, list))
        }
    for key in ("pulse_cfg",):
        if key in value:
            summary[key] = _module_summary(value[key])
    return summary


def _changed_paths(
    proposed: dict[str, Any], current: dict[str, Any], prefix: str = ""
) -> list[str]:
    paths = []
    for key in sorted(proposed.keys() | current.keys()):
        path = f"{prefix}.{key}" if prefix else key
        before, after = current.get(key), proposed.get(key)
        if isinstance(before, dict) and isinstance(after, dict):
            paths.extend(_changed_paths(after, before, path))
        elif key not in current or key not in proposed or before != after:
            paths.append(path)
    return paths


def _candidate(item: dict[str, Any]) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "id": item["id"],
        "kind": item.get("kind"),
        "target": item.get("target_name"),
        "resolved_target": item.get("resolved_target"),
        "proposed": deepcopy(item.get("proposed", item.get("proposed_value"))),
        "current": deepcopy(item.get("current")),
    }
    if item.get("element") is not None:
        candidate["element"] = TypeAdapter(ElementIdentity).validate_python(
            item["element"], strict=True
        )
    if item.get("kind") == "module":
        proposed = candidate["proposed"]
        current = candidate["current"]
        candidate.update(
            proposed=_module_summary(proposed),
            current=_module_summary(current),
            changes=_changed_paths(proposed, current)
            if current is not None
            else ["create"],
        )
    return candidate


def _destination(native: dict[str, Any] | None) -> dict[str, Any]:
    if native is None or "active_label" not in native:
        return {}
    return {"context": {"active_label": native["active_label"]}}


def _writeback(snapshot: dict[str, Any]) -> dict[str, Any]:
    primary = snapshot.get("writeback") or {}
    post = snapshot.get("post_writeback") or {}
    return {
        "destination": _destination(
            primary.get("destination_context") or post.get("destination_context")
        ),
        "stages": {
            "primary": [_candidate(item) for item in primary.get("items", [])],
            "post": [_candidate(item) for item in post.get("items", [])],
        },
        "requires": primary.get("requires", []) + post.get("requires", []),
        "receipts": snapshot.get("written", ()),
    }


def _analysis_step(
    execution: dict[str, Any] | None, start: dict[str, Any] | None
) -> dict[str, Any]:
    if execution is None:
        return _step(start["status"] if start else "not_started", start)
    outcome = execution.get("operation_outcome")
    if outcome is not None:
        return _step(outcome["status"], outcome)
    receipt = execution["start"]
    return _step(
        "interactive"
        if execution["op"] is not None and execution["status"] == "interactive"
        else receipt["status"],
        receipt,
    )


def _analysis_save_step(execution: dict[str, Any] | None) -> dict[str, Any]:
    return _step(execution["save_status"] if execution else "not_started")


def _analysis_envelope(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Locate an analysis-only execution in the shared summary, not a fake recipe."""
    stage = snapshot["stage"]
    return {
        "execution": snapshot["execution"],
        "recipe": None,
        "tab": snapshot["tab"],
        "op": snapshot["op"],
        "run_op": None,
        "status": snapshot["status"],
        "phase": snapshot["phase"],
        "cancel_requested": snapshot["cancel_requested"],
        "run_start": {"status": "not_started"},
        "raw_save": None,
        "analysis_stage": stage,
        "analysis": snapshot if stage == "primary" else None,
        "post_analysis": snapshot if stage == "post" else None,
        "writeback": snapshot["writeback"] if stage == "primary" else None,
        "post_writeback": snapshot["writeback"] if stage == "post" else None,
        "missing": [],
        "error": snapshot["error"],
    }


def _script_execution(
    snapshot: dict[str, YamlValue], detail: ReplyDetail
) -> dict[str, YamlValue]:
    projected = TypeAdapter(dict[str, YamlValue]).validate_python(snapshot)
    status = projected["status"]
    failure: dict[str, YamlValue] = {}
    if status == "failed":
        error = projected["error"]
        if not isinstance(error, dict):
            raise ValueError("Failed recipe is missing its error diagnostic")
        failure["failure"] = {"phase": error["phase"], "reason": error["message"]}
    if detail == "full":
        return {**projected, **failure}
    if status == "needs_parameters":
        return {"status": status, "missing": projected["missing"], "outcome": None}
    outcome = projected.get("outcome")
    if isinstance(outcome, dict):
        return {
            **{key: outcome[key] for key in ("status", "report", "runs")},
            "execution": projected["execution"],
            **failure,
        }
    result: dict[str, YamlValue] = {
        "status": status,
        "execution": projected["execution"],
    }
    if projected.get("pending_question") is not None:
        result["pending_question"] = projected["pending_question"]
    if projected.get("pending_analysis") is not None:
        result["pending"] = projected["pending_analysis"]
    return result


def project_execution(
    snapshot: dict[str, Any],
    *,
    detail: ReplyDetail = "summary",
) -> dict[str, Any]:
    """Project a recipe outcome or standalone analysis without external reads.

    Recipe terminal summaries contain status/report/runs/execution; failed
    replies also contain failure with attempted phase and human reason, without
    a formatted traceback. Nonterminal replies retain execution and explicit
    pending question/analysis. Missing
    parameters return needs_parameters/missing/outcome=None. Full replies retain
    the complete snapshot, including detached per-Run conditions, native analysis
    estimates/invalid paths, explicit writeback candidates and errors.
    Estimates remain at each stage\'s owner-native summary, with original values,
    stderr, units and element identities only when supplied by that owner.
    Standalone analysis retains the same native summary without recipe labels.
    """
    if "pending_question" in snapshot:
        encoded = TypeAdapter(dict[str, YamlValue]).validate_python(
            to_jsonable_python(snapshot)
        )
        return _script_execution(encoded, detail)
    return _analysis_execution(snapshot, detail=detail)


def _analysis_execution(
    snapshot: dict[str, Any], *, detail: ReplyDetail
) -> dict[str, Any]:
    """Project standalone analysis captures without recipe declarations."""
    if detail == "full":
        return deepcopy(snapshot)
    if "recipe" not in snapshot:
        snapshot = _analysis_envelope(snapshot)
    primary_execution = snapshot.get("analysis")
    post_execution = snapshot.get("post_analysis")
    current_execution = (
        post_execution
        if snapshot.get("analysis_stage") == "post"
        else primary_execution
    )
    interaction = (current_execution or {}).get("interaction")
    primary = _pane(primary_execution)
    post = _pane(post_execution)
    invalid = [
        *(primary or {}).get("invalid", []),
        *(post or {}).get("invalid", []),
    ]
    raw = snapshot["raw_save"]
    reports = [
        raw,
        (primary_execution or {}).get("save_report"),
        (post_execution or {}).get("save_report"),
    ]
    artifacts: dict[str, Any] = {}
    run_id = None
    for report in reports:
        if report is None:
            continue
        run_id = report["run_id"]
        for section, captures in report["artifacts"].items():
            artifacts.setdefault(section, {}).update(captures)
    raw_step = _raw_save_step(raw)
    outcome = snapshot.get("run_outcome")
    run = _step(
        outcome["status"]
        if outcome
        else "running"
        if snapshot["run_op"] is not None
        else snapshot["run_start"]["status"],
        outcome or snapshot.get("run_start"),
    )
    return deepcopy(
        {
            "execution": snapshot["execution"],
            "run_id": run_id,
            "raw_data": (current_execution or {}).get("raw_data"),
            "recipe": snapshot["recipe"],
            "tab": snapshot["tab"],
            "op": snapshot["op"],
            "run_op": snapshot["run_op"],
            "status": snapshot["status"],
            "phase": snapshot["phase"],
            "cancel_requested": snapshot["cancel_requested"],
            "steps": {
                "run": run,
                "raw_save": raw_step,
                "analysis": {
                    "primary": _analysis_step(
                        primary_execution,
                        snapshot.get("analysis_starts", {}).get("primary"),
                    ),
                    "post": _analysis_step(
                        post_execution, snapshot.get("analysis_starts", {}).get("post")
                    ),
                },
                "analysis_save": {
                    "primary": _analysis_save_step(primary_execution),
                    "post": _analysis_save_step(post_execution),
                },
            },
            "actual": {},
            "analysis": {
                "stage": snapshot.get("analysis_stage"),
                "primary": primary,
                "post": post,
            },
            "artifacts": artifacts,
            "writeback": _writeback(snapshot),
            "interaction": interaction,
            "question_items": snapshot.get("question_items"),
            "question_preview": snapshot.get("question_preview"),
            "missing": snapshot["missing"],
            "invalid": invalid,
            "error": snapshot["error"],
        }
    )


def _raw_save_step(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Describe raw Save evidence without inferring admission from absent reports."""
    raw_step: dict[str, Any] | None = None
    if raw is not None:
        raw_artifacts = [
            artifact
            for section in raw["artifacts"].values()
            for artifact in section.values()
        ]
        raw_step = _step(
            "saved"
            if raw_artifacts
            and all(artifact["status"] == "Saved" for artifact in raw_artifacts)
            else "incomplete"
        )
    return raw_step
