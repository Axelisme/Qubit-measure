"""MCP-only projections of captured execution facts; never read the live GUI."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, TypeAdapter

from zcu_tools.core.element import ElementIdentity
from zcu_tools.mcp.measure.recipe_capture import RecipeActual
from zcu_tools.mcp.measure.recipe_form import RecipeFieldPath, field_path

if TYPE_CHECKING:
    from zcu_tools.mcp.measure.recipe import RecipeDefinition
    from zcu_tools.mcp.measure.recipe_api import RecipeDeclaration

ReplyDetail = Literal["summary", "full"]


@dataclass(frozen=True)
class SummaryParameter:
    """One recipe-owned cfg fact selected for fixed Run capture and public summary.

    name is the public actual mapping key, not a position. field is a nonempty
    tuple of str field names and int original list indexes captured at Run.
    Numeric strings remain literal names. unit is its native unit label, or None for no
    declared label. It never requests a value conversion.
    """

    name: str
    field: RecipeFieldPath
    unit: str | None = None

    def __post_init__(self) -> None:
        field_path(self.field)


@dataclass(frozen=True)
class SummaryEstimate:
    """One scalar analysis estimate declared by the recipe owner.

    name is its public key, never a position. value_key/error_key are nonempty
    tuples of str field names and int original list indexes into native summary.
    Numeric strings stay names. Missing paths are omitted; present null stays null.
    error_key=None means no uncertainty field. unit labels the native value, or
    None when the recipe declares no unit. No conversion occurs. element_key
    optionally selects an author-provided ElementIdentity in that same native
    summary. Missing identity is omitted, never inferred from value_key or name.
    """

    name: str
    value_key: RecipeFieldPath
    error_key: RecipeFieldPath | None = None
    unit: str | None = None
    element_key: RecipeFieldPath | None = None

    def __post_init__(self) -> None:
        field_path(self.value_key)
        if self.error_key is not None:
            field_path(self.error_key)
        if self.element_key is not None:
            field_path(self.element_key)


@dataclass(frozen=True)
class SummaryElements:
    """Capture declared leaves for every original model-list element at each Run.

    name prefixes opaque public keys "name/index/parameter.name"; these keys are
    labels, not field paths. field is the nonempty canonical model-list path.
    parameters declares nonempty relative logical leaf paths, units and labels,
    shared by every element. source is a nonblank author-chosen fallback label
    for leaves without a recorded origin (for example "default"). It never
    replaces missing/invalid values or an explicit/derived recorded origin.
    Names come only from the owner's FixedElements capture, never participants.
    """

    name: str
    field: RecipeFieldPath
    parameters: tuple[SummaryParameter, ...]
    source: str

    def __post_init__(self) -> None:
        field_path(self.field)
        if not self.parameters or not self.source.strip():
            raise ValueError("Element summaries require leaves and a source label")


@dataclass(frozen=True)
class SummaryEstimateRows:
    """Project relative scalar declarations for each captured native summary row.

    name prefixes opaque keys "name/row/estimate.name"; row is the summary list
    position, NOT an ElementIdentity. rows_key selects a native summary list;
    missing/null lists contribute no estimates, non-list values raise ValueError.
    estimates supplies relative value/error/element paths for each row in native
    order. Only an explicit author-provided ElementIdentity at element_key is
    projected. No position/name fields, participants or row indexes are inferred.
    """

    name: str
    rows_key: RecipeFieldPath
    estimates: tuple[SummaryEstimate, ...]

    def __post_init__(self) -> None:
        field_path(self.rows_key)
        if not self.estimates:
            raise ValueError("Row summaries require estimate declarations")


class StepReply(TypedDict):
    status: str
    reason: NotRequired[str]


class ParameterReply(TypedDict):
    """One captured condition, including explicitly supplied values.

    value is the resolved native value. source preserves the author\'s label or
    per-component sweep labels. unit, when present, is a native display label.
    element, when present, is the declaring owner\'s captured original identity.
    """

    value: Any
    source: Any
    unit: NotRequired[str]
    element: NotRequired[ElementIdentity]


class EstimateReply(TypedDict):
    """One author-provided scalar estimate from the captured native summary.

    value is the estimate, including an author-provided null. stderr is the
    declared uncertainty or None. unit is a native label or None. element is
    optional author-supplied identity; absence never triggers identity lookup.
    """

    value: Any
    stderr: Any
    unit: str | None
    element: NotRequired[ElementIdentity]


def project_control(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Keep control intent and GUI acknowledgement apart from execution results."""
    return {
        "execution": snapshot["execution"],
        "op": snapshot.get("op"),
        "run_op": snapshot.get("run_op"),
        "status": snapshot["status"],
        "phase": snapshot["phase"],
        "cancel_requested": snapshot.get("cancel_requested", False),
        "finish_early_requested": snapshot.get("finish_early_requested", False),
        "gui_cancel": deepcopy(snapshot.get("gui_cancel")),
    }


def _step(status: str, outcome: dict[str, Any] | None = None) -> StepReply:
    step: StepReply = {"status": status}
    if outcome and outcome.get("reason") is not None:
        step["reason"] = outcome["reason"]
    return step


def _actual(
    captured: RecipeActual | None, definition: RecipeDefinition | None
) -> dict[str, Any]:
    fields = captured["fields"] if captured is not None else {}
    parameters: dict[str, ParameterReply] = {}
    static = (
        {
            parameter.name: parameter
            for parameter in definition.summary_parameters
            if isinstance(parameter, SummaryParameter)
        }
        if definition
        else {}
    )
    for name, fact in fields.items():
        source = fact.get("source")
        projected: ParameterReply = {
            "value": deepcopy(fact["value"]),
            "source": deepcopy(source),
        }
        if "element" in fact:
            projected["element"] = deepcopy(fact["element"])
        unit = fact.get("unit", static[name].unit if name in static else None)
        if unit is not None:
            projected["unit"] = unit
        parameters[name] = projected
    return parameters


def _summary_value(summary: dict[str, Any], path: RecipeFieldPath) -> tuple[bool, Any]:
    value: Any = summary
    for segment in path:
        if isinstance(segment, str) and isinstance(value, dict) and segment in value:
            value = value[segment]
            continue
        if (
            type(segment) is int
            and isinstance(value, list)
            and 0 <= segment < len(value)
        ):
            value = value[segment]
            continue
        return False, None
    return True, value


def _expand_estimates(
    summary: dict[str, Any],
    declarations: tuple[SummaryEstimate | SummaryEstimateRows, ...],
) -> tuple[SummaryEstimate, ...]:
    expanded: list[SummaryEstimate] = []
    for declaration in declarations:
        if isinstance(declaration, SummaryEstimate):
            expanded.append(declaration)
            continue
        present, rows = _summary_value(summary, declaration.rows_key)
        if not present or rows is None:
            continue
        if not isinstance(rows, list):
            raise ValueError(f"Estimate rows {declaration.rows_key!r} must be a list")
        for index in range(len(rows)):
            prefix = (*declaration.rows_key, index)
            for estimate in declaration.estimates:
                expanded.append(
                    SummaryEstimate(
                        f"{declaration.name}/{index}/{estimate.name}",
                        (*prefix, *estimate.value_key),
                        (*prefix, *estimate.error_key)
                        if estimate.error_key is not None
                        else None,
                        estimate.unit,
                        (*prefix, *estimate.element_key)
                        if estimate.element_key is not None
                        else None,
                    )
                )
    return tuple(expanded)


def _summary_estimates(
    details: dict[str, Any],
    definition: RecipeDefinition | None,
    recipes: Sequence[RecipeDefinition],
) -> tuple[SummaryEstimate, ...]:
    if definition is not None:
        return _expand_estimates(details, definition.summary_estimates)
    # Analysis-only carries no recipe identity. Only unambiguous declarations
    # may project present scalar keys; never infer an experiment or read the GUI.
    by_key: dict[RecipeFieldPath, SummaryEstimate] = {}
    for recipe in recipes:
        for estimate in _expand_estimates(details, recipe.summary_estimates):
            if not _summary_value(details, estimate.value_key)[0]:
                continue
            previous = by_key.setdefault(estimate.value_key, estimate)
            if previous != estimate:
                raise ValueError(
                    f"Ambiguous analysis estimate {estimate.value_key!r}: "
                    "recipe declarations disagree"
                )
    return tuple(by_key.values())


def _pane(
    execution: dict[str, Any] | None,
    definition: RecipeDefinition | None,
    recipes: Sequence[RecipeDefinition],
) -> dict[str, Any] | None:
    if execution is None:
        return None
    result = execution.get("result") or {}
    summary = result.get("summary")
    estimates: dict[str, EstimateReply] = {}
    if isinstance(summary, dict):
        for estimate in _summary_estimates(summary, definition, recipes):
            present, value = _summary_value(summary, estimate.value_key)
            if not present:
                continue
            estimates[estimate.name] = {
                "value": value,
                "stderr": _summary_value(summary, estimate.error_key)[1]
                if estimate.error_key is not None
                else None,
                "unit": estimate.unit,
            }
            if estimate.element_key is not None:
                present, element = _summary_value(summary, estimate.element_key)
                if present and element is not None:
                    estimates[estimate.name]["element"] = TypeAdapter(
                        ElementIdentity
                    ).validate_python(element, strict=True)
    return {
        "params": result.get("params", execution.get("params")),
        "estimates": estimates,
        "summary": summary,
        "invalid": result.get("invalid", []),
    }


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
) -> StepReply:
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


def _analysis_save_step(execution: dict[str, Any] | None) -> StepReply:
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
        "finish_early_requested": False,
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


def _script_execution(snapshot: dict[str, object]) -> dict[str, object]:
    """Encode the new dataclass snapshot's native models at the reply boundary."""
    projected = deepcopy(snapshot)
    pending = projected.get("pending_question")
    if isinstance(pending, dict):
        question = pending["question"]
        if isinstance(question, BaseModel):
            pending["question"] = question.model_dump(mode="json")
    outcome = projected.get("outcome")
    if isinstance(outcome, dict):
        report = outcome["report"]
        if isinstance(report, BaseModel):
            outcome["report"] = report.model_dump(mode="json")
        outcome["runs"] = list(outcome["runs"])
    return projected


def project_execution(
    snapshot: dict[str, Any],
    *,
    recipes: Sequence[RecipeDefinition | RecipeDeclaration],
    definition: RecipeDefinition | RecipeDeclaration | None = None,
    detail: ReplyDetail = "summary",
) -> dict[str, Any]:
    """Project local captures using injected declarations, without any GUI read.

    recipes supplies the owning session declarations for analysis-only estimates.
    definition selects one recipe's actual/estimate fields; None means standalone
    analysis. Conflicting present estimate keys raise ValueError. detail selects
    the operator summary or detached native detail. Both expose recipe actual as
    a flat summary-name mapping of value/source/unit, including explicit inputs
    and individually labeled sweep components. Internal cfg refs never appear
    in actual or candidates. Each analysis pane
    exposes params, declared scalar estimates, unchanged native summary and
    unchanged native invalid entries. No fit_quality interpretation occurs.
    artifacts preserves each captured native Save report's section/member/output
    evidence, and run_id comes from the latest captured report. Summary interaction
    comes from analysis_stage; full retains both stages. No identity is inferred.
    steps.raw_save is None without a SaveReport; failed calls retain their cause
    only in error. No admission state is inferred from missing reports.
    Queries perform no RPC, image read, filesystem check or guard refresh.
    """
    from zcu_tools.mcp.measure.recipe import RecipeDefinition

    if "pending_question" in snapshot:
        return _script_execution(snapshot)
    recipes = tuple(item for item in recipes if isinstance(item, RecipeDefinition))
    definition = definition if isinstance(definition, RecipeDefinition) else None
    if detail == "full":
        full = deepcopy(snapshot)
        if "recipe" in snapshot:
            full["actual"] = _actual(snapshot.get("actual"), definition)
        return full
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
    primary = _pane(primary_execution, definition, recipes)
    post = _pane(post_execution, definition, recipes)
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
            "finish_early_requested": snapshot["finish_early_requested"],
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
            "actual": _actual(snapshot.get("actual"), definition),
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


def _raw_save_step(raw: dict[str, Any] | None) -> StepReply | None:
    """Describe raw Save evidence without inferring admission from absent reports."""
    raw_step: StepReply | None = None
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
