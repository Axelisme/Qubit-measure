"""Persist the wall-time batch plan and render reports from committed row facts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, NotRequired, TypedDict

from pydantic import Field, StrictFloat, TypeAdapter

CORE_TARGET_S = 180.0


class PlannedScene(TypedDict):
    """Exact adapter and journal location for one planned scene."""

    adapter: str
    directory: str


class BatchPlan(TypedDict):
    """Revision, input identities and planned scene journals."""

    format_version: Literal[2]
    commit: str
    inputs: dict[str, str]
    manifest_scenes: int
    scenes: dict[str, PlannedScene]


class RunStart(TypedDict):
    """The UTC ISO-8601 and monotonic seconds sampled before Run dispatch."""

    started_at: str
    monotonic_s: float
    cold_start: bool


type RowStatus = Literal[
    "ok",
    "missing_breakdown",
    "missing_defaults",
    "missing_adapter",
    "not_runnable",
    "not_run",
    "incomplete",
    "timeout",
    "run_failed",
]


class Row(TypedDict):
    """One report row and its measured or unavailable Run facts.

    scene_id identifies the workload; adapter records its exact experiment key.
    case_directory is relative to the output (or absolute for retained evidence),
    or None when there are no case artifacts. Timestamps are UTC ISO-8601 strings;
    wall_s is exact measured seconds, and wall_s_lower_bound is used instead for
    timed-out Runs. An incomplete row has only the pre-dispatch started_at and
    neither duration. Breakdown is complete, partial, missing, or None if not
    attempted. Acquires counts observed main records. The stage fields are
    measured seconds or None. Materialize includes matched deferred work;
    remainder is acquire wall minus the stages inside it. core_s is the
    sum of the four stages for a complete Run breakdown. For an incomplete
    breakdown, core_s_lower_bound sums only acquires with all stages measured
    and their matched deferred work, or is None if none qualify.
    Status distinguishes complete, missing
    data/defaults, unsupported, failed, timed-out and uncompleted rows. Reason
    explains an exclusion, unresolved defaults or a failure, when applicable.
    failure_record is present on failed Runs and names the diagnostic relative
    to case_directory. cold_start is true
    for the first dispatch in its process, false for later Runs and None for
    unstarted rows. reused is true when this batch retained another batch's row.
    """

    scene_id: str
    adapter: str
    case_directory: str | None
    started_at: str | None
    finished_at: str | None
    wall_s: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)] | None
    wall_s_lower_bound: float | None
    breakdown: Literal["complete", "partial", "missing"] | None
    acquires: int | None
    compile_s: float | None
    evolve_s: float | None
    readout_s: float | None
    materialize_s: StrictFloat | None
    core_s: StrictFloat | None
    core_s_lower_bound: StrictFloat | None
    remainder_s: float | None
    status: RowStatus
    reason: str | None
    failure_record: NotRequired[str]
    cold_start: bool | None
    reused: bool


def new_row(
    scene_id: str,
    adapter: str,
    directory: str | None,
    status: RowStatus,
    reason: str | None,
) -> Row:
    """Make a row with an identity and classification, before attaching timings."""
    return {
        "scene_id": scene_id,
        "adapter": adapter,
        "case_directory": directory,
        "started_at": None,
        "finished_at": None,
        "wall_s": None,
        "wall_s_lower_bound": None,
        "breakdown": None,
        "acquires": None,
        "compile_s": None,
        "evolve_s": None,
        "readout_s": None,
        "materialize_s": None,
        "core_s": None,
        "core_s_lower_bound": None,
        "remainder_s": None,
        "status": status,
        "reason": reason,
        "cold_start": None,
        "reused": False,
    }


def _seconds(seconds: float | None) -> str:
    return "—" if seconds is None else f"{seconds:.3f}"


def _summary_row(row: Row) -> str:
    stages = " | ".join(
        _seconds(seconds)
        for seconds in (
            row["compile_s"],
            row["evolve_s"],
            row["readout_s"],
            row["materialize_s"],
        )
    )
    core = _seconds(row["core_s"])
    if (core_bound := row["core_s_lower_bound"]) is not None:
        core = f"≥{_seconds(core_bound)}"
    remainder = _seconds(row["remainder_s"])
    status = {
        "ok": "OK",
        "missing_breakdown": "分項缺",
        "missing_defaults": "推不出預設",
        "missing_adapter": "missing_adapter",
        "not_runnable": "FTF 不能跑",
        "not_run": "not_run",
        "incomplete": "incomplete",
        "timeout": "timeout",
        "run_failed": "run_failed",
    }[row["status"]]
    if row["status"] == "timeout":
        status += "（部分）" if row["breakdown"] == "partial" else "（分項缺）"
    if row["reason"] is not None:
        reason = " ".join(row["reason"].splitlines()).replace("|", r"\|")
        status = f"{status}: {reason}"
    acquires = "—" if row["acquires"] is None else str(row["acquires"])
    lower_bound = row["wall_s_lower_bound"]
    wall = _seconds(row["wall_s"])
    if lower_bound is not None:
        wall = f">{lower_bound}"
    source = (
        ", ".join(
            note
            for note, applies in (
                ("cold start", row["cold_start"]),
                ("reused", row["reused"]),
            )
            if applies
        )
        or "—"
    )
    return (
        f"| {row['scene_id']} | {row['adapter']} | {wall} | {stages} | {core} | "
        f"{remainder} | {acquires} | {status} | {source} |\n"
    )


def _write_report(
    output: Path, rows: list[Row], plan: BatchPlan, resolved: int
) -> None:
    write_json(
        output / "report.json",
        {
            "commit": plan["commit"],
            "inputs": plan["inputs"],
            "rows": rows,
            "coverage": {
                "resolved": resolved,
                "selected": len(rows),
                "manifest": plan["manifest_scenes"],
            },
        },
    )
    (output / "summary.md").write_text(
        f"Coverage: {resolved}/{len(rows)} selected scenes resolved; "
        f"{plan['manifest_scenes']} manifest scenes.\n\n"
        + _run_summary(rows)
        + "| Scene | Adapter | Run (s) | Compile (s) | Evolve (s) | Readout (s) | Materialize (s) | Core (s) | Remainder (s) | Acquires | Status | Source |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n"
        + "".join(map(_summary_row, rows))
        + _core_summary(rows)
        + _failure_list(rows),
        encoding="utf-8",
    )


def _core_summary(rows: list[Row]) -> str:
    over_target: list[str] = []
    lower_bound_over_target: list[str] = []
    undetermined: list[str] = []
    for row in rows:
        scene_id = row["scene_id"]
        if (core := row["core_s"]) is not None:
            if core > CORE_TARGET_S:
                over_target.append(scene_id)
        elif (bound := row["core_s_lower_bound"]) is not None and bound > CORE_TARGET_S:
            lower_bound_over_target.append(scene_id)
        else:
            undetermined.append(scene_id)
    return (
        f"\nCore >{CORE_TARGET_S:g} s: {_format_names(over_target)}\n"
        f"Core 已知至少超標: {_format_names(lower_bound_over_target)}\n"
        f"Core 未判定: {_format_names(undetermined)}\n"
    )


def _run_summary(rows: list[Row]) -> str:
    timed = sum(row["status"] in {"ok", "missing_breakdown"} for row in rows)
    timeouts = sum(row["status"] == "timeout" for row in rows)
    failures = sum(row["status"] == "run_failed" for row in rows)
    return f"Runs: {timed} timed, {timeouts} timeout, {failures} run_failed.\n\n"


def _failure_list(rows: list[Row]) -> str:
    by_reason: dict[str, list[str]] = {}
    for row in rows:
        if row["status"] != "run_failed":
            continue
        reason = row["reason"]
        directory = row["case_directory"]
        record = row.get("failure_record")
        if not reason or directory is None or not record:
            raise ValueError(f"Run failure for {row['scene_id']!r} lacks diagnostics")
        record = (Path(directory) / record).as_posix()
        by_reason.setdefault(reason, []).append(
            f"  - `{row['scene_id']}` — `{record}`\n"
        )
    groups = (
        f"- {' '.join(reason.splitlines())}\n{''.join(entries)}"
        for reason, entries in by_reason.items()
    )
    return "\nRun failures:\n" + ("".join(groups) or "—\n")


def write_json(path: Path, value: object) -> None:
    """Atomically replace path with JSON in its existing parent directory."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_plan(output: Path) -> BatchPlan:
    """Load a wall-time journal's plan, raising on absent or malformed data."""
    raw = (output / "batch.json").read_bytes()
    document = json.loads(raw)
    if (
        isinstance(document, dict)
        and "format_version" not in document
        and "cases" in document
    ):
        commit = document.get("commit")
        original = f" ({commit})" if isinstance(commit, str) else ""
        raise ValueError(
            f"Legacy wall-time journal; use --report-only at the original batch commit{original}"
        )
    return TypeAdapter(BatchPlan).validate_json(raw)


def _read_row(location: Path, scene_id: str, adapter: str) -> Row | None:
    path = location / "row.json"
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    row = TypeAdapter(Row).validate_json(data)
    if row["scene_id"] != scene_id:
        raise ValueError(
            f"{path}: expected scene {scene_id!r}, got {row['scene_id']!r}"
        )
    if row["adapter"] != adapter:
        raise ValueError(
            f"{path}: expected adapter {adapter!r}, got {row['adapter']!r}"
        )
    if row["status"] in {"ok", "missing_breakdown"} and row["wall_s"] is None:
        raise ValueError(f"{path}: {row['status']} requires Run seconds (wall_s)")
    return row


class JournalUsageError(ValueError):
    """The supplied arguments cannot recover or reuse a wall-time journal."""


def reusable_rows(
    source: Path, plan: BatchPlan, available: dict[str, str]
) -> dict[str, Row]:
    """Read resolved rows for a new plan with the same commit and input hashes.

    Paths in the retained rows refer to the original artifacts. Raise
    JournalUsageError for different provenance; missing or malformed journals
    propagate errors.
    """
    previous = read_plan(source)
    if previous["commit"] != plan["commit"]:
        raise JournalUsageError(
            "resume source commit differs from the current Git HEAD"
        )
    if previous["inputs"] != plan["inputs"]:
        raise JournalUsageError(
            "resume source input SHA-256 differs from the current inputs"
        )
    rows: dict[str, Row] = {}
    for scene_id, scene in previous["scenes"].items():
        adapter = scene["adapter"]
        row = _read_row(source / scene["directory"], scene_id, adapter)
        if row is not None and _resolved(row) and available.get(scene_id) == adapter:
            if (artifact := row["case_directory"]) is not None:
                row["case_directory"] = (source.resolve() / artifact).as_posix()
            row["reused"] = True
            rows[scene_id] = row
    return rows


def _resolved(row: Row) -> bool:
    return row["status"] not in {"incomplete", "not_run", "missing_adapter"}


def assemble_report(output: Path, plan: BatchPlan) -> bool:
    """Render both reports from the batch journal in output.

    Return whether every planned row is resolved. Unmatched starts do not
    establish a Run duration. Malformed or unreadable journal data propagate.
    """
    rows = []
    for scene_id, scene in plan["scenes"].items():
        directory, adapter = scene["directory"], scene["adapter"]
        location = output / directory
        row = _read_row(location, scene_id, adapter)
        if row is None:
            row = _uncommitted_row(scene_id, adapter, directory, location)
        rows.append(row)
    resolved = sum(_resolved(row) for row in rows)
    _write_report(output, rows, plan, resolved)
    return resolved == len(rows)


def _uncommitted_row(
    scene_id: str,
    adapter: str,
    directory: str,
    location: Path,
) -> Row:
    try:
        data = (location / "start.json").read_bytes()
    except FileNotFoundError:
        return new_row(scene_id, adapter, None, "not_run", None)
    start = TypeAdapter(RunStart).validate_json(data)
    row = new_row(scene_id, adapter, directory, "incomplete", None)
    row.update(
        started_at=start["started_at"],
        cold_start=start["cold_start"],
    )
    return row


def _format_names(names: Sequence[str]) -> str:
    return ", ".join(f"`{name}`" for name in names) or "—"
