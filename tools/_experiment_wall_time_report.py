"""Persist the wall-time batch plan and render reports from committed row facts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict

from pydantic import TypeAdapter

RUN_TARGET_S = 180.0


class BatchPlan(TypedDict):
    """The revision, fixed input hashes, case locations and guide inventory.

    cases maps selected adapters to directories relative to the batch output.
    adapters maps all registered names to their declared guide, or None.
    guides lists the discovered operator guides.
    """

    commit: str
    inputs: dict[str, str]
    cases: dict[str, str]
    adapters: dict[str, str | None]
    guides: list[str]


class RunStart(TypedDict):
    """The UTC ISO-8601 and monotonic seconds sampled before Run dispatch."""

    started_at: str
    monotonic_s: float
    cold_start: bool


class BatchEnd(TypedDict):
    """The UTC ISO-8601 and monotonic seconds at the batch interruption.

    monotonic_s is None when the end was only observed by the outer runner.
    """

    ended_at: str
    monotonic_s: float | None


type RowStatus = Literal[
    "ok",
    "missing_breakdown",
    "missing_defaults",
    "not_runnable",
    "not_run",
    "incomplete",
    "timeout",
]


class Row(TypedDict):
    """One report row and its measured or unavailable Run facts.

    adapter and guide identify the experiment and its declared operator guide.
    case_directory is relative to the output (or absolute for retained evidence),
    or None when there are no case artifacts. Timestamps are UTC ISO-8601 strings;
    wall_s is exact measured seconds, and wall_s_lower_bound is used instead for
    incomplete or timed-out Runs. Breakdown is complete, partial, missing, or
    None if not attempted. Acquires counts observed records; the four stage
    fields are measured seconds or None. Status distinguishes complete, missing
    data/defaults, unsupported, timed-out and uncompleted rows. Reason explains
    an exclusion or unresolved defaults, when applicable. cold_start is true
    for the first dispatch in its process, false for later Runs and None for
    unstarted rows. reused is true when this batch retained another batch's row.
    """

    adapter: str
    guide: str | None
    case_directory: str | None
    started_at: str | None
    finished_at: str | None
    wall_s: float | None
    wall_s_lower_bound: float | None
    breakdown: Literal["complete", "partial", "missing"] | None
    acquires: int | None
    compile_s: float | None
    evolve_s: float | None
    readout_s: float | None
    remainder_s: float | None
    status: RowStatus
    reason: str | None
    cold_start: bool | None
    reused: bool


def new_row(
    adapter: str,
    guide: str | None,
    directory: str | None,
    status: RowStatus,
    reason: str | None,
) -> Row:
    """Make a row with an identity and classification, before attaching timings."""
    return {
        "adapter": adapter,
        "guide": guide,
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
        "remainder_s": None,
        "status": status,
        "reason": reason,
        "cold_start": None,
        "reused": False,
    }


def _seconds(seconds: float | None) -> str:
    return "—" if seconds is None else f"{seconds:.3f}"


def _summary_row(row: Row) -> str:
    times = " | ".join(
        _seconds(seconds)
        for seconds in (
            row["compile_s"],
            row["evolve_s"],
            row["readout_s"],
            row["remainder_s"],
        )
    )
    status = {
        "ok": "OK",
        "missing_breakdown": "分項缺",
        "missing_defaults": "推不出預設",
        "not_runnable": "FTF 不能跑",
        "not_run": "not_run",
        "incomplete": "incomplete",
        "timeout": "timeout",
    }[row["status"]]
    if row["status"] == "timeout":
        status += "（部分）" if row["breakdown"] == "partial" else "（分項缺）"
    if row["reason"] is not None:
        reason = " ".join(row["reason"].splitlines()).replace("|", r"\|")
        status = f"{status}: {reason}"
    guide = f"`{row['guide']}`" if row["guide"] else "—"
    acquires = "—" if row["acquires"] is None else str(row["acquires"])
    lower_bound = row["wall_s_lower_bound"]
    wall = _seconds(row["wall_s"])
    if lower_bound is not None:
        operator = ">" if row["status"] == "timeout" else "≥"
        wall = f"{operator}{lower_bound}"
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
    return f"| {row['adapter']} | {wall} | {times} | {acquires} | {status} | {guide} | {source} |\n"


def _write_report(output: Path, rows: list[Row], plan: BatchPlan) -> None:
    guides_by_adapter = plan["adapters"]
    guide_paths = plan["guides"]
    no_guide = sorted(
        adapter for adapter, guide in guides_by_adapter.items() if guide is None
    )
    inventoried = set(guide_paths)
    missing = {
        adapter: guide
        for adapter, guide in sorted(guides_by_adapter.items())
        if guide is not None and guide not in inventoried
    }
    no_adapter = sorted(inventoried.difference(guides_by_adapter.values()))
    over_target = [
        row["adapter"]
        for row in rows
        if row["status"] == "timeout"
        or ((wall := row["wall_s"]) is not None and wall > RUN_TARGET_S)
    ]
    resolved = sum(_resolved(row) for row in rows)
    write_json(
        output / "report.json",
        {
            "commit": plan["commit"],
            "inputs": plan["inputs"],
            "rows": rows,
            "coverage": {
                "resolved": resolved,
                "selected": len(rows),
                "registered": len(guides_by_adapter),
            },
            "adapters_without_guides": no_guide,
            "declared_guides_not_in_inventory": missing,
            "guides_without_adapters": no_adapter,
        },
    )
    (output / "summary.md").write_text(
        f"Coverage: {resolved}/{len(rows)} selected rows resolved; "
        f"{len(guides_by_adapter)} registered adapters.\n\n"
        "| Experiment | Run (s) | Compile (s) | Evolve (s) | Readout (s) | Remainder (s) | Acquires | Status | Guide | Source |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |\n"
        + "".join(map(_summary_row, rows))
        + f"\nRun >{RUN_TARGET_S:g} s: {_format_names(over_target)}\n"
        + f"\nAdapters without declared guides: {_format_names(no_guide)}\n"
        + "\nDeclared guides not in inventory: "
        + (
            ", ".join(f"`{adapter}` → `{guide}`" for adapter, guide in missing.items())
            or "—"
        )
        + f"\n\nGuides without adapters: {_format_names(no_adapter)}\n",
        encoding="utf-8",
    )


def write_json(path: Path, value: object) -> None:
    """Atomically replace path with JSON in its existing parent directory."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_plan(output: Path) -> BatchPlan:
    """Load a wall-time journal's plan, raising on absent or malformed data."""
    return TypeAdapter(BatchPlan).validate_json((output / "batch.json").read_bytes())


def read_batch_end(output: Path) -> BatchEnd | None:
    """Return the recorded stop time, or None if no stop has been recorded yet."""
    try:
        data = (output / "batch-end.json").read_bytes()
    except FileNotFoundError:
        return None
    return TypeAdapter(BatchEnd).validate_json(data)


def read_row(location: Path) -> Row | None:
    """Load a committed row from its journal directory, or None if not committed."""
    try:
        data = (location / "row.json").read_bytes()
    except FileNotFoundError:
        return None
    return TypeAdapter(Row).validate_json(data)


def reusable_rows(source: Path, plan: BatchPlan) -> dict[str, Row]:
    """Read resolved rows for a new plan with the same commit and input hashes.

    Paths in the retained rows refer to the original artifacts. Raise ValueError
    for different provenance; missing or malformed journals propagate errors.
    """
    previous = read_plan(source)
    if previous["commit"] != plan["commit"]:
        raise ValueError("resume source commit differs from the current Git HEAD")
    if previous["inputs"] != plan["inputs"]:
        raise ValueError("resume source input SHA-256 differs from the current inputs")
    rows: dict[str, Row] = {}
    for adapter, directory in previous["cases"].items():
        row = read_row(source / directory)
        if row is not None and _resolved(row):
            if adapter not in plan["adapters"]:
                raise ValueError(
                    f"resume source adapter is not registered: {adapter!r}"
                )
            if (artifact := row["case_directory"]) is not None:
                row["case_directory"] = (source.resolve() / artifact).as_posix()
            row["reused"] = True
            rows[adapter] = row
    return rows


def _resolved(row: Row) -> bool:
    return row["status"] not in {"incomplete", "not_run"}


def assemble_report(output: Path, plan: BatchPlan, ended: BatchEnd | None) -> None:
    """Render both reports from disk; ended supplies the bounds for unmatched Starts.

    Raises ValueError if an incomplete Run lacks a batch end or if that end is
    before the persisted start. Malformed or unreadable journal data propagate.
    """
    rows = []
    for adapter, directory in plan["cases"].items():
        location = output / directory
        row = read_row(location)
        if row is None:
            row = _uncommitted_row(
                adapter, plan["adapters"][adapter], directory, location, ended
            )
        rows.append(row)
    _write_report(output, rows, plan)


def _uncommitted_row(
    adapter: str,
    guide: str | None,
    directory: str,
    location: Path,
    ended: BatchEnd | None,
) -> Row:
    try:
        data = (location / "start.json").read_bytes()
    except FileNotFoundError:
        return new_row(adapter, guide, None, "not_run", None)
    start = TypeAdapter(RunStart).validate_json(data)
    if ended is None:
        raise ValueError("an incomplete Run requires --ended-at from the batch receipt")
    if ended["monotonic_s"] is not None:
        elapsed_s = ended["monotonic_s"] - start["monotonic_s"]
    else:
        end = datetime.fromisoformat(ended["ended_at"])
        began = datetime.fromisoformat(start["started_at"])
        elapsed_s = (end - began).total_seconds()
    if elapsed_s < 0:
        raise ValueError(f"batch end precedes Run start for {adapter!r}")
    row = new_row(adapter, guide, directory, "incomplete", None)
    row.update(
        started_at=start["started_at"],
        wall_s_lower_bound=elapsed_s,
        cold_start=start["cold_start"],
    )
    return row


def _format_names(names: Sequence[str]) -> str:
    return ", ".join(f"`{name}`" for name in names) or "—"
