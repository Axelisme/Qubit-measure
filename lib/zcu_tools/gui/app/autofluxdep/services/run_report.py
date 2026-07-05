"""Terminal markdown report for an autofluxdep Run Result Artifact."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def write_markdown_report(
    filepath: str | Path,
    manifest: Mapping[str, Any],
    journal_events: Iterable[Mapping[str, Any]],
) -> str:
    """Write the terminal-only run report sidecar."""
    text = render_markdown_report(manifest, journal_events)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def render_markdown_report(
    manifest: Mapping[str, Any],
    journal_events: Iterable[Mapping[str, Any]],
) -> str:
    events = list(journal_events)
    counts = Counter(str(event.get("type", "")) for event in events)
    node_rows = Counter(
        str(event.get("node", ""))
        for event in events
        if event.get("type") == "node_row_written"
    )
    node_skips = Counter(
        str(event.get("node", ""))
        for event in events
        if event.get("type") == "node_skipped"
    )
    node_failures = Counter(
        str(event.get("node", ""))
        for event in events
        if event.get("type") == "node_failed"
    )
    terminal = dict(manifest.get("terminal", {}))
    files = dict(manifest.get("files", {}))
    paths = dict(manifest.get("paths", {}))
    workflow = dict(manifest.get("workflow", {}))
    flux = dict(workflow.get("flux", {}))
    lines = [
        f"# Autofluxdep Run {manifest.get('run_id', '')}",
        "",
        f"- Status: {terminal.get('status', '')}",
        f"- Created: {manifest.get('created_at', '')}",
        f"- Finalized: {terminal.get('finalized_at', '')}",
        f"- Project: {_project_label(manifest)}",
        f"- Flux points: {len(flux.get('values', ()))}, device: {flux.get('device_name')}",
        f"- Workflow hash: {workflow.get('workflow_hash', '')}",
        "",
        "## Artifact Paths",
        "",
        f"- Metadata root: {paths.get('metadata_root', '')}",
        f"- Data root: {paths.get('data_root', '')}",
        "- Manifest: manifest.json",
        f"- Journal: {files.get('journal', 'journal.jsonl')}",
    ]
    for node_file in files.get("nodes", ()):
        lines.append(f"- Node {node_file.get('name')}: {node_file.get('path')}")
    for label, path in dict(manifest.get("exports", {})).items():
        lines.append(f"- Export {label}: {path}")
    for label, path in dict(manifest.get("reports", {})).items():
        lines.append(f"- Report {label}: {path}")

    lines.extend(
        [
            "",
            "## Node Summary",
            "",
            "| Node | Rows | Skips | Failures |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    node_names = _node_names(manifest, node_rows, node_skips, node_failures)
    for name in node_names:
        lines.append(
            f"| {name} | {node_rows[name]} | {node_skips[name]} | {node_failures[name]} |"
        )

    lines.extend(
        [
            "",
            "## Event Summary",
            "",
            f"- Node rows written: {counts['node_row_written']}",
            f"- Node skipped: {counts['node_skipped']}",
            f"- Node failed: {counts['node_failed']}",
            f"- Flux committed: {counts['flux_committed']}",
            f"- Run failed events: {counts['run_failed']}",
            f"- Run finalized events: {counts['run_finalized']}",
        ]
    )
    if terminal.get("error"):
        lines.extend(["", "## Terminal Error", "", str(terminal["error"])])
    return "\n".join(lines) + "\n"


def _project_label(manifest: Mapping[str, Any]) -> str:
    project = dict(manifest.get("project", {}))
    return f"{project.get('chip_name', '')}/{project.get('qub_name', '')}"


def _node_names(
    manifest: Mapping[str, Any],
    *counters: Counter[str],
) -> list[str]:
    names = [
        str(node.get("name", ""))
        for node in dict(manifest.get("workflow", {})).get("nodes", ())
    ]
    seen = set(names)
    for counter in counters:
        for name in counter:
            if name and name not in seen:
                names.append(name)
                seen.add(name)
    return names


__all__ = ["render_markdown_report", "write_markdown_report"]
