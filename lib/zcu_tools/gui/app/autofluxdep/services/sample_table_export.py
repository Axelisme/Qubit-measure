"""Export autofluxdep run artifacts to notebook-style SampleTable CSV files."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zcu_tools.gui.app.autofluxdep.services.run_store import (
    load_journal_events,
    load_manifest,
)
from zcu_tools.meta_tool.table import SampleTable

SAMPLE_COLUMNS: tuple[str, ...] = (
    "Freq (MHz)",
    "T1 (us)",
    "T1err (us)",
    "T2r (us)",
    "T2r err (us)",
    "T2e (us)",
    "T2e err (us)",
)

_PATCH_SAMPLE_KEYS: tuple[tuple[str, str], ...] = (
    ("qubit_freq", "Freq (MHz)"),
    ("q_f", "Freq (MHz)"),
    ("freq", "Freq (MHz)"),
    ("t1", "T1 (us)"),
    ("t1err", "T1err (us)"),
    ("t1_err", "T1err (us)"),
    ("t2r", "T2r (us)"),
    ("t2rerr", "T2r err (us)"),
    ("t2r_err", "T2r err (us)"),
    ("t2e", "T2e (us)"),
    ("t2eerr", "T2e err (us)"),
    ("t2e_err", "T2e err (us)"),
)

_SUMMARY_SAMPLE_FIELDS: Mapping[str, tuple[str, str]] = {
    "qubit_freq": ("Freq (MHz)", "fit_freq"),
    "t1": ("T1 (us)", "fit_value"),
    "t2ramsey": ("T2r (us)", "fit_value"),
    "t2echo": ("T2e (us)", "fit_value"),
}

_NON_TERMINAL_STATUSES = {"running", "paused"}


@dataclass(frozen=True)
class SampleTableExportResult:
    path: str
    row_count: int


def resolve_run_manifest(path: str | Path) -> Path:
    """Resolve a run directory or manifest path to ``manifest.json``."""

    candidate = Path(path)
    if candidate.is_dir():
        manifest_path = candidate / "manifest.json"
        if manifest_path.exists():
            return manifest_path
        paired_manifest = _find_paired_metadata_manifest(candidate)
        if paired_manifest is not None:
            return paired_manifest
        candidate = manifest_path
    if not candidate.exists():
        raise FileNotFoundError(
            "autofluxdep manifest not found; pass the metadata run directory, "
            f"manifest.json, or a paired data run directory: {candidate}"
        )
    if candidate.is_dir():
        raise IsADirectoryError(
            f"autofluxdep manifest path is a directory: {candidate}"
        )
    return candidate


def _find_paired_metadata_manifest(data_run_dir: Path) -> Path | None:
    """Find metadata manifest for a heavy data-root run directory, if possible."""

    if data_run_dir.parent.name != "autofluxdep_runs":
        return None
    search_root = _paired_search_root(data_run_dir)
    if search_root is None:
        return None

    slug = data_run_dir.name
    for manifest_path in search_root.rglob(f"autofluxdep_runs/{slug}/manifest.json"):
        manifest = load_manifest(manifest_path)
        paths = _require_mapping(manifest.get("paths"), "manifest paths")
        manifest_data_root = paths.get("data_root")
        if not isinstance(manifest_data_root, str):
            continue
        if Path(manifest_data_root).resolve() == data_run_dir.resolve():
            return manifest_path
    return None


def _paired_search_root(run_dir: Path) -> Path | None:
    for parent in run_dir.parents:
        if parent.name == "Database":
            return parent.parent
    return None


def default_sample_table_path(manifest_or_path: Mapping[str, Any] | str | Path) -> Path:
    """Return the deterministic sample CSV path for a run artifact."""

    manifest = (
        load_manifest(resolve_run_manifest(manifest_or_path))
        if isinstance(manifest_or_path, (str, Path))
        else manifest_or_path
    )
    paths = _require_mapping(manifest.get("paths"), "manifest paths")
    data_root = paths.get("data_root")
    if not isinstance(data_root, str) or not data_root:
        raise ValueError("autofluxdep manifest is missing paths.data_root")
    return Path(data_root) / "exports" / "sample" / "samples.csv"


def export_sample_table_from_artifact(
    path: str | Path,
    filepath: str | Path | None = None,
) -> SampleTableExportResult:
    """Write a notebook-style ``SampleTable`` CSV from a run artifact.

    ``path`` accepts either the run directory or its ``manifest.json``. The output
    CSV is replaced if it already exists, so repeated exports of the same artifact
    stay deterministic instead of appending duplicate sample rows.
    """

    manifest_path = resolve_run_manifest(path)
    manifest = load_manifest(manifest_path)
    _validate_terminal_status(manifest)
    events = load_journal_events(_journal_path(manifest, manifest_path))
    rows = sample_rows_from_journal(events)
    output = (
        Path(filepath) if filepath is not None else default_sample_table_path(manifest)
    )
    _write_sample_rows(rows, output)
    return SampleTableExportResult(path=str(output), row_count=len(rows))


def sample_rows_from_journal(
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, float]]:
    """Build notebook sample rows from committed flux points in journal events."""

    committed_flux_indices = _committed_flux_indices(events)
    if not committed_flux_indices:
        raise ValueError("autofluxdep run has no completed flux points to export")

    rows_by_flux: dict[int, dict[str, float]] = {
        flux_idx: {} for flux_idx in committed_flux_indices
    }
    for event in events:
        if event.get("type") != "node_row_written":
            continue
        flux_idx = _event_flux_idx(event)
        if flux_idx not in rows_by_flux:
            continue
        row = rows_by_flux[flux_idx]
        _apply_patch_values(row, _optional_mapping(event.get("patch")))
        _apply_row_summary(
            row,
            node=str(event.get("node", "")),
            node_type=str(event.get("node_type", "")),
            summary=_optional_mapping(event.get("row_summary")),
        )

    rows = [rows_by_flux[idx] for idx in committed_flux_indices if rows_by_flux[idx]]
    if not rows:
        raise ValueError(
            "autofluxdep run has completed flux points but no sample table fields"
        )
    return rows


def _committed_flux_indices(events: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    flux_indices: list[int] = []
    seen: set[int] = set()
    for event in events:
        if event.get("type") != "flux_committed":
            continue
        flux_idx = _event_flux_idx(event)
        if flux_idx in seen:
            continue
        seen.add(flux_idx)
        flux_indices.append(flux_idx)
    return tuple(flux_indices)


def _apply_patch_values(
    row: dict[str, float],
    patch: Mapping[str, Any] | None,
) -> None:
    if patch is None:
        return
    for key, column in _PATCH_SAMPLE_KEYS:
        _set_sample_value(row, column, patch.get(key))


def _apply_row_summary(
    row: dict[str, float],
    *,
    node: str,
    node_type: str,
    summary: Mapping[str, Any] | None,
) -> None:
    if summary is None:
        return
    for candidate in (node_type, node):
        spec = _SUMMARY_SAMPLE_FIELDS.get(candidate)
        if spec is None:
            continue
        column, summary_key = spec
        _set_sample_value(row, column, summary.get(summary_key))
        return


def _set_sample_value(row: dict[str, float], column: str, value: Any) -> None:
    if column in row:
        return
    number = _finite_number(value)
    if number is not None:
        row[column] = number


def _write_sample_rows(rows: Sequence[Mapping[str, float]], output: Path) -> None:
    columns: dict[str, list[float | None]] = {}
    for column in SAMPLE_COLUMNS:
        if not any(column in row for row in rows):
            continue
        columns[column] = [row.get(column) for row in rows]
    if not columns:
        raise ValueError("autofluxdep sample export has no columns to write")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not output.is_file():
            raise IsADirectoryError(f"sample table output is not a file: {output}")
        output.unlink()
    table = SampleTable(output)
    table.extend_samples(**columns)


def _validate_terminal_status(manifest: Mapping[str, Any]) -> None:
    terminal = _require_mapping(manifest.get("terminal"), "manifest terminal")
    status = terminal.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("autofluxdep manifest is missing terminal.status")
    if status in _NON_TERMINAL_STATUSES:
        raise ValueError(
            f"cannot export sample table from non-terminal autofluxdep run: {status}"
        )


def _journal_path(manifest: Mapping[str, Any], manifest_path: Path) -> Path:
    paths = _require_mapping(manifest.get("paths"), "manifest paths")
    metadata_root = paths.get("metadata_root")
    journal = paths.get("journal", "journal.jsonl")
    if not isinstance(journal, str) or not journal:
        raise ValueError("autofluxdep manifest is missing paths.journal")
    journal_path = Path(journal)
    if journal_path.is_absolute():
        return journal_path
    if isinstance(metadata_root, str) and metadata_root:
        return Path(metadata_root) / journal_path
    return manifest_path.parent / journal_path


def _event_flux_idx(event: Mapping[str, Any]) -> int:
    flux_idx = event.get("flux_idx")
    if not isinstance(flux_idx, int):
        raise ValueError(
            f"autofluxdep journal event has invalid flux_idx: {flux_idx!r}"
        )
    return flux_idx


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _require_mapping(value: Any, subject: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"autofluxdep {subject} must be a mapping")
    return value


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"autofluxdep journal field must be a mapping, got {value!r}")
    return value
