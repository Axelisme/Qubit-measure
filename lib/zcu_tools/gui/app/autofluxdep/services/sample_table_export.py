"""Export autofluxdep run artifacts to flat SampleTable v2 CSV files.

The authoritative device unit is snapshotted once at run creation into the
manifest (``workflow.flux.unit``); export never infers, defaults, or re-reads
live device state. Journal ``flux_value`` passes through unchanged as
``dev_value`` with the snapshot ``dev_unit``. AutoFlux has no point calibration
frame, so no ``flux``/``flux_int``/``flux_period`` columns are fabricated.
Rows are the successfully committed flux points (one per "flux_committed"
journal event, in order): for each point, node values are read from the row's
journal patch (user-set keys) and row summary (node-computed keys), so the
export is a pure function of the run artifact.

Bare sweeps, missing devices, unknown/unsupported units, and legacy empty-unit
artifacts are permitted to run under existing policy but cannot be exported:
every such failure happens before any CSV mutation. Appends proceed only when
both the existing table and the candidate rows validate as flat v2;
``append=False`` overwrites the file with a fresh v2 export.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from zcu_tools.gui.app.autofluxdep.services.run_store import (
    load_journal_events,
    load_manifest,
)
from zcu_tools.meta_tool.sample_schema import (
    DEV_UNIT_COLUMN,
    DEV_VALUE_COLUMN,
    validate_sample_table_v2,
)
from zcu_tools.meta_tool.table import SampleTable

DATE_COLUMN = "date"

SAMPLE_COLUMNS: tuple[str, ...] = (
    DEV_VALUE_COLUMN,
    DEV_UNIT_COLUMN,
    "Freq (MHz)",
    "T1 (us)",
    "T1err (us)",
    "T2r (us)",
    "T2r err (us)",
    "T2e (us)",
    "T2e err (us)",
    DATE_COLUMN,
)

_FLUX_UNITS = frozenset({"A", "V"})

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
    *,
    append: bool = True,
) -> SampleTableExportResult:
    """Write a flat SampleTable v2 CSV from a run artifact.

    ``path`` accepts either the run directory or its ``manifest.json``. The
    device unit is the run-creation snapshot; live device state is never read.
    By default, existing valid v2 CSV files are preserved and the exported rows
    are appended; any unit/table validation failure occurs before the CSV is
    touched, so a failed export never mutates existing files.
    """

    manifest_path = resolve_run_manifest(path)
    manifest = load_manifest(manifest_path)
    _validate_terminal_status(manifest)
    dev_unit = _require_manifest_flux_unit(manifest)
    events = load_journal_events(_journal_path(manifest, manifest_path))
    rows = sample_rows_from_journal(
        events,
        dev_unit=dev_unit,
        date=_sample_date(),
    )
    output = (
        Path(filepath) if filepath is not None else default_sample_table_path(manifest)
    )
    _write_sample_rows(rows, output, append=append)
    return SampleTableExportResult(path=str(output), row_count=len(rows))


def sample_rows_from_journal(
    events: Sequence[Mapping[str, Any]],
    *,
    dev_unit: str,
    date: str | None = None,
) -> list[dict[str, float | str]]:
    """Build flat v2 sample rows from committed flux points in journal events.

    ``dev_unit`` is the run-creation snapshot unit and applies unchanged to
    every exported row; journal ``flux_value`` is written unchanged as
    ``dev_value``. No optional ``flux``/``flux_int``/``flux_period`` columns
    are fabricated — AutoFlux has no calibration frame.
    """

    if dev_unit not in _FLUX_UNITS:
        raise ValueError(
            f"sample table export requires an authoritative A/V unit, got {dev_unit!r}"
        )
    committed_flux_points = _committed_flux_points(events)
    if not committed_flux_points:
        raise ValueError("autofluxdep run has no completed flux points to export")

    rows_by_flux: dict[int, dict[str, float | str]] = {
        flux_idx: _sample_row_seed(flux_value, dev_unit, date)
        for flux_idx, flux_value in committed_flux_points
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

    return [rows_by_flux[idx] for idx, _ in committed_flux_points]


def _committed_flux_points(
    events: Sequence[Mapping[str, Any]],
) -> tuple[tuple[int, float], ...]:
    flux_points: list[tuple[int, float]] = []
    seen: set[int] = set()
    for event in events:
        if event.get("type") != "flux_committed":
            continue
        flux_idx = _event_flux_idx(event)
        if flux_idx in seen:
            continue
        seen.add(flux_idx)
        flux_points.append((flux_idx, _event_flux_value(event)))
    return tuple(flux_points)


def _apply_patch_values(
    row: dict[str, float | str],
    patch: Mapping[str, Any] | None,
) -> None:
    if patch is None:
        return
    for key, column in _PATCH_SAMPLE_KEYS:
        _set_sample_value(row, column, patch.get(key))


def _apply_row_summary(
    row: dict[str, float | str],
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


def _set_sample_value(row: dict[str, float | str], column: str, value: Any) -> None:
    if column in row:
        return
    number = _finite_number(value)
    if number is not None:
        row[column] = number


def _sample_row_seed(
    flux_value: float,
    dev_unit: str,
    date: str | None,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        DEV_VALUE_COLUMN: flux_value,
        DEV_UNIT_COLUMN: dev_unit,
    }
    if date is not None:
        row[DATE_COLUMN] = date
    return row


def _write_sample_rows(
    rows: Sequence[Mapping[str, float | str]],
    output: Path,
    *,
    append: bool,
) -> None:
    columns: dict[str, list[float | str | None]] = {}
    for column in SAMPLE_COLUMNS:
        if not any(column in row for row in rows):
            continue
        columns[column] = [row.get(column) for row in rows]
    if not columns:
        raise ValueError("autofluxdep sample export has no columns to write")

    candidate = pd.DataFrame(columns)
    validate_sample_table_v2(candidate, allow_empty=True)
    if output.exists():
        if not output.is_file():
            raise IsADirectoryError(f"sample table output is not a file: {output}")
        if append:
            validate_sample_table_v2(_load_existing_table(output), allow_empty=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not append:
        output.unlink()
    table = SampleTable(output)
    table.extend_samples(**columns)


def _load_existing_table(output: Path) -> pd.DataFrame:
    if output.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(output)


def _validate_terminal_status(manifest: Mapping[str, Any]) -> None:
    terminal = _require_mapping(manifest.get("terminal"), "manifest terminal")
    status = terminal.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("autofluxdep manifest is missing terminal.status")
    if status in _NON_TERMINAL_STATUSES:
        raise ValueError(
            f"cannot export sample table from non-terminal autofluxdep run: {status}"
        )


def _require_manifest_flux_unit(manifest: Mapping[str, Any]) -> str:
    """Return the run-snapshot authoritative A/V unit or fail before mutation."""

    workflow = _require_mapping(manifest.get("workflow"), "manifest workflow")
    flux = _require_mapping(workflow.get("flux"), "manifest workflow.flux")
    unit = flux.get("unit")
    if isinstance(unit, str) and unit in _FLUX_UNITS:
        return unit
    raise ValueError(
        "autofluxdep run has no authoritative A/V flux unit; bare, missing, "
        "unsupported, or legacy empty-unit artifacts cannot export a flat-v2 "
        "sample table"
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


def _sample_date() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _event_flux_idx(event: Mapping[str, Any]) -> int:
    flux_idx = event.get("flux_idx")
    if not isinstance(flux_idx, int):
        raise ValueError(
            f"autofluxdep journal event has invalid flux_idx: {flux_idx!r}"
        )
    return flux_idx


def _event_flux_value(event: Mapping[str, Any]) -> float:
    flux_value = _finite_number(event.get("flux_value"))
    if flux_value is None:
        raise ValueError(
            "autofluxdep journal event has invalid flux_value: "
            f"{event.get('flux_value')!r}"
        )
    return flux_value


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
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
