"""Project legacy analysis sections through the entry fit-file owners."""

import json
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, JsonValue, ValidationError
from zcu_tools.datafile import JsonObject
from zcu_tools.resources.analysis_source import AnalysisSource
from zcu_tools.resources.dispersive_file import (
    DispersiveFile,
    load_dispersive_file,
    save_dispersive_file,
)
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.t1_curve_file import (
    T1CurveFile,
    load_t1_curve_file,
    save_t1_curve_file,
)

from .backup import backup_source_path
from .errors import MigrationInputError
from .models import KeyMappingItem
from .state import MigrationSession


def proven_created_at(value: object) -> str | None:
    """Return owner-validated timestamp text, or None if unproven.

    Accept YYYY-MM-DDTHH:MM[:SS[.ffffff]] with Z or +/-HH:MM timezone under
    AnalysisSource validation. Non-strings and invalid timestamps are unknown.
    Never infer a timezone, normalize the text or substitute now.
    """
    if not isinstance(value, str):
        return None
    try:
        return AnalysisSource(run_id=None, created_at=value).created_at
    except ValidationError:
        return None


def _publish[Fit: BaseModel](
    path: Path,
    planned: Fit,
    load: Callable[[Path], Fit],
    save: Callable[[Path, Fit], None],
) -> None:
    if path.is_symlink():
        raise MigrationInputError(f"{path}: analysis file symlink conflict")
    if path.exists():
        try:
            existing = load(path)
        except ValueError as exc:
            raise MigrationInputError(
                f"{path}: analysis file validation failed"
            ) from exc
        # Presence distinguishes an active-but-unknown T1 channel from absence.
        if existing.model_dump(exclude_unset=True) != planned.model_dump(
            exclude_unset=True
        ):
            raise MigrationInputError(
                f"{path}: analysis file differs from planned legacy fit"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    save(path, planned)


def _row(
    source: Path, key: str, target: Path, path: str, reason: str | None = None
) -> KeyMappingItem:
    return KeyMappingItem(
        old_file=source,
        old_key=key,
        new_file=target,
        new_path=path,
        action="pending" if reason else "value",
        reason=reason or "Preserve legacy analysis values through the format owner",
    )


def _source_rows(
    session: MigrationSession,
    source: Path,
    key: str,
    target: Path,
    timestamp: JsonValue,
) -> tuple[JsonObject, list[KeyMappingItem]]:
    created_at = proven_created_at(timestamp)
    backup = session.manifest.report.backup_path
    original = (
        backup_source_path(backup, session.manifest.report.source, source)
        if backup is not None
        else source
    )
    reason = (
        None
        if created_at is not None
        else (
            f"No proven timezone-aware ISO 8601 timestamp; original at {original}#{key}.timestamp"
        )
    )
    return {"run_id": None, "created_at": created_at}, [
        _row(source, f"{key}.timestamp", target, "source.created_at", reason)
    ]


def convert_analysis_files(
    session: MigrationSession,
    entry: ResultEntry | None,
    source: Path,
    name: str,
    values: JsonObject,
) -> list[KeyMappingItem]:
    """Plan/publish dispersive and T1 data for one already resolved component.

    session supplies destination, dry-run and verified-backup evidence; entry is
    its handle (None only in dry-run). source is the baselined legacy params.json,
    name is the unique migrated qubit, and values is the parsed JSON object.
    Format owners validate known content; invalid sections remain located pending
    without a file. Unknown section fields remain in the caller's verbatim ext
    preservation. Timestamp provenance uses only proven_created_at evidence;
    run_id stays null. Transition-only input creates a dispersive file with no
    invented g/bare_rf. Dry-run validates/reports without writes. An unfinished
    resume accepts equal owner-loaded files unchanged and rejects conflicts;
    the parameters checkpoint prevents rewriting completed user-edited files.
    I/O errors propagate. This function does not edit source data or ext.
    """
    root = session.manifest.report.destination.result_path
    dispersive_path = (
        entry.dispersive_path if entry is not None else root / "dispersive"
    )
    t1_path = entry.t1_curve_path if entry is not None else root / "t1_curve"
    rows = _convert_dispersive(
        session, source, dispersive_path / f"{name}.json", values
    )
    if "t1_curve_fit" in values:
        rows.extend(
            _convert_t1(
                session, source, t1_path / f"{name}.json", values["t1_curve_fit"]
            )
        )
    return rows


def _convert_dispersive(
    session: MigrationSession, source: Path, target: Path, values: JsonObject
) -> list[KeyMappingItem]:
    rows: list[KeyMappingItem] = []
    fit = values.get("fluxdep_fit")
    transitions = fit.get("plot_transitions") if isinstance(fit, dict) else None
    if "dispersive" in values or (isinstance(fit, dict) and "plot_transitions" in fit):
        raw = values.get("dispersive", {})
        if not isinstance(raw, dict):
            rows.append(
                _row(
                    source,
                    "dispersive",
                    target,
                    "params",
                    "Content validation requires a dispersive object",
                )
            )
        else:
            params: JsonObject = {
                key: raw[key] for key in ("g", "bare_rf") if key in raw
            }
            if isinstance(fit, dict) and "plot_transitions" in fit:
                params["plot_transitions"] = transitions
            provenance_key = "dispersive" if "dispersive" in values else "fluxdep_fit"
            provenance_raw = raw if provenance_key == "dispersive" else fit
            timestamp = (
                provenance_raw.get("timestamp")
                if isinstance(provenance_raw, dict)
                else None
            )
            provenance, evidence_rows = _source_rows(
                session, source, provenance_key, target, timestamp
            )
            try:
                document = DispersiveFile.model_validate_json(
                    json.dumps(
                        {
                            "format": "zcu.dispersive",
                            "version": 1,
                            "params": params,
                            "source": provenance,
                        }
                    )
                )
            except ValidationError as exc:
                rows.append(
                    _row(
                        source,
                        "dispersive",
                        target,
                        "params",
                        f"Content validation failed: {exc}",
                    )
                )
            else:
                if not session.dry_run:
                    _publish(
                        target, document, load_dispersive_file, save_dispersive_file
                    )
                rows.extend(evidence_rows)
                if "dispersive" in values:
                    rows.append(_row(source, "dispersive", target, "params"))
                if isinstance(fit, dict) and "plot_transitions" in fit:
                    rows.append(
                        _row(
                            source,
                            "fluxdep_fit.plot_transitions",
                            target,
                            "params.plot_transitions",
                        )
                    )
    return rows


def _convert_t1(
    session: MigrationSession, source: Path, target: Path, raw: JsonValue
) -> list[KeyMappingItem]:
    rows: list[KeyMappingItem] = []
    if not isinstance(raw, dict):
        rows.append(
            _row(
                source,
                "t1_curve_fit",
                target,
                "params",
                "Content validation requires a T1 curve object",
            )
        )
    else:
        params = {
            key: value
            for key, value in raw.items()
            if key
            in (
                "params",
                "stderr",
                "bounds",
                "fixed",
                "free",
                "cost",
                "init",
                "reduced_chi2",
                "success",
                "message",
                "residual_mode",
                "loss",
                "max_nfev",
            )
        }
        provenance, evidence_rows = _source_rows(
            session, source, "t1_curve_fit", target, raw.get("timestamp")
        )
        try:
            t1_document = T1CurveFile.model_validate_json(
                json.dumps(
                    {
                        "format": "zcu.t1-curve",
                        "version": 1,
                        "params": params,
                        "source": provenance,
                    }
                )
            )
        except ValidationError as exc:
            rows.append(
                _row(
                    source,
                    "t1_curve_fit",
                    target,
                    "params",
                    f"Content validation failed: {exc}",
                )
            )
        else:
            if not session.dry_run:
                _publish(target, t1_document, load_t1_curve_file, save_t1_curve_file)
            rows.extend(evidence_rows)
            rows.append(_row(source, "t1_curve_fit", target, "params"))
    return rows
