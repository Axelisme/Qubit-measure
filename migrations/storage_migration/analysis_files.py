"""Project legacy analysis sections through the entry fit-file owners."""

import json
from pathlib import Path

from pydantic import JsonValue, ValidationError
from zcu_tools.datafile import JsonObject
from zcu_tools.resources.analysis_source import AnalysisSource
from zcu_tools.resources.dispersive_file import (
    DispersiveFile,
    save_dispersive_file,
)
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.t1_curve_file import (
    T1CurveFile,
    save_t1_curve_file,
)

from .backup import backup_source_path
from .documents import publish_document
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


def _row(
    source: Path,
    key: str,
    target: Path,
    path: str,
    reason: str | None = None,
    *,
    source_keys: tuple[str, ...] | None = None,
) -> KeyMappingItem:
    return KeyMappingItem(
        old_file=source,
        old_key=key,
        new_file=target,
        new_path=path,
        action="pending" if reason else "value",
        source_keys=source_keys,
        reason=reason or "Preserve legacy analysis values through the format owner",
    )


def _source_rows(
    session: MigrationSession,
    source: Path,
    key: str | None,
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
    timestamp_keys = (key, "timestamp") if key is not None else ("timestamp",)
    timestamp_key = ".".join(timestamp_keys)
    reason = (
        None
        if created_at is not None
        else (
            f"No proven timezone-aware ISO 8601 timestamp; original at {original}#{timestamp_key}"
        )
    )
    return {"run_id": None, "created_at": created_at}, [
        _row(
            source,
            timestamp_key,
            target,
            "source.created_at",
            reason,
            source_keys=timestamp_keys,
        )
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
    without a file. Successful rows identify transferred source subtrees through
    literal source_keys, including supported empty containers. Section summaries
    without source_keys do not claim that unknown nested fields were transferred.
    The caller retains only unconverted values in ext, not a second full copy.
    Timestamp provenance uses only proven_created_at evidence;
    run_id stays null. Transition-only input creates a dispersive file with no
    invented g/bare_rf. Earlier root allows supplies the annotations unchanged
    when fluxdep_fit.plot_transitions is absent; its timestamp comes from the root.
    No transition names or formulas are reinterpreted. Dry-run validates/reports
    without writes. An unfinished
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
    transition_keys = (
        ("fluxdep_fit", "plot_transitions")
        if isinstance(fit, dict) and "plot_transitions" in fit
        else ("allows",)
        if "allows" in values
        else None
    )
    transitions = (
        fit["plot_transitions"]
        if isinstance(fit, dict)
        and transition_keys == ("fluxdep_fit", "plot_transitions")
        else values.get("allows")
    )
    if "dispersive" in values or transition_keys is not None:
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
            if transition_keys is not None:
                params["plot_transitions"] = transitions
            provenance_key = (
                "dispersive"
                if "dispersive" in values
                else "fluxdep_fit"
                if transition_keys == ("fluxdep_fit", "plot_transitions")
                else None
            )
            provenance_raw = (
                raw
                if provenance_key == "dispersive"
                else fit
                if provenance_key == "fluxdep_fit"
                else values
            )
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
                    publish_document(
                        session,
                        source,
                        target,
                        "dispersive",
                        lambda path: save_dispersive_file(path, document),
                    )
                rows.extend(evidence_rows)
                if "dispersive" in values:
                    rows.append(
                        _row(
                            source,
                            "dispersive",
                            target,
                            "params",
                            source_keys=("dispersive",) if not raw else None,
                        )
                    )
                    for key in ("g", "bare_rf"):
                        if key in raw:
                            rows.append(
                                _row(
                                    source,
                                    f"dispersive.{key}",
                                    target,
                                    f"params.{key}",
                                    source_keys=("dispersive", key),
                                )
                            )
                if transition_keys is not None:
                    rows.append(
                        _row(
                            source,
                            ".".join(transition_keys),
                            target,
                            "params.plot_transitions",
                            source_keys=transition_keys,
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
                publish_document(
                    session,
                    source,
                    target,
                    "t1_curve",
                    lambda path: save_t1_curve_file(path, t1_document),
                )
            rows.extend(evidence_rows)
            rows.append(
                _row(
                    source,
                    "t1_curve_fit",
                    target,
                    "params",
                    source_keys=("t1_curve_fit",) if not raw else None,
                )
            )
            rows.extend(
                _row(
                    source,
                    f"t1_curve_fit.{key}",
                    target,
                    f"params.{key}",
                    source_keys=("t1_curve_fit", key),
                )
                for key in params
            )
    return rows
