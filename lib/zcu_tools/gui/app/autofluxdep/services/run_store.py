"""Run Result Artifact owner for autofluxdep sweeps."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from zcu_tools.gui.app.autofluxdep.nodes.builder import PlacedNode
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch
from zcu_tools.gui.app.autofluxdep.nodes.result import QubitFreqResult
from zcu_tools.gui.app.autofluxdep.orchestrator import InfoStore, SkipReason
from zcu_tools.gui.app.autofluxdep.services.fluxdep_export import (
    export_qubit_freq_fluxdep_spectrum,
)
from zcu_tools.gui.app.autofluxdep.services.result_io import (
    result_role_specs,
    result_row_role_names,
    write_result_row,
)
from zcu_tools.gui.app.autofluxdep.services.run_report import write_markdown_report
from zcu_tools.gui.app.autofluxdep.state import ProjectInfo
from zcu_tools.utils.datasaver import (
    LabberMetadata,
    StreamingGroupedLabberWriter,
    open_streaming_grouped_labber_data,
)

MANIFEST_FORMAT_VERSION = 1
JOURNAL_EVENT_VERSION = 1
ARTIFACT_KIND = "zcu_tools.autofluxdep.run"


class RunStore:
    """Owns all persisted state for one autofluxdep run artifact."""

    def __init__(
        self,
        *,
        run_dir: Path,
        data_dir: Path,
        run_id: str,
        project: ProjectInfo,
        flux_values: Sequence[float],
        flux_device_name: str | None,
        nodes: Sequence[PlacedNode],
        results: Mapping[str, object],
    ) -> None:
        # ``run_dir`` is the lightweight metadata root under result_dir.
        self.run_dir = run_dir
        # ``data_dir`` is the heavy Labber HDF5 root under database_path.
        self.data_dir = data_dir
        self.run_id = run_id
        self._project = project
        self._flux_values = [float(value) for value in flux_values]
        self._nodes = list(nodes)
        self._results = dict(results)
        self._writers: dict[str, StreamingGroupedLabberWriter] = {}
        self._node_file_by_name: dict[str, str] = {}
        self._node_type_by_name = {node.name: node.type_name for node in self._nodes}
        self._seq = 0
        self._created_at = _utc_now()
        self._journal_path = self.run_dir / "journal.jsonl"
        self._manifest_path = self.run_dir / "manifest.json"
        self._row_counts: Counter[str] = Counter()
        self._skip_counts: Counter[str] = Counter()
        self._failure_counts: Counter[str] = Counter()
        self._exports: dict[str, str] = {}
        self._reports: dict[str, str] = {}
        self._terminal_errors: list[str] = []
        self._manifest = self._initial_manifest(flux_device_name)

    @classmethod
    def create(
        cls,
        *,
        project: ProjectInfo,
        flux_values: Sequence[float],
        flux_device_name: str | None,
        nodes: Sequence[PlacedNode],
        results: Mapping[str, object],
    ) -> RunStore:
        """Create the run directory, manifest, journal, and node writers."""
        if project is None:
            raise RuntimeError("autofluxdep Run Result Artifact requires a project")
        run_id = _run_id()
        run_suffix = run_id.rsplit("-", maxsplit=1)[-1]
        slug = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_flux-sweep-{run_suffix}"
        run_dir = Path(project.result_dir) / "autofluxdep_runs" / slug
        data_dir = Path(project.database_path) / "autofluxdep_runs" / slug
        run_dir.mkdir(parents=True, exist_ok=False)
        data_dir.mkdir(parents=True, exist_ok=False)
        (data_dir / "nodes").mkdir()
        (data_dir / "exports" / "fluxdep").mkdir(parents=True)
        store = cls(
            run_dir=run_dir,
            data_dir=data_dir,
            run_id=run_id,
            project=project,
            flux_values=flux_values,
            flux_device_name=flux_device_name,
            nodes=nodes,
            results=results,
        )
        try:
            store._journal_path.write_text("", encoding="utf-8")
            store._open_node_writers()
            store._write_manifest()
        except Exception:
            store.close_writers()
            raise
        return store

    def write_node_row(
        self,
        provider_name: str,
        flux_idx: int,
        patch: Patch,
        info: InfoStore,
    ) -> None:
        """Commit one completed node row to HDF5 and journal."""
        del info
        result = self._results.get(provider_name)
        if result is None:
            roles_written: tuple[str, ...] = ()
        else:
            roles_written = result_row_role_names(result, flux_idx)
        patch_values = _json_safe(
            patch.values(), subject="patch values", nonfinite_to_none=False
        )
        provided_modules = _json_safe_modules(patch.modules())
        row_summary = _json_safe(
            _row_summary(result, int(flux_idx)),
            subject="row summary",
            nonfinite_to_none=True,
        )
        provide_status = (
            "empty_patch" if not patch.values() and not patch.modules() else "provided"
        )
        payload = _json_safe(
            {
                "flux_idx": int(flux_idx),
                "flux_value": self._flux_values[int(flux_idx)],
                "node": provider_name,
                "node_type": self._node_type_by_name.get(provider_name, provider_name),
                "result_file": self._node_file_by_name.get(provider_name),
                "roles_written": list(roles_written),
                "measurement_status": "completed",
                "provide_status": provide_status,
                "patch": patch_values,
                "provided_modules": provided_modules,
                "row_summary": row_summary,
            },
            subject="node_row_written payload",
            nonfinite_to_none=False,
        )
        if result is not None:
            writer = self._writers[provider_name]
            roles_written = write_result_row(
                writer,
                provider_name,
                self._node_type_by_name.get(provider_name, provider_name),
                result,
                flux_idx,
                timestamp=time.time(),
            )
            writer.flush()
        self._row_counts[provider_name] += 1
        self._append_event("node_row_written", payload)

    def record_node_skipped(
        self,
        provider_name: str,
        flux_idx: int,
        reason: SkipReason,
    ) -> None:
        self._skip_counts[provider_name] += 1
        self._append_event(
            "node_skipped",
            {
                "flux_idx": int(flux_idx),
                "flux_value": self._flux_values[int(flux_idx)],
                "node": provider_name,
                "node_type": self._node_type_by_name.get(provider_name, provider_name),
                "reason": {
                    "missing_info_keys": list(reason.missing_info_keys),
                    "missing_modules": list(reason.missing_modules),
                },
            },
        )

    def record_node_failed(
        self,
        provider_name: str,
        flux_idx: int,
        exc: Exception,
        stage: str,
        *,
        row_committed: bool = False,
    ) -> None:
        self._failure_counts[provider_name] += 1
        self._append_event(
            "node_failed",
            {
                "flux_idx": int(flux_idx),
                "flux_value": self._flux_values[int(flux_idx)],
                "node": provider_name,
                "node_type": self._node_type_by_name.get(provider_name, provider_name),
                "stage": stage,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "row_committed": bool(row_committed),
            },
        )

    def commit_flux(self, flux_idx: int, flux: float, info: InfoStore) -> None:
        self._append_event(
            "flux_committed",
            {
                "flux_idx": int(flux_idx),
                "flux_value": float(flux),
                "node_rows_written": sum(
                    1
                    for event in self.iter_journal_events()
                    if event.get("type") == "node_row_written"
                    and event.get("flux_idx") == int(flux_idx)
                ),
                "nodes_skipped": sum(
                    1
                    for event in self.iter_journal_events()
                    if event.get("type") == "node_skipped"
                    and event.get("flux_idx") == int(flux_idx)
                ),
                "info_keys": sorted(str(key) for key in info.point),
            },
        )

    def record_run_failed(
        self,
        exc: Exception,
        *,
        flux_idx: int | None = None,
        node: str | None = None,
        stage: str | None = None,
    ) -> None:
        self._append_event(
            "run_failed",
            {
                "flux_idx": flux_idx,
                "node": node,
                "stage": stage,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
        )

    def finalize(
        self,
        status: str,
        *,
        error: Exception | None = None,
    ) -> None:
        """Close writers, generate terminal sidecars, and finalize manifest."""
        self.close_writers(finalize=True)
        terminal = self._manifest["terminal"]
        terminal["status"] = status
        terminal["finalized_at"] = _utc_now()
        terminal["error"] = str(error) if error is not None else None
        self._manifest["terminal"] = terminal
        export_errors: list[str] = []
        report_errors: list[str] = []
        try:
            self._generate_exports()
        except Exception as exc:
            export_errors.append(str(exc))
        self._manifest["exports"] = dict(self._exports)
        self._reports["markdown"] = "report.md"
        self._manifest["reports"] = dict(self._reports)
        self._append_event(
            "run_finalized",
            {
                "terminal_status": status,
                "completed_flux_count": self._completed_flux_count(),
                "node_row_count": int(sum(self._row_counts.values())),
                "skip_count": int(sum(self._skip_counts.values())),
                "failure_count": int(sum(self._failure_counts.values())),
                "exports": dict(self._exports),
                "reports": dict(self._reports),
                "export_errors": export_errors,
                "report_errors": report_errors,
            },
        )
        try:
            self._generate_report()
        except Exception as exc:
            report_errors.append(str(exc))
            self._reports.pop("markdown", None)
        self._terminal_errors.extend(export_errors)
        self._terminal_errors.extend(report_errors)
        terminal_error = str(error) if error is not None else None
        if self._terminal_errors:
            suffix = "; ".join(self._terminal_errors)
            terminal_error = f"{terminal_error}; {suffix}" if terminal_error else suffix
        terminal["error"] = terminal_error
        self._manifest["updated_at"] = _utc_now()
        self._manifest["exports"] = dict(self._exports)
        self._manifest["reports"] = dict(self._reports)
        self._manifest["terminal"] = terminal
        self._write_manifest()
        if export_errors or report_errors:
            raise RuntimeError("; ".join(export_errors + report_errors))

    def close_writers(self, *, finalize: bool = False) -> None:
        for writer in self._writers.values():
            if finalize:
                try:
                    writer.finalize()
                except RuntimeError:
                    pass
            writer.close()

    def iter_journal_events(self) -> list[dict[str, Any]]:
        return list(load_journal_events(self._journal_path))

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    def _open_node_writers(self) -> None:
        node_files: list[dict[str, Any]] = []
        for index, node in enumerate(self._nodes):
            result = self._results.get(node.name)
            if result is None:
                continue
            filename = f"{index:03d}-{_safe_slug(node.name)}.hdf5"
            relpath = f"nodes/{filename}"
            specs = result_role_specs(node.name, node.type_name, result)
            writer = open_streaming_grouped_labber_data(
                str(self.data_dir / relpath),
                specs,
                metadata=LabberMetadata(
                    comment=f"autofluxdep run {self.run_id} node {node.name}",
                    project=f"{self._project.chip_name}/{self._project.qub_name}",
                ),
            )
            self._writers[node.name] = writer
            self._node_file_by_name[node.name] = relpath
            node_files.append(
                {
                    "name": node.name,
                    "type": node.type_name,
                    "path": relpath,
                    "roles": [str(spec.role) for spec in specs],
                }
            )
        self._manifest["files"]["nodes"] = node_files

    def _initial_manifest(self, flux_device_name: str | None) -> dict[str, Any]:
        nodes = []
        workflow_payload: list[dict[str, Any]] = []
        for index, node in enumerate(self._nodes):
            cfg_raw = _json_safe(node.schema.to_persisted_raw(), subject="node cfg")
            cfg_hash = _sha256_json(cfg_raw)
            entry = {
                "index": index,
                "name": node.name,
                "type": node.type_name,
                "cfg_hash": f"sha256:{cfg_hash}",
            }
            nodes.append(entry)
            workflow_payload.append({**entry, "cfg": cfg_raw})
        workflow_hash = _sha256_json(workflow_payload)
        return {
            "format_version": MANIFEST_FORMAT_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "run_id": self.run_id,
            "created_at": self._created_at,
            "updated_at": self._created_at,
            "project": {
                "chip_name": self._project.chip_name,
                "qub_name": self._project.qub_name,
                "result_dir": self._project.result_dir,
                "database_path": self._project.database_path,
                "params_path": self._project.params_path,
            },
            "paths": {
                "metadata_root": str(self.run_dir),
                "data_root": str(self.data_dir),
                "journal": "journal.jsonl",
            },
            "workflow": {
                "workflow_hash": f"sha256:{workflow_hash}",
                "workflow_snapshot_version": 1,
                "nodes": nodes,
                "flux": {
                    "device_name": flux_device_name,
                    "values": list(self._flux_values),
                    "unit": "",
                },
            },
            "files": {"journal": "journal.jsonl", "nodes": []},
            "exports": {},
            "reports": {},
            "terminal": {
                "status": "running",
                "finalized_at": None,
                "error": None,
            },
        }

    def _append_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        self._seq += 1
        event = {
            "event_version": JOURNAL_EVENT_VERSION,
            "seq": self._seq,
            "run_id": self.run_id,
            "time": _utc_now(),
            "type": event_type,
            **dict(payload),
        }
        line = json.dumps(event, sort_keys=True, separators=(",", ":"))
        with self._journal_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _write_manifest(self) -> None:
        tmp_path = self._manifest_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(self._manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, self._manifest_path)

    def _generate_exports(self) -> None:
        for node_name, result in self._results.items():
            if not isinstance(result, QubitFreqResult):
                continue
            committed_mask = self._committed_node_row_mask(node_name, result.n_flux)
            if not committed_mask.any():
                continue
            relpath = "exports/fluxdep/qubit_freq.hdf5"
            written = export_qubit_freq_fluxdep_spectrum(
                result,
                self.data_dir / relpath,
                committed_mask=committed_mask,
            )
            self._exports["fluxdep_spectrum"] = _relative(self.data_dir, Path(written))
            break

    def _generate_report(self) -> None:
        relpath = "report.md"
        written = write_markdown_report(
            self.run_dir / relpath,
            self._manifest,
            self.iter_journal_events(),
        )
        self._reports["markdown"] = _relative(self.run_dir, Path(written))

    def _committed_node_row_mask(self, node_name: str, n_flux: int) -> np.ndarray:
        mask = np.zeros(int(n_flux), dtype=np.bool_)
        for event in self.iter_journal_events():
            if event.get("type") != "node_row_written":
                continue
            if event.get("node") != node_name:
                continue
            flux_idx = int(event.get("flux_idx", -1))
            if 0 <= flux_idx < mask.shape[0]:
                mask[flux_idx] = True
        return mask

    def _completed_flux_count(self) -> int:
        return sum(
            1
            for event in self.iter_journal_events()
            if event.get("type") == "flux_committed"
        )


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    version = int(manifest.get("format_version", 0))
    if version != MANIFEST_FORMAT_VERSION:
        raise ValueError(f"unsupported autofluxdep manifest format_version {version}")
    return manifest


def load_journal_events(path: str | Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        version = int(event.get("event_version", 0))
        if version != JOURNAL_EVENT_VERSION:
            raise ValueError(f"unsupported autofluxdep journal event_version {version}")
        events.append(event)
    return events


def _row_summary(result: object | None, flux_idx: int) -> dict[str, Any]:
    if result is None:
        return {}
    summary: dict[str, Any] = {}
    for attr in (
        "fit_freq",
        "predict_freq",
        "snr",
        "fit_value",
        "best_freq",
        "best_gain",
    ):
        values = getattr(result, attr, None)
        if values is None:
            continue
        value = np.asarray(values).reshape(-1)[flux_idx]
        summary[attr] = None if not np.isfinite(value) else float(value)
    return summary


def _json_safe_modules(modules: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(name): _json_safe(
            module,
            subject=f"provided module {name!r}",
            nonfinite_to_none=False,
        )
        for name, module in modules.items()
    }


def _json_safe(value: Any, *, subject: str, nonfinite_to_none: bool = True) -> Any:
    try:
        converted = _to_json_tree(value, nonfinite_to_none=nonfinite_to_none)
    except TypeError as exc:
        raise TypeError(f"{subject} is not strict JSON-safe: {exc}") from exc
    try:
        encoded = json.dumps(converted, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{subject} is not strict JSON-safe: {exc}") from exc
    return json.loads(encoded)


def _to_json_tree(value: Any, *, nonfinite_to_none: bool) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            if nonfinite_to_none:
                return None
            raise TypeError("non-finite float is not strict JSON")
        return value
    if isinstance(value, np.generic):
        return _to_json_tree(value.item(), nonfinite_to_none=nonfinite_to_none)
    if isinstance(value, np.ndarray):
        return [
            _to_json_tree(item, nonfinite_to_none=nonfinite_to_none)
            for item in value.tolist()
        ]
    if isinstance(value, Mapping):
        return {
            str(key): _to_json_tree(item, nonfinite_to_none=nonfinite_to_none)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _to_json_tree(item, nonfinite_to_none=nonfinite_to_none) for item in value
        ]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_json_tree(
            model_dump(mode="json", exclude_none=True),
            nonfinite_to_none=nonfinite_to_none,
        )
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _to_json_tree(to_dict(), nonfinite_to_none=nonfinite_to_none)
    raise TypeError(f"object of type {type(value).__name__} is not JSON-safe")


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _run_id() -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "node"


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


__all__ = [
    "ARTIFACT_KIND",
    "JOURNAL_EVENT_VERSION",
    "MANIFEST_FORMAT_VERSION",
    "RunStore",
    "load_journal_events",
    "load_manifest",
]
