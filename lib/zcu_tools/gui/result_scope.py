"""Result-scope discovery and params.json identity management.

A result scope is the first-level project root for one chip/qubit pair: the
directory that owns ``params.json`` and contains the ExperimentManager contexts.
This module is Qt-free and owns the only legacy params.json identity parser.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

PARAMS_SCHEMA_VERSION = 1
UNKNOWN_PROJECT_NAME = "unknown"
UNKNOWN_RESONATOR_NAME = "unknown"


@dataclass(frozen=True)
class ResultScope:
    scope_id: str
    chip_name: str
    qub_name: str
    result_dir: str
    params_path: str
    source: Literal["discovered", "generated"]


@dataclass(frozen=True)
class ProjectPaths:
    result_dir: str
    database_path: str
    params_path: str


class ResultScopeError(RuntimeError):
    """Expected result-scope failure with a stable reason code."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _clean_name(value: object) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _identity_from_project(raw: Mapping[str, Any]) -> tuple[str, str] | None:
    project = raw.get("project")
    if not isinstance(project, Mapping):
        return None
    chip = _clean_name(project.get("chip_name"))
    qub = _clean_name(project.get("qubit_name")) or _clean_name(project.get("qub_name"))
    if chip and qub:
        return chip, qub
    return None


def _identity_from_legacy_name(raw: Mapping[str, Any]) -> tuple[str, str] | None:
    name = _clean_name(raw.get("name"))
    if not name:
        return None
    parts = [part.strip() for part in name.split("/")]
    if len(parts) == 2 and all(parts):
        return parts[0], parts[1]
    return None


def _identity_from_result_path(
    params_path: str | Path, result_root: str | Path
) -> tuple[str, str]:
    """Infer project identity from a params.json path under result/.

    This is intentionally narrow: a normal two-level scope is chip/qubit; a
    single-level legacy scope uses the same name for chip and qubit; anything
    else is kept explicit as unknown instead of guessing from an arbitrary tree.
    """

    try:
        rel_dir = (
            Path(params_path).resolve().parent.relative_to(Path(result_root).resolve())
        )
    except ValueError:
        return UNKNOWN_PROJECT_NAME, UNKNOWN_PROJECT_NAME
    parts = rel_dir.parts
    if len(parts) == 2 and all(parts):
        return parts[0], parts[1]
    if len(parts) == 1 and parts[0]:
        return parts[0], parts[0]
    return UNKNOWN_PROJECT_NAME, UNKNOWN_PROJECT_NAME


def _load_params_object(path: Path, *, action: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultScopeError(
            f"Failed to {action} params identity at {path}: {exc}",
            reason_code="params_read_failed",
        ) from exc
    if not isinstance(loaded, dict):
        raise ResultScopeError(
            f"params.json at {path} must contain a JSON object",
            reason_code="params_not_object",
        )
    return loaded


def _install_project_identity(
    raw: dict[str, Any],
    *,
    chip_name: str,
    qub_name: str,
    resonator_name: str = UNKNOWN_RESONATOR_NAME,
) -> None:
    project = raw.get("project")
    if not isinstance(project, dict):
        project = {}
    project["chip_name"] = chip_name
    project["qubit_name"] = qub_name
    project.setdefault("resonator_name", resonator_name)
    raw["schema_version"] = PARAMS_SCHEMA_VERSION
    raw["project"] = project
    raw["name"] = f"{chip_name}/{qub_name}"


def _write_params_object(path: Path, raw: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=4)
        f.write("\n")


def migrate_params_v0_to_v1_project_info(
    params_path: str | Path,
    *,
    result_root: str | Path,
) -> tuple[str, str]:
    """In-place upgrade params.json to schema v1 project identity.

    Existing canonical ``project`` identity wins. When it is missing, identity is
    inferred from the params path under result/: ``chip/qubit`` for two levels,
    ``name/name`` for one level, and ``unknown/unknown`` otherwise. The migration
    preserves unrelated sections such as fluxdep_fit and dispersive.
    """

    path = Path(params_path)
    raw = _load_params_object(path, action="migrate")
    identity = _identity_from_project(raw)
    if identity is None:
        identity = _identity_from_result_path(path, result_root)
    chip_name, qub_name = identity
    _install_project_identity(raw, chip_name=chip_name, qub_name=qub_name)
    _write_params_object(path, raw)
    return chip_name, qub_name


def read_params_identity(params_path: str | Path) -> tuple[str, str]:
    """Read chip/qubit identity from params.json.

    The canonical form is ``project.{chip_name, qubit_name}``. The only supported
    legacy fallback is ``name == "chip/qubit"``; remove
    ``_identity_from_legacy_name`` when the migration window ends.
    """

    path = Path(params_path)
    raw = _load_params_object(path, action="read")
    identity = _identity_from_project(raw) or _identity_from_legacy_name(raw)
    if identity is None:
        raise ResultScopeError(
            f"params.json at {path} has no project identity",
            reason_code="params_missing_identity",
        )
    return identity


def write_params_identity(
    params_path: str | Path,
    *,
    chip_name: str,
    qub_name: str,
    resonator_name: str = UNKNOWN_RESONATOR_NAME,
) -> None:
    """Create or migrate params.json to the canonical project identity shape."""

    path = Path(params_path)
    raw: dict[str, Any] = {}
    if path.exists():
        raw = _load_params_object(path, action="migrate")

    _install_project_identity(
        raw,
        chip_name=chip_name,
        qub_name=qub_name,
        resonator_name=resonator_name,
    )
    _write_params_object(path, raw)


class ResultScopeManager:
    """Locate existing result scopes and create generated ones."""

    def __init__(self, project_root: str | Path) -> None:
        self._project_root = Path(project_root).resolve()

    @property
    def project_root(self) -> Path:
        return self._project_root

    @property
    def result_root(self) -> Path:
        return self._project_root / "result"

    def derive_paths(self, chip_name: str, qub_name: str) -> ProjectPaths:
        from zcu_tools.utils.datasaver import get_datafolder_path

        result_dir = self.result_root / chip_name / qub_name
        database_path = get_datafolder_path(
            str(self._project_root / "Database"), str(Path(chip_name) / qub_name)
        )
        return ProjectPaths(
            result_dir=str(result_dir),
            database_path=database_path,
            params_path=str(result_dir / "params.json"),
        )

    def list_scopes(self) -> tuple[ResultScope, ...]:
        if not self.result_root.exists():
            logger.debug("result scope list: result_root=%s missing", self.result_root)
            return ()
        scopes: list[ResultScope] = []
        for params_path in sorted(self.result_root.rglob("params.json")):
            try:
                chip, qub = migrate_params_v0_to_v1_project_info(
                    params_path, result_root=self.result_root
                )
            except ResultScopeError as exc:
                logger.warning("Skipping result scope %s: %s", params_path, exc)
                continue
            result_dir = params_path.parent.resolve()
            scopes.append(
                ResultScope(
                    scope_id=str(result_dir),
                    chip_name=chip,
                    qub_name=qub,
                    result_dir=str(result_dir),
                    params_path=str(params_path.resolve()),
                    source="discovered",
                )
            )
        logger.debug(
            "result scope list: result_root=%s count=%d", self.result_root, len(scopes)
        )
        return tuple(scopes)

    def list_chip_names(self) -> tuple[str, ...]:
        return tuple(sorted({scope.chip_name for scope in self.list_scopes()}))

    def list_qub_names(self, chip_name: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    scope.qub_name
                    for scope in self.list_scopes()
                    if scope.chip_name == chip_name
                }
            )
        )

    def scopes_for(self, chip_name: str, qub_name: str) -> tuple[ResultScope, ...]:
        return tuple(
            scope
            for scope in self.list_scopes()
            if scope.chip_name == chip_name and scope.qub_name == qub_name
        )

    def ensure_scope(
        self,
        *,
        chip_name: str,
        qub_name: str,
        scope_id: str | None = None,
    ) -> ResultScope:
        if scope_id:
            scope = self._require_discovered_scope(scope_id)
            if (scope.chip_name, scope.qub_name) != (chip_name, qub_name):
                raise ResultScopeError(
                    f"Result scope {scope_id!r} belongs to "
                    f"{scope.chip_name}/{scope.qub_name}, not {chip_name}/{qub_name}",
                    reason_code="scope_identity_mismatch",
                )
            write_params_identity(
                scope.params_path, chip_name=scope.chip_name, qub_name=scope.qub_name
            )
            return scope

        paths = self.derive_paths(chip_name, qub_name)
        params_path = Path(paths.params_path)
        if params_path.exists():
            actual_chip, actual_qub = migrate_params_v0_to_v1_project_info(
                params_path, result_root=self.result_root
            )
            if (actual_chip, actual_qub) != (chip_name, qub_name):
                raise ResultScopeError(
                    f"Generated params path {params_path} belongs to "
                    f"{actual_chip}/{actual_qub}, not {chip_name}/{qub_name}",
                    reason_code="scope_identity_mismatch",
                )
        write_params_identity(params_path, chip_name=chip_name, qub_name=qub_name)
        result_dir = Path(paths.result_dir).resolve()
        return ResultScope(
            scope_id=str(result_dir),
            chip_name=chip_name,
            qub_name=qub_name,
            result_dir=str(result_dir),
            params_path=str(params_path.resolve()),
            source="generated",
        )

    def _require_discovered_scope(self, scope_id: str) -> ResultScope:
        wanted = str(Path(scope_id).resolve())
        for scope in self.list_scopes():
            if scope.scope_id == wanted:
                return scope
        raise ResultScopeError(
            f"Unknown result scope id: {scope_id!r}",
            reason_code="scope_not_found",
        )
