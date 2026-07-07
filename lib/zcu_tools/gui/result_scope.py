"""Result-scope discovery and params.json identity management.

A result scope is the first-level project root for one chip/qubit pair: the
directory that owns ``params.json`` and contains the ExperimentManager contexts.
This module is Qt-free; params.json parsing and migration policy lives in
``meta_tool.QubitParams``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from zcu_tools.meta_tool import (
    UNKNOWN_RESONATOR_NAME,
    ParamsProject,
    QubitParams,
    QubitParamsError,
)

logger = logging.getLogger(__name__)


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


def _params_error(
    exc: QubitParamsError | OSError, *, action: str, path: Path
) -> ResultScopeError:
    reason_code = (
        exc.reason_code if isinstance(exc, QubitParamsError) else "params_read_failed"
    )
    return ResultScopeError(
        f"Failed to {action} params identity at {path}: {exc}",
        reason_code=reason_code,
    )


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
    try:
        project = QubitParams(path).migrate_project_from_path(result_root=result_root)
    except (OSError, QubitParamsError) as exc:
        raise _params_error(exc, action="migrate", path=path) from exc
    return project.chip_name, project.qub_name


def read_params_identity(params_path: str | Path) -> tuple[str, str]:
    """Read chip/qubit identity from params.json.

    The canonical form is ``project.{chip_name, qubit_name}``. The only supported
    legacy fallback is ``name == "chip/qubit"``; remove
    ``_identity_from_legacy_name`` when the migration window ends.
    """

    path = Path(params_path)
    try:
        project = QubitParams(path, readonly=True).require_project()
    except (OSError, QubitParamsError) as exc:
        raise _params_error(exc, action="read", path=path) from exc
    return project.chip_name, project.qub_name


def write_params_identity(
    params_path: str | Path,
    *,
    chip_name: str,
    qub_name: str,
    resonator_name: str = UNKNOWN_RESONATOR_NAME,
) -> None:
    """Create or migrate params.json to the canonical project identity shape."""

    path = Path(params_path)
    try:
        QubitParams(path).ensure_project(
            ParamsProject(
                chip_name=chip_name,
                qub_name=qub_name,
                resonator_name=resonator_name,
            )
        )
    except (OSError, QubitParamsError) as exc:
        raise _params_error(exc, action="write", path=path) from exc


class ResultScopeManager:
    """Locate existing result scopes and create generated ones."""

    def __init__(self, project_root: str | Path) -> None:
        self._project_root = Path(project_root).resolve()
        self._scope_cache: tuple[ResultScope, ...] | None = None

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

    def list_scopes(self, *, refresh: bool = False) -> tuple[ResultScope, ...]:
        if self._scope_cache is not None and not refresh:
            return self._scope_cache
        if not self.result_root.exists():
            logger.debug("result scope list: result_root=%s missing", self.result_root)
            self._scope_cache = ()
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
        self._scope_cache = tuple(scopes)
        return self._scope_cache

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
        self._scope_cache = None
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
        had_cache = self._scope_cache is not None
        for scope in self.list_scopes():
            if scope.scope_id == wanted:
                return scope
        if had_cache:
            for scope in self.list_scopes(refresh=True):
                if scope.scope_id == wanted:
                    return scope
        raise ResultScopeError(
            f"Unknown result scope id: {scope_id!r}",
            reason_code="scope_not_found",
        )
