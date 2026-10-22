"""Opened entry and selected fluxonium component for standalone analysis apps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.aliases import AliasRow, AliasTable
from zcu_tools.resources.entry import ResultEntry, snapshot


def fluxonium_components(entry: ResultEntry) -> tuple[str, ...]:
    """List setup fluxonium qubits in document order, without changing the entry."""
    return tuple(
        name
        for name, model in snapshot(entry.setup).components.items()
        if model.kind == "qubit/fluxonium" or model.kind.startswith("qubit/fluxonium/")
    )


@dataclass(frozen=True)
class AnalysisTarget:
    """Existing entry and concrete setup target; analysis never performs Save.

    result_root is the absolute Result root; entry is its single-segment name.
    component is a validated fluxonium qubit name. resource owns setup, predictor
    paths and analysis ledger. Its Database binding is unused by analysis.
    """

    result_root: Path
    entry: str
    component: str
    resource: ResultEntry

    @classmethod
    def open(
        cls,
        project_root: str | Path,
        entry: str,
        component: str | None = None,
    ) -> AnalysisTarget:
        """Open an existing entry, defaulting to its setup qubit alias.

        project_root contains result/ and Database/. Database is only bound,
        never read or written. Missing entry propagates FileNotFoundError.
        Missing/unbound aliases preserve AliasTable errors. Unknown components
        and non-fluxonium kinds raise InvalidInputError before publication.
        """
        project = Path(project_root).resolve()
        root = project / "result"
        resource = ResultEntry.open(
            entry, result_root=root, database_root=project / "Database"
        )
        document = snapshot(resource.setup)
        if component is None:
            aliases = resource.load_aliases().aliases
            component = AliasTable(
                [AliasRow(k, v) for k, v in aliases.items()]
            ).resolve("qubit", document)
        if component not in document.components:
            raise InvalidInputError(
                f"Component {component!r} does not exist in entry {entry!r}",
                reason_code="component_not_found",
            )
        if component not in fluxonium_components(resource):
            raise InvalidInputError(
                f"Component {component!r} must be a fluxonium qubit, got "
                f"{document.components[component].kind!r}",
                reason_code="component_kind",
            )
        return cls(root, entry, component, resource)

    @property
    def predictor_path(self) -> Path:
        """Predictor JSON destination for the selected component; creates nothing."""
        return self.resource.predictors_path / f"{self.component}.json"


class AnalysisTargetPayload(TypedDict):
    """Opened entry identity and paths; every field is null before opening one."""

    result_root: str | None
    entry: str | None
    component: str | None
    result_path: str | None
    predictors_path: str | None


def target_payload(target: AnalysisTarget | None) -> AnalysisTargetPayload:
    """Project current target identity and paths without reading files."""
    if target is None:
        return {
            "result_root": None,
            "entry": None,
            "component": None,
            "result_path": None,
            "predictors_path": None,
        }
    return {
        "result_root": str(target.result_root),
        "entry": target.entry,
        "component": target.component,
        "result_path": str(target.resource.result_path),
        "predictors_path": str(target.resource.predictors_path),
    }


def require_target(target: AnalysisTarget | None) -> AnalysisTarget:
    """Return the opened target, or reject an analysis write with no_entry."""
    if target is None:
        raise FailedPreconditionError(
            "Open an entry and select a component first", reason_code="no_entry"
        )
    return target
