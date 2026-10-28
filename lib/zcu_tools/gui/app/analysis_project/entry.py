"""Opened entry and selected fluxonium component for standalone analysis apps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.aliases import AliasRow, AliasTable
from zcu_tools.resources.entry import ResultEntry, snapshot

from .selection import AnalysisSelection


def fluxonium_components(entry: ResultEntry) -> tuple[str, ...]:
    """List setup fluxonium qubits in document order, without changing the entry."""
    return tuple(
        name
        for name, model in snapshot(entry.setup).components.items()
        if model.kind == "qubit/fluxonium" or model.kind.startswith("qubit/fluxonium/")
    )


@dataclass(frozen=True)
class AnalysisEntry:
    """Opened entry with legal choices, before choosing an analysis target.

    Analysis binds the real Database root but never reads or saves Database.
    Missing or invalid qubit aliases do not prevent explicit component choice.
    """

    result_root: Path
    entry: str
    resource: ResultEntry

    @classmethod
    def open(cls, project_root: str | Path, entry: str) -> AnalysisEntry:
        """Open an existing entry and require at least one fluxonium component."""
        project = Path(project_root).resolve()
        root = project / "result"
        resource = ResultEntry.open(
            entry, result_root=root, database_root=project / "Database"
        )
        opened = cls(root, entry, resource)
        if not opened.components:
            raise InvalidInputError(
                f"Entry {entry!r} has no fluxonium components",
                reason_code="no_fluxonium_components",
            )
        return opened

    @staticmethod
    def list_names(project_root: str | Path) -> tuple[str, ...]:
        """List existing entries under the same project-root convention as open."""
        return tuple(
            ResultEntry.list_names(result_root=Path(project_root).resolve() / "result")
        )

    @property
    def components(self) -> tuple[str, ...]:
        """Legal fluxonium choices in setup order, without any default selection."""
        return fluxonium_components(self.resource)

    def select(self, component: str | None = None) -> AnalysisTarget:
        """Validate an explicit name, or resolve the qubit alias when omitted.

        Missing/unbound aliases preserve AliasTable errors. Unknown components
        and non-fluxonium kinds raise InvalidInputError before publication.
        No default is substituted for an absent or invalid alias.
        """
        document = snapshot(self.resource.setup)
        if component is None:
            aliases = self.resource.load_aliases().aliases
            component = AliasTable(
                [AliasRow(k, v) for k, v in aliases.items()]
            ).resolve("qubit", document)
        if component not in document.components:
            raise InvalidInputError(
                f"Component {component!r} does not exist in entry {self.entry!r}",
                reason_code="component_not_found",
            )
        if component not in self.components:
            raise InvalidInputError(
                f"Component {component!r} must be a fluxonium qubit, got "
                f"{document.components[component].kind!r}",
                reason_code="component_kind",
            )
        return AnalysisTarget(self.result_root, self.entry, component, self.resource)


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
        selection: AnalysisSelection,
    ) -> AnalysisTarget:
        """Open an existing entry, defaulting to its setup qubit alias.

        selection names the project root, entry and optional component.
        An empty entry is invalid here; GUI composition leaves it unopened.
        Database is only bound,
        never read or written. Missing entry propagates FileNotFoundError.
        Missing/unbound aliases preserve AliasTable errors. Unknown components
        and non-fluxonium kinds raise InvalidInputError before publication.
        """
        return AnalysisEntry.open(selection.project_root, selection.entry).select(
            selection.component
        )

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
