"""Ordered session-only aliases and their reads against a fixed point snapshot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING

from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.resources.entry import RoleResolutionError, document_value

if TYPE_CHECKING:
    from zcu_tools.resources.entry import RoleSpec
    from zcu_tools.resources.entry.schema import PointDocument


@dataclass(frozen=True)
class AliasRow:
    """One authored row in the ordered global alias table.

    alias is a public identifier, such as qubit or resonator. value is a
    concrete component or device identifier, an empty string for unbound, or a
    dotted reference beginning with $, such as $qubit.resonator. References may name
    only an earlier row. AliasTable validates names, values and ordering as
    one batch; constructing a row alone does not validate a table.
    """

    alias: str
    value: str


@dataclass(frozen=True)
class ResolvedAliasRow:
    """One row projected from a fixed capture without consumer kind checks.

    alias and value preserve the authored name and binding in table order.
    resolved is the concrete component name on success, otherwise None.
    error is None on success or the RoleResolutionError diagnostic on failure.
    Context source failures are not represented as rows; the whole read raises.
    """

    alias: str
    value: str
    resolved: str | None
    error: str | None = None


class AliasTable:
    """Own an ordered in-memory table; read links only when an alias is used.

    Construction and replace validate the complete authored table without
    reading components. InvalidInputError(alias_invalid) leaves the old table
    unchanged. Reads use only the supplied fixed PointDocument, not a provider,
    file, registry, or hardware. Empty bindings remain legal until read. No
    persistence, default policy or event publication belongs to this owner.
    """

    def __init__(self, rows: Sequence[AliasRow] = ()) -> None:
        """Install validated rows in order; duplicate or invalid rows raise."""
        self._rows: tuple[AliasRow, ...] = ()
        self.replace(rows)

    @property
    def rows(self) -> tuple[AliasRow, ...]:
        """Return immutable authored rows, including unexpanded $ references."""
        return self._rows

    def replace(self, rows: Sequence[AliasRow]) -> None:
        """Atomically replace all rows after syntax and backward-link checks.

        Names and dotted reference segments must be public identifiers. Empty
        values are unbound. A $ reference needs an alias and at least one field;
        it may refer only to a preceding row. Concrete names are not checked
        against point or device catalogs; replacing rows needs no snapshot.
        Raises InvalidInputError(alias_invalid) without modifying the old rows.
        """
        candidate = tuple(rows)
        preceding: set[str] = set()
        for row in candidate:
            if not _public_identifier(row.alias) or row.alias in preceding:
                raise InvalidInputError(
                    f"Invalid or duplicate alias {row.alias!r}",
                    reason_code="alias_invalid",
                )
            if row.value.startswith("$"):
                parts = row.value[1:].split(".")
                if (
                    len(parts) < 2
                    or not all(_public_identifier(part) for part in parts)
                    or parts[0] not in preceding
                ):
                    raise InvalidInputError(
                        f"Alias {row.alias!r} must reference an earlier row's field",
                        reason_code="alias_invalid",
                    )
            elif row.value and not _public_identifier(row.value):
                raise InvalidInputError(
                    f"Invalid component name for alias {row.alias!r}: {row.value!r}",
                    reason_code="alias_invalid",
                )
            preceding.add(row.alias)
        self._rows = candidate

    def with_override(self, alias: str, component: str) -> AliasTable:
        """Return an independent one-use table with this concrete binding.

        Replace an existing row in place or append a previously absent alias.
        Dependent $ rows retain their authored links. Empty or malformed
        component/device identifiers raise InvalidInputError(alias_invalid).
        Field readability and optional consumer kind are checked by resolve.
        The original table is never changed.
        """
        if not _public_identifier(component):
            raise InvalidInputError(
                "A temporary target must be a nonempty component identifier",
                reason_code="alias_invalid",
            )
        rows = [
            AliasRow(row.alias, component) if row.alias == alias else row
            for row in self._rows
        ]
        if not any(row.alias == alias for row in self._rows):
            rows.append(AliasRow(alias, component))
        return AliasTable(rows)

    def resolve(
        self, alias: str, point: PointDocument, *, kind: str | None = None
    ) -> str:
        """Read an alias's component or device name from this fixed point.

        Concrete names need not belong to point.components or a device catalog.
        Follow only this row's earlier-row dependency chain. kind optionally
        requires a component matching a fnmatch pattern for component consumers.
        Unknown aliases, unbound links, unreadable fields, non-name link
        targets and wrong kinds raise RoleResolutionError naming the requested
        alias. No same-named concrete component substitutes for an alias.
        """
        name = self._resolve_name(alias, point, kind=kind)
        if name is None:
            raise RoleResolutionError(alias, None, kind, "Alias is not bound")
        return name

    def names(
        self, point: PointDocument, declarations: Mapping[str, RoleSpec]
    ) -> dict[str, str]:
        """Capture all bound rows as concrete names for Run/storage records.

        Empty dependency chains are omitted. Consumer declarations constrain
        their rows' kinds; other authored rows still contribute resolved names.
        Nonempty invalid links and wrong kinds raise RoleResolutionError.
        Returned names are independent data; no declaration.via inference or
        concrete-name fallback is performed.
        """
        names: dict[str, str] = {}
        for row in self._rows:
            spec = declarations.get(row.alias)
            name = self._resolve_name(
                row.alias, point, kind=None if spec is None else spec.kind
            )
            if name is not None:
                names[row.alias] = name
        return names

    def _resolve_name(
        self, alias: str, point: PointDocument, *, kind: str | None = None
    ) -> str | None:
        authored = {row.alias: row.value for row in self._rows}
        if alias not in authored:
            raise RoleResolutionError(alias, None, None, "Unknown alias")
        # Validated links always move to an earlier row, so no cycle machinery is needed.
        links: list[tuple[str, str]] = []
        current = alias
        value = authored[current]
        while value.startswith("$"):
            dependency, suffix = value[1:].split(".", 1)
            links.append((current, suffix))
            current = dependency
            value = authored[current]
        if not value:
            return None
        name = value
        for referring_alias, suffix in reversed(links):
            try:
                target = document_value(point, f"{name}.{suffix}")
            except AttributeError as cause:
                raise RoleResolutionError(
                    alias,
                    None,
                    None,
                    f"Unreadable link for {referring_alias!r}: {name}.{suffix}",
                ) from cause
            if not isinstance(target, str) or not _public_identifier(target):
                raise RoleResolutionError(
                    alias,
                    None,
                    None,
                    f"Link for {referring_alias!r} must name a component or device",
                )
            name = target
        component = point.components.get(name)
        if kind is not None and (
            component is None or not fnmatchcase(component.kind, kind)
        ):
            raise RoleResolutionError(
                alias, None, kind, f"Alias targets wrong-kind component {name!r}"
            )
        return name


def _public_identifier(text: str) -> bool:
    return text.isidentifier() and not text.startswith("_")
