"""Fixed point/library read projections and consumer role bindings."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from fnmatch import fnmatchcase
from types import MappingProxyType
from typing import TYPE_CHECKING

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.aliases import AliasRow, AliasTable, ResolvedAliasRow
from zcu_tools.gui.session.context_types import (
    ContextSourceUnavailableError,
)
from zcu_tools.resources.document_store import DocumentValue
from zcu_tools.resources.entry import (
    RoleResolutionError,
    document_value,
    resolve_path,
)
from zcu_tools.resources.entry.roles import resolve_names
from zcu_tools.resources.entry.storage import PointStorageBinding

_LIBRARY_PARTITIONS = ("pulses", "readout", "reset", "waveforms")

if TYPE_CHECKING:
    from zcu_tools.format_version import YamlValue
    from zcu_tools.gui.session.context_types import ContextSourceFault, ContextTarget
    from zcu_tools.resources.context.library import LibrarySnapshot
    from zcu_tools.resources.entry import RoleSpec
    from zcu_tools.resources.entry.schema import PointDocument


class ModuleSlotError(AttributeError):
    """A point module slot is unset or references a missing library entry.

    The message names the requested slot path and whether it is unset or dangling.
    This retains AttributeError ancestry for parameter readers, but lets expression
    and cfg consumers distinguish invalid input from unexpected source failures.
    """


@dataclass(frozen=True)
class RoleBinding:
    """Fixed role inference inputs and resolved aliases for one capture.

    focus is the captured component name, or None for default-focus inference.
    declarations preserves ordered role kind/via rules. names contains only
    bound role-to-component names; omitted declared roles are unbound.
    explicit preserves manual choices: a component name binds that role, None
    explicitly leaves it unbound, and an absent key allows inference. It is
    process-local capture state, not storage or ledger metadata.
    aliases optionally retains immutable authored alias rows. When present,
    ParameterContext resolves names from these rows rather than focus or via;
    None denotes an already resolved binding without authored alias rows.
    Construction detaches mappings. ParameterContext also detaches returned
    bindings, so consumer edits do not change another capture.
    """

    focus: str | None
    declarations: Mapping[str, RoleSpec]
    names: Mapping[str, str]
    explicit: Mapping[str, str | None] = field(default_factory=dict)
    aliases: tuple[AliasRow, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "declarations", deepcopy(dict(self.declarations)))
        object.__setattr__(self, "names", deepcopy(dict(self.names)))
        object.__setattr__(self, "explicit", dict(self.explicit))


class ParameterContext:
    """An isolated working-unit point/library capture with an explicit target.

    Construction accepts validated point and library snapshots, a SavedPoint
    destination, and an optional fixed consumer RoleBinding.
    Binding declarations may omit names for unbound roles. Their $role reads
    raise RoleResolutionError naming the role; bare paths always name concrete
    components or general fields, for both Run and Load captures.
    source_faults contains only point/module_cfg diagnostics, not setup faults.
    The target, authored alias rows, diagnostics and already resolved bindings
    remain readable on an unavailable capture. Data reads and alias-derived
    names then raise ContextSourceUnavailableError.
    Optional aliases retain authored ordered rows from one environment capture.
    $alias reads return a name string, without requiring component or device
    catalog membership. $alias.path expands rows against this fixed point.
    Consumer declarations constrain kinds but do not infer via/focus bindings.
    Reads do not refresh, read YAML, or consult current selection. Library
    materialization may use its captured waveform asset capability.
    """

    def __init__(
        self,
        *,
        point: PointDocument,
        library: LibrarySnapshot,
        target: ContextTarget,
        roles: RoleBinding | None,
        source_faults: tuple[ContextSourceFault, ...] = (),
        storage: PointStorageBinding | None = None,
    ) -> None:
        """Detach input data and retain the immutable library projection.

        RoleResolutionError reports missing or wrong-kind destinations for
        declared component consumers. Undeclared names may identify devices.
        Declared roles absent from names are unbound, not constructor failures.
        A setup diagnostic is outside this capture contract and raises
        ValueError. storage optionally retains a resource-owned saved-point
        capability with matching target identity; omit it for synthetic
        captures. Mismatched storage origins also raise ValueError.
        roles.aliases optionally supplies fixed authored rows. When supplied,
        roles contributes consumer declarations; names are resolved lazily for $
        reads and role/storage projections. Invalid alias syntax/order raises
        InvalidInputError. Empty aliases do not prevent constructing a capture.
        No session selection generation is accepted or compared.
        """
        if any(fault.document == "setup" for fault in source_faults):
            raise ValueError("Parameter captures accept only point/module_cfg faults")
        if storage is not None and (
            str(storage.entry_id) != target.entry_id or storage.point != target.point
        ):
            raise ValueError("Storage origin must match the saved-point target")
        self._point = point.model_copy(deep=True)
        self._library = library
        self._target = target.model_copy(deep=True)
        self._roles = deepcopy(roles)
        self._source_faults = deepcopy(source_faults)
        self._storage = storage
        self._aliases = (
            AliasTable(self._roles.aliases)
            if self._roles is not None and self._roles.aliases is not None
            else None
        )
        if (
            self._aliases is None
            and self._roles is not None
            and not self._source_faults
        ):
            for role, name in self._roles.names.items():
                if (
                    role in self._roles.declarations
                    and name not in self._point.components
                ):
                    spec = self._roles.declarations.get(role)
                    raise RoleResolutionError(
                        role,
                        self._roles.focus,
                        None if spec is None else spec.kind,
                        f"Captured target {name!r} is missing or has the wrong kind",
                    )
            for role, spec in self._roles.declarations.items():
                if role not in self._roles.names:
                    continue
                name = self._roles.names[role]
                component = self._point.components.get(name)
                if component is None or not fnmatchcase(component.kind, spec.kind):
                    raise RoleResolutionError(
                        role,
                        self._roles.focus,
                        spec.kind,
                        f"Captured target {name!r} is missing or has the wrong kind",
                    )

    @property
    def target(self) -> ContextTarget:
        """Return an independent saved-point destination."""
        return self._target.model_copy(deep=True)

    @property
    def roles(self) -> RoleBinding | None:
        """Return consumer declarations and names from this fixed capture.

        With authored aliases, resolve every bound row now, including custom
        rows outside consumer declarations. Unbound rows are omitted. Invalid
        links/kinds and unavailable sources raise. No
        current global table is read. The returned binding has resolved names,
        not authored rows, so historical target reads never re-expand links.
        None means no consumer declarations.
        """
        if self._aliases is not None and self._roles is not None:
            self._ensure_readable()
            names = self._aliases.names(self._point, self._roles.declarations)
            return RoleBinding(None, self._roles.declarations, names)
        return deepcopy(self._roles)

    @property
    def aliases(self) -> tuple[AliasRow, ...]:
        """Return fixed authored alias rows; empty means none were captured."""
        return () if self._aliases is None else self._aliases.rows

    def resolved_aliases(self) -> tuple[ResolvedAliasRow, ...]:
        """Project every authored row against this capture's fixed point.

        Preserve order and softlink values. No consumer kind declaration limits
        this projection. A RoleResolutionError becomes a row with resolved=None
        and its diagnostic in error; other rows remain readable. Source faults
        raise ContextSourceUnavailableError for the whole read, even with no rows.
        Reads neither live aliases nor files and never materializes assets.
        """
        self._ensure_readable()
        if self._aliases is None:
            return ()
        rows: list[ResolvedAliasRow] = []
        for row in self._aliases.rows:
            try:
                name = self._aliases.resolve(row.alias, self._point)
            except RoleResolutionError as exc:
                rows.append(ResolvedAliasRow(row.alias, row.value, None, str(exc)))
            else:
                rows.append(ResolvedAliasRow(row.alias, row.value, name))
        return tuple(rows)

    @property
    def storage(self) -> PointStorageBinding | None:
        """Return the fixed origin capability, or None for synthetic unbound captures.

        Its roots, ledger and description belong to capture time. This projection
        adds this consumer's fixed role names without consulting session focus.
        Alias-derived names can raise source or role resolution errors.
        No file I/O, refresh or ledger append occurs.
        """
        if self._storage is None:
            return None
        roles = self.roles
        names = {} if roles is None else dict(roles.names)
        return replace(self._storage, roles=MappingProxyType(names))

    def capture_storage(self) -> PointStorageBinding:
        """Return storage at this capture's origin, never the current selection.

        Necessary source faults raise ContextSourceUnavailableError. Synthetic
        captures without an origin capability raise FailedPreconditionError with
        reason_code storage_unavailable. This performs no I/O or ledger append.
        """
        self._ensure_readable()
        storage = self.storage
        if storage is None:
            raise FailedPreconditionError(
                "Capture has no saved-point storage", reason_code="storage_unavailable"
            )
        return storage

    @property
    def source_faults(self) -> tuple[ContextSourceFault, ...]:
        """Return detached point/module_cfg diagnostics, empty when healthy."""
        return deepcopy(self._source_faults)

    @property
    def library(self) -> LibrarySnapshot:
        """Return the fixed read-only library; unavailable sources raise."""
        self._ensure_readable()
        return self._library

    def _ensure_readable(self) -> None:
        if self._source_faults:
            raise ContextSourceUnavailableError(self._source_faults)

    def document(self) -> PointDocument:
        """Copy the fixed point without I/O or model hooks; unavailable sources raise."""
        self._ensure_readable()
        return self._point.model_copy(deep=True)

    def bind_aliases(
        self,
        rows: tuple[AliasRow, ...],
        declarations: Mapping[str, RoleSpec],
    ) -> ParameterContext:
        """Return a fixed capture of authored aliases and consumer kind rules.

        Validate syntax/order, not bindings. Empty or missing rows are legal
        until used; reads report RoleResolutionError naming the alias. Use $
        for aliases and bare paths for concrete components. Declaration.via and
        focus never infer aliases. Point/library/target/storage stay at this
        capture's origin. Unavailable sources and invalid rows propagate.
        """
        self._ensure_readable()
        return ParameterContext(
            point=self._point,
            library=self._library,
            target=self._target,
            roles=RoleBinding(None, declarations, {}, aliases=rows),
            storage=self._storage,
        )

    def bind_roles(
        self,
        declarations: Mapping[str, RoleSpec],
        *,
        focus: str | None,
        explicit: Mapping[str, str | None],
    ) -> ParameterContext:
        """Resolve roles on this fixed point and return an independent capture.

        Declarations are ordered kind/via rules. Explicit concrete names win
        over focus/via inference. None explicitly leaves its role unbound;
        an absent explicit key allows inference. The returned RoleBinding keeps
        the same focus and explicit choices. Empty declarations remain empty.
        Auto inference with zero/multiple candidates leaves roles unbound;
        their use raises RoleResolutionError. Invalid explicit choices and
        dangling/wrong-kind references still raise during binding.
        target and source data are retained; live session state is never read.
        """
        self._ensure_readable()
        names = resolve_names(self._point.components, declarations, focus, explicit)
        return ParameterContext(
            point=self._point,
            library=self._library,
            target=self._target,
            roles=RoleBinding(focus, declarations, names, explicit),
            storage=self._storage,
        )

    def resolve_path(self, path: str) -> str:
        """Expand $ aliases while keeping bare paths in the concrete namespace.

        $alias returns its resolved name; $alias.path expands its prefix.
        Both use authored rows or fixed resolved names, never concrete-prefix
        inference. Missing, unbound and wrong-kind aliases raise
        RoleResolutionError. Bare paths name concrete components or general
        fields, independent of alias names or bindings, in Run and Load captures.
        Other entry syntax/collision/unknown namespace errors propagate.
        """
        self._ensure_readable()
        if path.startswith("$"):
            alias, separator, suffix = path[1:].partition(".")
            if not alias or (separator and not suffix):
                raise ValueError(f"Expected $alias or $alias.path: {path!r}")
            spec = (
                self._roles.declarations.get(alias) if self._roles is not None else None
            )
            if self._aliases is not None:
                name = self._aliases.resolve(
                    alias, self._point, kind=spec.kind if spec is not None else None
                )
            else:
                if self._roles is None or alias not in self._roles.names:
                    raise RoleResolutionError(
                        alias,
                        None,
                        spec.kind if spec is not None else None,
                        "Alias is not bound",
                        message=f"${alias} is not bound in this capture",
                    )
                name = self._roles.names[alias]
            if not separator:
                return name
            return resolve_path(self._point, f"{name}.{suffix}", roles={})
        return resolve_path(self._point, path, roles={})

    def library_key(self, path: str) -> str:
        """Resolve a three-segment library root or a point module slot.

        Direct roots need not exist. Slots must contain a valid string root;
        absent/null or dangling slots raise ModuleSlotError naming the slot path.
        Unexpected source failures propagate unchanged. Bad syntax,
        nonstring references, and role/collision errors remain explicit.
        No materialization or file read occurs.
        """
        resolved = self.resolve_path(path)
        parts = resolved.split(".")
        is_slot = len(parts) == 3 and parts[0] != "general" and parts[1] == "module"
        if is_slot:
            reference = self._point_value(resolved)
            if not reference.present or reference.value is None:
                raise ModuleSlotError(f"Unset module slot: {path!r}")
            if not isinstance(reference.value, str):
                raise ValueError(f"Module slot must hold a library key: {path!r}")
            resolved = self.resolve_path(reference.value)
            parts = resolved.split(".")
        if (
            len(parts) != 3
            or parts[0] == "general"
            or parts[1] not in _LIBRARY_PARTITIONS
        ):
            raise ValueError(f"Expected a library root: {path!r}")
        kind = "waveform" if parts[1] == "waveforms" else "module"
        try:
            self._library.entry_value(resolved, kind=kind)
        except KeyError as cause:
            if is_slot:
                raise ModuleSlotError(f"Dangling module slot: {path!r}") from cause
            # Direct roots are valid keys even before an entry is created.
        return resolved

    def has_library_key(self, path: str) -> bool:
        """Test a captured library root or module slot without materializing it.

        path uses the same direct-key, slot and $alias grammar as library_key.
        Missing roots, unset/dangling slots and missing/unbound aliases return
        False. Invalid syntax, other alias errors and source failures propagate.
        The fixed point/library are never refreshed, changed or expanded.
        """
        try:
            key = self.library_key(path)
        except ModuleSlotError:
            return False
        except RoleResolutionError as exc:
            if exc.reason in (
                "Unknown alias",
                "Alias is not bound",
                "Role is not bound",
            ):
                return False
            raise
        kind = "waveform" if key.split(".")[1] == "waveforms" else "module"
        try:
            self._library.entry_value(key, kind=kind)
        except KeyError:
            return False
        return True

    def _library_path(self, resolved: str) -> str | None:
        parts = resolved.split(".")
        if len(parts) < 2 or parts[0] == "general":
            return None
        if parts[1] in _LIBRARY_PARTITIONS:
            return resolved
        if parts[1] == "module" and len(parts) > 3:
            root = self.library_key(".".join(parts[:3]))
            return ".".join([root, *parts[3:]])
        return None

    def lookup(self, path: str) -> DocumentValue:
        """Read a fixed name or point/library value, preserving explicit null.

        $alias returns a name string without catalog validation. Field paths use
        their point/library owner. Missing data returns present=False; syntax,
        namespace, source and materialization errors propagate. Unset/dangling
        module slots raise ModuleSlotError naming the slot path.
        """
        resolved = self.resolve_path(path)
        if path.startswith("$") and "." not in path:
            return DocumentValue(True, resolved)
        library_path = self._library_path(resolved)
        if library_path is not None:
            return self._library.lookup(library_path)
        return self._point_value(resolved)

    def _point_value(self, resolved: str) -> DocumentValue:
        try:
            return DocumentValue(True, document_value(self._point, resolved))
        except AttributeError:
            # Only the requested document path is read here, never references.
            return DocumentValue(False, None)

    def lookup_point(self, path: str) -> DocumentValue:
        """Read only point/general values, without materialization or asset I/O.

        Missing leaves return present=False; explicit null is present=True.
        Library partitions and module-slot namespaces, including their roots,
        raise InvalidInputError with reason_code parameter_path_not_readable.
        $alias returns its fixed name. Other syntax, namespace and source errors
        propagate unchanged.
        """
        resolved = self.resolve_path(path)
        if path.startswith("$") and "." not in path:
            return DocumentValue(True, resolved)
        parts = resolved.split(".")
        if (
            len(parts) > 1
            and parts[0] != "general"
            and parts[1] in (*_LIBRARY_PARTITIONS, "module")
        ):
            raise InvalidInputError(
                f"Parameter path is not point-readable: {path!r}",
                reason_code="parameter_path_not_readable",
            )
        return self._point_value(resolved)

    def value(self, path: str) -> YamlValue:
        """Return an isolated routed value; absent requested data raises AttributeError."""
        result = self.lookup(path)
        if not result.present:
            raise AttributeError(f"Missing parameter: {path!r}")
        return result.value
