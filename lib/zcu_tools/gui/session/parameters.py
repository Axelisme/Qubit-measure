"""Fixed point/library read projections and consumer role bindings."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from types import MappingProxyType
from typing import TYPE_CHECKING

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.context_types import (
    ContextSourceUnavailableError,
    SavedPoint,
)
from zcu_tools.resources.document_store import DocumentValue
from zcu_tools.resources.entry import (
    RoleResolutionError,
    document_meta,
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
    from zcu_tools.resources.entry.provenance import Provenance
    from zcu_tools.resources.entry.schema import PointDocument


@dataclass(frozen=True)
class RoleBinding:
    """Captured focus (or None), ordered declarations and role-to-component names.

    Construction detaches both mappings. ParameterContext also detaches returned
    bindings, so editing a consumer's mapping never changes another capture.
    """

    focus: str | None
    declarations: Mapping[str, RoleSpec]
    names: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "declarations", deepcopy(dict(self.declarations)))
        object.__setattr__(self, "names", deepcopy(dict(self.names)))


class ParameterContext:
    """An isolated working-unit point/library capture with an explicit target.

    Construction accepts validated point and library snapshots, a SavedPoint
    or EntryDraft destination, and an optional fixed consumer RoleBinding.
    source_faults contains only point/module_cfg diagnostics, not setup faults.
    The target, roles, and diagnostics remain readable on an unavailable
    capture; all data reads then raise ContextSourceUnavailableError.
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

        RoleResolutionError reports missing/wrong-kind captured destinations.
        A setup diagnostic is outside this capture contract and raises
        ValueError. storage optionally retains a resource-owned saved-point
        capability with matching target identity; omit it for drafts or synthetic
        captures. Mismatched storage origins also raise ValueError.
        No session selection generation is accepted or compared.
        """
        if any(fault.document == "setup" for fault in source_faults):
            raise ValueError("Parameter captures accept only point/module_cfg faults")
        if storage is not None and (
            not isinstance(target, SavedPoint)
            or str(storage.entry_id) != target.entry_id
            or storage.point != target.point
        ):
            raise ValueError("Storage origin must match the saved-point target")
        self._point = point.model_copy(deep=True)
        self._library = library
        self._target = target.model_copy(deep=True)
        self._roles = deepcopy(roles)
        self._source_faults = deepcopy(source_faults)
        self._storage = storage
        if self._roles is not None and not self._source_faults:
            for role, spec in self._roles.declarations.items():
                name = self._roles.names.get(role)
                component = (
                    self._point.components.get(name) if name is not None else None
                )
                if component is None or not fnmatchcase(component.kind, spec.kind):
                    raise RoleResolutionError(
                        role,
                        self._roles.focus,
                        spec.kind,
                        f"Captured target {name!r} is missing or has the wrong kind",
                    )

    @property
    def target(self) -> ContextTarget:
        """Return an independent persisted-point or entry-draft destination."""
        return self._target.model_copy(deep=True)

    @property
    def roles(self) -> RoleBinding | None:
        """Return a detached fixed binding; None means direct paths only."""
        return deepcopy(self._roles)

    @property
    def storage(self) -> PointStorageBinding | None:
        """Return the fixed origin capability, or None for draft/unbound captures.

        Its roots, ledger and description belong to capture time. This projection
        adds this consumer's fixed role names without consulting session focus.
        No file I/O, refresh or ledger append occurs.
        """
        if self._storage is None:
            return None
        names = {} if self._roles is None else dict(self._roles.names)
        return replace(self._storage, roles=MappingProxyType(names))

    def capture_storage(self) -> PointStorageBinding:
        """Return storage at this capture's origin, never the current selection.

        Necessary source faults raise ContextSourceUnavailableError. Drafts and
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

    def bind_roles(
        self,
        declarations: Mapping[str, RoleSpec],
        *,
        focus: str | None,
        explicit: Mapping[str, str],
    ) -> ParameterContext:
        """Resolve roles on this fixed point and return an independent capture.

        Declarations are ordered kind/via rules. Explicit concrete names win
        over focus/via inference. Empty declarations remain empty. Entry
        RoleResolutionError propagates for invalid names or failed inference.
        target and source data are retained; live session state is never read.
        """
        self._ensure_readable()
        names = resolve_names(self._point.components, declarations, focus, explicit)
        return ParameterContext(
            point=self._point,
            library=self._library,
            target=self._target,
            roles=RoleBinding(focus, declarations, names),
            storage=self._storage,
        )

    def resolve_path(self, path: str) -> str:
        """Expand only the captured role prefix, preserving the remaining path.

        Accepts direct components, captured roles, and general.* paths.
        Entry syntax/collision/unknown-namespace errors propagate.
        """
        self._ensure_readable()
        names = self._roles.names if self._roles is not None else {}
        return resolve_path(self._point, path, roles=names)

    def library_key(self, path: str) -> str:
        """Resolve a three-segment library root or a point module slot.

        Direct roots need not exist. Slots must contain a valid string root;
        absent/null or dangling slots raise AttributeError. Bad syntax,
        nonstring references, and role/collision errors remain explicit.
        No materialization or file read occurs.
        """
        resolved = self.resolve_path(path)
        parts = resolved.split(".")
        is_slot = len(parts) == 3 and parts[0] != "general" and parts[1] == "module"
        if is_slot:
            reference = self._point_value(resolved)
            if not reference.present or reference.value is None:
                raise AttributeError(f"Unset module slot: {path!r}")
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
                raise AttributeError(f"Dangling module slot: {path!r}") from cause
            # Direct roots are valid keys even before an entry is created.
        return resolved

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
        """Read a point/library value, distinguishing absent from explicit null.

        Uses the fixed roles and routes library suffixes through their owner.
        Missing requested data returns present=False. Invalid syntax,
        unknown namespaces, dangling slots, and materialization/asset errors
        propagate rather than becoming missing.
        """
        resolved = self.resolve_path(path)
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
        Other syntax, namespace, and source errors propagate unchanged.
        """
        resolved = self.resolve_path(path)
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

    def meta(self, path: str) -> Provenance | None:
        """Read the routed provenance, or None for absent/unsourced data.

        Uses concrete library leaf sources, never synthesizing slot sources.
        Syntax, namespace, and source errors propagate.
        """
        resolved = self.resolve_path(path)
        library_path = self._library_path(resolved)
        if library_path is not None:
            return self._library.meta(library_path)
        return document_meta(self._point, resolved)
