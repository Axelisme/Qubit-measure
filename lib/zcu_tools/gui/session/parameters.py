"""Fixed point/library read projections and consumer role bindings."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zcu_tools.format_version import YamlValue
    from zcu_tools.gui.session.context_types import ContextSourceFault, ContextTarget
    from zcu_tools.resources.context.library import LibrarySnapshot
    from zcu_tools.resources.document_store import DocumentValue
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
    ) -> None:
        """Detach input data and retain the immutable library projection.

        RoleResolutionError reports missing/wrong-kind captured destinations.
        A setup diagnostic is outside this capture contract and raises
        ValueError. No session selection generation is accepted or compared.
        """
        self._point = point.model_copy(deep=True)
        self._library = library
        self._target = target.model_copy(deep=True)
        self._roles = deepcopy(roles)
        self._source_faults = deepcopy(source_faults)

    @property
    def target(self) -> ContextTarget:
        """Return an independent persisted-point or entry-draft destination."""
        return self._target.model_copy(deep=True)

    @property
    def roles(self) -> RoleBinding | None:
        """Return a detached fixed binding; None means direct paths only."""
        return deepcopy(self._roles)

    @property
    def source_faults(self) -> tuple[ContextSourceFault, ...]:
        """Return detached point/module_cfg diagnostics, empty when healthy."""
        return deepcopy(self._source_faults)

    @property
    def library(self) -> LibrarySnapshot:
        """Return the fixed read-only library; unavailable sources raise."""
        raise NotImplementedError("ParameterContext library read is not implemented")

    def document(self) -> PointDocument:
        """Copy the fixed point without I/O or model hooks; unavailable sources raise."""
        raise NotImplementedError("ParameterContext document read is not implemented")

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
        raise NotImplementedError("ParameterContext role binding is not implemented")

    def resolve_path(self, path: str) -> str:
        """Expand only the captured role prefix, preserving the remaining path.

        Accepts direct components, captured roles, and general.* paths.
        Entry syntax/collision/unknown-namespace errors propagate.
        """
        raise NotImplementedError("ParameterContext path expansion is not implemented")

    def library_key(self, path: str) -> str:
        """Resolve a three-segment library root or a point module slot.

        Direct roots need not exist. Slots must contain a valid string root;
        absent/null or dangling slots raise AttributeError. Bad syntax,
        nonstring references, and role/collision errors remain explicit.
        No materialization or file read occurs.
        """
        raise NotImplementedError("ParameterContext library routing is not implemented")

    def lookup(self, path: str) -> DocumentValue:
        """Read a point/library value, distinguishing absent from explicit null.

        Uses the fixed roles and routes library suffixes through their owner.
        Missing requested data returns present=False. Invalid syntax,
        unknown namespaces, dangling slots, and materialization/asset errors
        propagate rather than becoming missing.
        """
        raise NotImplementedError("ParameterContext lookup is not implemented")

    def lookup_point(self, path: str) -> DocumentValue:
        """Read only point/general values, without materialization or asset I/O.

        Missing leaves return present=False; explicit null is present=True.
        Library partitions and module-slot namespaces, including their roots,
        raise InvalidInputError with reason_code parameter_path_not_readable.
        Other syntax, namespace, and source errors propagate unchanged.
        """
        raise NotImplementedError("ParameterContext point lookup is not implemented")

    def value(self, path: str) -> YamlValue:
        """Return an isolated routed value; absent requested data raises AttributeError."""
        raise NotImplementedError("ParameterContext value read is not implemented")

    def meta(self, path: str) -> Provenance | None:
        """Read the routed provenance, or None for absent/unsourced data.

        Uses concrete library leaf sources, never synthesizing slot sources.
        Syntax, namespace, and source errors propagate.
        """
        raise NotImplementedError("ParameterContext provenance read is not implemented")
