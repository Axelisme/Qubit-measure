"""Working-point views over one complete parameter document."""

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from zcu_tools.format_version import YamlValue
from zcu_tools.resources.document_store import DocumentStore

from . import _point_origin
from .provenance import Provenance
from .roles import RoleSpec, RoleView, resolve_names
from .schema import ComponentSchema, PointDocument, PointGeneral
from .views import (
    ComponentView,
    EditView,
    GeneralView,
    add_component_to_draft,
    document_meta,
    stage_component,
    stage_general,
)


class PointView:
    """A bound point's cached working-unit values, independent of setup.

    Obtain a view from ResultEntry.use_point or new_point. Component and general
    reads do no I/O. Unknown components or unset fields raise AttributeError.
    Assignments and edit() commit only this point's document. Successful field
    writes clear cloned_from even when the value is unchanged. Call refresh()
    to observe disk edits; another view does not change this view's binding.
    """

    def __init__(
        self,
        store: DocumentStore[PointDocument],
        source: Path,
        *,
        ledger: Path,
        entry_id: str,
    ) -> None:
        """Bind one validated working-unit store without I/O.

        source is store's point.yaml path under the owning entry's points/label.
        The store owns single-file validation, conflict checks and persistence.
        ResultEntry supplies these dependencies; construction does not reload or
        validate them. There are no shared setup or multi-file callbacks.
        """
        self._store = store
        self._source = source
        self._ledger = ledger
        self._entry_id = entry_id
        _point_origin.register(self, source)

    @property
    def general(self) -> GeneralView:
        """Return this point's cached creation time, description and YAML ext."""
        return GeneralView(lambda: self._store.snapshot().general, self._edit_general)

    @contextmanager
    def _edit_general(self, field: str) -> Generator[PointGeneral]:
        with self._store.edit() as draft, stage_general(draft, field) as candidate:
            yield candidate

    @property
    def description(self) -> str | None:
        """Return the cached description, or None when it is unset."""
        return self._store.snapshot().general.description

    @description.setter
    def description(self, value: str | None) -> None:
        with self.edit() as draft:
            draft.description = value

    def __getattr__(self, name: str) -> ComponentView:
        if name not in self._store.snapshot().components:
            raise AttributeError(f"Unknown component {name!r}")
        return ComponentView(
            lambda: self._store.snapshot().components[name],
            lambda field: self._edit_component(name, field),
            name,
        )

    @contextmanager
    def _edit_component(self, name: str, field: str) -> Generator[ComponentSchema]:
        with (
            self._store.edit() as draft,
            stage_component(draft, name, field) as candidate,
        ):
            yield candidate

    @contextmanager
    def edit(self) -> Generator[EditView]:
        """Yield a working-unit draft and commit one document on normal exit.

        Reload only point.yaml on entry. Required fields, original field
        validators and model hooks must validate. Body, schema, custom validation,
        conflict or I/O failures keep the previous file and snapshot.
        Independent handles merge disjoint leaves or reject the entire edit on
        a same-leaf conflict. No other point or setup file is read or written.
        """
        with self._store.edit() as draft:
            yield EditView(draft, ledger=self._ledger, entry_id=self._entry_id)

    def resolve(
        self,
        roles: Mapping[str, RoleSpec] | Sequence[str] | None = None,
        *,
        focus: str | None = None,
        **explicit: str,
    ) -> RoleView:
        """Resolve only declared roles from this point's cached components.

        Explicit choices precede one matching focus, then RoleSpec.via or a
        role-name string field on already resolved components.
        None uses the caller-registered notebook shorthand. Errors identify the
        role, focus, kind and reason. The returned name mapping stays fixed while
        values follow this bound point's working-unit snapshot and edits.
        """
        names = resolve_names(self._store.snapshot().components, roles, focus, explicit)
        return RoleView(names, self.__getattr__)

    def meta(self, path: str) -> Provenance | None:
        """Return an independent cached source in working units, or None."""
        return document_meta(self._store.snapshot(), path)

    def refresh(self) -> None:
        """Reload only point.yaml; invalid data keeps the cached snapshot.

        Missing files, incompatible headers or invalid component models raise
        without publication. No files are committed.
        """
        self._store.refresh()

    def add_component(self, name: str, *, kind: str, **fields: YamlValue) -> None:
        """Add a complete registered component to this point, not its template.

        name is an unreserved public identifier and kind is a registered kind.
        fields are working-unit YAML values, including all required model fields.
        Original field validators and defaults run; references resolve only in
        this point, when resolve is called. Invalid names, schema values or
        duplicate components raise without publishing a value or source.
        Failure keeps the file and cached snapshot; success commits one edit.
        """
        with self._store.edit() as draft:
            add_component_to_draft(draft, name, kind, fields, self._source)
