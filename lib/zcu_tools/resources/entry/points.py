"""Working-point views over one complete parameter document."""

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from zcu_tools.format_version import YamlMap, YamlValue
from zcu_tools.resources.document_store import DocumentStore

from . import _point_origin
from .roles import RoleSpec, RoleView, resolve_names
from .schema import ComponentSchema, PointDocument, PointGeneral
from .views import (
    ComponentView,
    EditView,
    GeneralView,
    add_component_to_draft,
    document_meta,
    point_origin,
    register_view_read,
    stage_component,
    stage_general,
)


class PointView:
    """A bound point's cached working-unit values, independent of setup.

    Obtain a view from ResultEntry.use_point or new_point. Component and general
    reads do no I/O. Complex tags read as native complex scalars; manual
    scalar writes encode through the core codec before schema validation.
    Package snapshot/document reads retain raw YAML tags. Malformed tag reads
    propagate the core codec error. Unknown components or unset fields raise AttributeError.
    Assignments and edit() commit only this point's document. Value writes
    keep opaque metadata; edit().set_meta replaces it explicitly. Call refresh()
    to observe disk edits; another view does not change this view's binding.
    """

    def __init__(
        self,
        store: DocumentStore[PointDocument],
        source: Path,
        *,
        entry_id: str,
        result_root: Path,
        database_root: Path,
    ) -> None:
        """Bind one validated working-unit store without I/O.

        source is store's point.yaml path under the owning entry's points/label.
        The store owns single-file validation, conflict checks and persistence.
        ResultEntry supplies these dependencies; construction does not reload or
        revalidate field values. Binding rejects components shadowing any public
        SetupView/PointView/EditView member with ValueError naming the member.
        entry_id is the owning entry's immutable UUID. result_root and database_root
        are resolved once. Clone identity binds source, entry_id and result_root,
        including for directly constructed views. database_root only configures
        Save output, not clone identity. Cloning requires a matching Result root
        and reloads the published source, not this cached snapshot. No later
        identity registration is required. There are no multi-file callbacks.
        ResultEntry's store callback rejects shadowing on refresh/mutation before
        publication. With a caller-provided store, the caller supplies validate
        for that subsequent lifecycle. Without it, refresh/mutation can publish
        shadowing names; those components cannot be read as view attributes and
        must be read through snapshot/document_value instead.
        """
        self._store = store
        self._source = source
        origin = _point_origin.PointOrigin(
            source=source,
            entry_id=entry_id,
            result_root=result_root.resolve(),
            database_root=database_root.resolve(),
        )
        register_view_read(self, store, source, origin=origin)

    @property
    def general(self) -> GeneralView:
        """Return cached general fields; tagged ext leaves read as native complex."""
        return GeneralView(lambda: self._store.snapshot().general, self._edit_general)

    @contextmanager
    def _edit_general(self, _field: str) -> Generator[PointGeneral]:
        with self._store.edit() as draft, stage_general(draft) as candidate:
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
    def _edit_component(self, name: str, _field: str) -> Generator[ComponentSchema]:
        with (
            self._store.edit() as draft,
            stage_component(draft, name) as candidate,
        ):
            yield candidate

    @contextmanager
    def edit(self) -> Generator[EditView]:
        """Yield a working-unit draft and commit one document on normal exit.

        Reload only point.yaml on entry. Values and opaque metadata in the
        draft commit together. Required fields, original field validators and
        model hooks must validate. Body, schema, metadata representation, custom
        validation, conflict or I/O failures keep the previous file and snapshot.
        Independent handles merge disjoint leaves or reject the entire edit on
        a same-leaf conflict. No other point or setup file is read or written.
        """
        with self._store.edit() as draft:
            yield EditView(draft)

    def resolve(
        self,
        roles: Mapping[str, RoleSpec] | Sequence[str] | None = None,
        *,
        focus: str | None = None,
        **explicit: str | None,
    ) -> RoleView:
        """Resolve only declared roles from this point's cached components.

        Explicit choices precede one matching focus, then RoleSpec.via or a
        role-name string field on already resolved components. An explicit None
        means deliberately unbound, not inference; reading that declared role
        raises RoleResolutionError.
        None uses the caller-registered notebook shorthand. Automatic inference
        with zero/multiple choices omits the role; its use then raises
        RoleResolutionError. Explicit invalid choices and dangling/wrong-kind
        targets fail here. Errors identify role, focus, kind and reason. The returned name mapping stays fixed while
        values follow this bound point's working-unit snapshot and edits. Its
        constructor origin also binds the returned roles to this entry/point.
        """
        names = resolve_names(self._store.snapshot().components, roles, focus, explicit)
        return RoleView(
            names, self.__getattr__, origin=point_origin(self), roles=roles, focus=focus
        )

    def meta(self, path: str) -> YamlMap | None:
        """Return isolated cached opaque metadata for a dotted path, or None.

        No value lookup, key interpretation or I/O occurs; {} means cleared.
        """
        return document_meta(self._store.snapshot(), path)

    def refresh(self) -> None:
        """Reload only point.yaml; invalid data keeps the cached snapshot.

        Missing files, incompatible headers or invalid component models raise
        without publication. No files are committed.
        """
        self._store.refresh()

    def add_component(self, name: str, *, kind: str, **fields: YamlValue) -> None:
        """Add a complete registered component to this point, not its template.

        name is a public identifier, not a keyword or a public member of
        SetupView, PointView or EditView. Shadowing raises ValueError naming the
        member. kind is a registered kind.
        fields are working-unit YAML values, including all required model fields.
        Original field validators and defaults run; references resolve only in
        this point, when resolve is called. Invalid names, schema values or
        duplicate components raise without publishing a value. No metadata is added.
        Failure keeps the file and cached snapshot; success commits one edit.
        """
        with self._store.edit() as draft:
            add_component_to_draft(draft, name, kind, fields, self._source)
