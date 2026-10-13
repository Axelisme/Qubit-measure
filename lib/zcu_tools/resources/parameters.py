"""Standalone caller-owned values and opaque metadata in one parameter document.

Open an existing file with open_parameters(path, MyValues). No kind registry,
workspace, ledger or experiment definitions are needed. Values stay directly
readable under components in YAML; metadata is a separate path-keyed table.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar, Literal, Self, override

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from zcu_tools.format_version import YamlMap

from .document_store import DocumentChange, DocumentStore
from .parameter_fields import FieldView, set_model_value, stage_model
from .parameter_metadata import (
    PARAMETER_FORMAT,
    PARAMETER_VERSION,
    read_metadata,
    validate_parameter_header,
    write_metadata,
)

__all__ = ("ParameterEdit", "ParameterView", "open_parameters")


class _ParameterDocument[T: BaseModel](BaseModel):
    model_config = ConfigDict(extra="forbid")
    _source: ClassVar[Path]

    format: str
    format_version: str
    components: T
    metadata: dict[str, YamlMap] = Field(default_factory=dict)

    @classmethod
    @override
    def model_validate(
        cls,
        obj: object,
        *,
        strict: bool | None = None,
        extra: Literal["allow", "ignore", "forbid"] | None = None,
        from_attributes: bool | None = None,
        context: object = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        # Reject old tables before forward-minor preservation can hide them.
        document = TypeAdapter(YamlMap).validate_python(obj, strict=True)
        validate_parameter_header(document, source=cls._source)
        return super().model_validate(
            document,
            strict=strict,
            extra=extra,
            from_attributes=from_attributes,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @field_validator("components", mode="before")
    @classmethod
    def validate_values(cls, value: object) -> object:
        # A separate call prevents the envelope's forward-minor override from
        # changing the caller model's declared extra policy.
        model = cls.model_fields["components"].annotation
        return TypeAdapter(model).validate_python(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, value: object) -> dict[str, YamlMap]:
        return TypeAdapter(dict[str, YamlMap]).validate_python(value, strict=True)


class ParameterEdit[T: BaseModel](FieldView):
    """One detached value/metadata draft yielded by ParameterView.edit().

    Field attributes/items follow this draft, not the published snapshot. Each
    assignment validates a complete candidate before replacing draft values.
    set_meta replaces one entire mapping without changing values. Body errors
    abort the edit; the owning DocumentStore validates and commits on normal exit.
    Do not construct this type directly or reuse it after its edit scope exits.
    Public method names use item access when also present as caller value fields.
    """

    _document: _ParameterDocument[T]

    def __init__(self, document: _ParameterDocument[T]) -> None:
        """Bind an internal detached document draft; use ParameterView.edit()."""
        self._document = document
        super().__init__(lambda: self._document.components, self._edit_values, "")

    def _publish_values(self, values: T) -> None:
        self._document.components = values

    @contextmanager
    def _edit_values(self, _path: str) -> Generator[T]:
        with stage_model(self._document.components, self._publish_values) as values:
            yield values

    def snapshot(self) -> T:
        """Return an isolated caller value model from the current draft, without I/O."""
        return self._document.components.model_copy(deep=True)

    def meta(self, path: str) -> YamlMap | None:
        """Read isolated opaque draft metadata; absent paths return None.

        path uses nonempty dotted logical field segments. No value lookup or
        domain interpretation occurs. Empty stored mappings return {}.
        """
        return read_metadata(self._document.metadata, path)

    def set_meta(self, path: str, metadata: object) -> None:
        """Replace a path's complete draft metadata with a detached YAML mapping.

        path uses nonempty dotted segments. metadata accepts only string keys
        and YAML null/bool/int/float/str/list/mapping values, checked strictly.
        Invalid representation raises ValidationError before changing the draft.
        {} clears the path. Values and other metadata paths remain unchanged.
        DocumentStore's usual mapping-leaf merge/conflict rules apply at commit.
        """
        write_metadata(self._document.metadata, path, metadata)
        self._document.model_fields_set.add("metadata")

    def set(self, path: str, value: object) -> None:
        """Assign a dotted value path with full caller-model validation.

        No list indexing or escaping is supported. Missing/non-container parents
        raise KeyError/AttributeError; schema errors raise ValidationError. Any
        failure leaves this draft's values and metadata unchanged. Metadata is
        retained unchanged, including for same-value writes.
        """
        with self._edit_values(path) as model:
            set_model_value(model, path, value)


class ParameterView[T: BaseModel](FieldView):
    """A live memory view of caller values with single-file edits and opaque metadata.

    Obtain it from open_parameters(path, MyValues). Field reads never reload
    disk. Scalar values are ordinary numbers/strings/etc.; nested model/mapping
    fields use live FieldViews. Attributes and item assignments perform one
    edit. Use edit() to commit several values and metadata together. Method
    names that also name caller fields remain accessible through item access.
    snapshot() returns only the isolated caller value model, not the envelope.
    No parameter metadata is generated, interpreted or cleared by value writes.
    """

    _store: DocumentStore[_ParameterDocument[T]]

    def __init__(self, store: DocumentStore[_ParameterDocument[T]]) -> None:
        """Bind an internal store; callers use open_parameters rather than this constructor."""
        self._store = store
        super().__init__(
            lambda: self._store.snapshot().components, self._edit_values, ""
        )

    @contextmanager
    def _edit_values(self, _path: str) -> Generator[T]:
        with self._store.edit() as document:

            def publish(values: T) -> None:
                document.components = values

            with stage_model(document.components, publish) as values:
                yield values

    @contextmanager
    def edit(self) -> Generator[ParameterEdit[T]]:
        """Reload under short locks and yield a detached, single-file transaction.

        The body runs unlocked. Normal exit merges changed mapping leaves using
        DocumentStore; concurrent changes to the same leaf raise ConflictError
        and reject the whole edit. Nested edits on this handle raise RuntimeError.
        Schema, representation, body, lock and pre-replace I/O errors leave the
        published snapshot and file unchanged. Successful empty edits adopt disk
        state without writing. Observers run after publication and unlock; their
        failures are logged, not propagated as rollback. No retry, cross-file
        atomicity or crash-recovery guarantee is provided.
        """
        with self._store.edit() as document:
            yield ParameterEdit(document)

    def snapshot(self) -> T:
        """Return a detached typed caller value model from memory, without reading disk."""
        return self._store.snapshot().components

    def meta(self, path: str) -> YamlMap | None:
        """Return isolated opaque metadata from memory, or None for an absent path.

        path uses nonempty dotted logical segments. {} means explicitly cleared
        metadata. No value lookup, schema hooks, source lookup or I/O occurs.
        """
        return read_metadata(self._store.snapshot().metadata, path)

    def refresh(self) -> bool:
        """Explicitly reload and validate disk; return whether its raw content changed.

        Failed header/schema/representation/lock/I/O validation preserves the
        prior snapshot. Successful publication uses DocumentStore notifications;
        observer errors are logged and cannot undo refresh.
        """
        return self._store.refresh()

    def subscribe(
        self, callback: Callable[[DocumentChange], None]
    ) -> Callable[[], None]:
        """Observe publication after unlock; return an idempotent unsubscribe function.

        callback receives DocumentChange with the file path, envelope mapping
        paths (components/metadata) and commit/refresh reason. Each registration
        is independent. Exceptions are logged with traceback after publication,
        without rolling back committed values or metadata.
        """
        return self._store.subscribe(callback)


def open_parameters[T: BaseModel](path: Path, model: type[T]) -> ParameterView[T]:
    """Open an existing parameter-container 2.x file with a caller-owned value model.

    path is the exact UTF-8 YAML file, not a directory. model is a Pydantic
    BaseModel class describing components values, with no required kind or
    registry. The envelope has format: zcu.parameter-container, format_version:
    '2.0' (or a forward 2.x minor), components and a path-keyed metadata mapping.
    Metadata uses strict string-keyed YAML values and has no domain semantics.
    No workspace, general section, roots, ledger or library is required.

    Returned attributes read ordinary scalar values; nested models/mappings are
    live field views. snapshot is an isolated, memory-only caller model. edit
    reloads and delegates atomic replacement, optimistic leaf conflicts and
    publication to DocumentStore; refresh explicitly reloads. Value writes keep
    metadata unchanged. Values and metadata may be submitted in one edit.

    Missing/unreadable files, YAML/header/version/schema/representation errors
    propagate; no file is created or migrated. Old majors and any provenance
    table are rejected with the path, actual version and expected version 2.0.
    Forward-minor envelope fields survive commits without overriding caller
    value-model extra policy. See edit/refresh for failure and observer semantics.
    """
    TypeAdapter(type[BaseModel]).validate_python(model)

    # Bind diagnostics per handle, not on a shared model class or global registry.
    class BoundDocument[ValueModel: BaseModel](_ParameterDocument[ValueModel]):
        _source: ClassVar[Path] = path

    store: DocumentStore[_ParameterDocument[T]] = DocumentStore(
        path,
        BoundDocument[model],
        format=PARAMETER_FORMAT,
        supported_version=PARAMETER_VERSION,
    )
    return ParameterView(store)
