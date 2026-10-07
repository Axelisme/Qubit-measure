"""Setup and transaction views over one typed document store.

Copy incoming values before assignment: notebook models can validate assignment.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, overload

from pydantic import BaseModel, TypeAdapter

from zcu_tools.format_version import YamlMap, YamlValue
from zcu_tools.resources.document_store import DocumentStore

from ._point_origin import PointOrigin
from .errors import RoleResolutionError
from .provenance import Provenance, validate_source
from .registry import component_registry
from .schema import (
    ComponentSchema,
    PointDocument,
    PointGeneral,
    SetupDocument,
    SetupGeneral,
    validate_component_name,
)

if TYPE_CHECKING:
    from .points import PointView


type _FieldNode = BaseModel | dict[str, object]
type _ViewDocument = (
    DocumentStore[SetupDocument]
    | DocumentStore[PointDocument]
    | SetupDocument
    | PointDocument
)


@dataclass(frozen=True)
class _ViewRead:
    """Package-owned access for three view types without reserving method names.

    document is the store/draft owned by this view.
    source is the bound YAML path, or None for an edit draft with no source getter.
    origin is a point's constructor identity, or None for setup/edit views.
    source_metadata validates/copies sources using an edit's local ledger;
    None for read-only views. It never changes the draft or produces an event.
    """

    document: _ViewDocument
    source: Path | None
    origin: PointOrigin | None
    source_metadata: Callable[[Provenance], YamlMap] | None


def register_view_read(
    view: object,
    document: _ViewDocument,
    source: Path | None,
    *,
    origin: PointOrigin | None = None,
    source_metadata: Callable[[Provenance], YamlMap] | None = None,
) -> None:
    """Bind one entry-internal read record during view construction, without I/O.

    Not a package export. document is the view's validated store/draft, source
    its YAML path (None for EditView), and origin its full point identity (None
    for setup/edit). source_metadata is an edit's local source-validation
    capability, absent for read-only views. The private attribute cannot collide
    with legal components.
    Already bound handles raise ValueError; objects without an instance dictionary
    raise TypeError. No model hook or callback runs.
    """
    state = vars(view)
    if "_entry_read" in state:
        raise ValueError("Entry view is already bound")
    state["_entry_read"] = _ViewRead(document, source, origin, source_metadata)


def _view_read(view: object) -> _ViewRead:
    # One owner-local reflective boundary keeps private state off the view namespace.
    binding: object = vars(view)["_entry_read"]
    if not isinstance(binding, _ViewRead):
        raise TypeError("Invalid entry view read binding")
    return binding


def point_origin(view: PointView) -> PointOrigin:
    """Return the point's immutable constructor identity without I/O.

    Entry-internal helper, not a package export. A non-point binding raises
    ValueError; no destination identity is used to fill missing information.
    """
    origin = _view_read(view).origin
    if origin is None:
        raise ValueError("View has no point origin")
    return origin


@overload
def snapshot(view: PointView) -> PointDocument: ...


@overload
def snapshot(view: SetupView) -> SetupDocument: ...


@overload
def snapshot(view: EditView) -> SetupDocument | PointDocument: ...


def snapshot(
    view: PointView | SetupView | EditView,
) -> SetupDocument | PointDocument:
    """Copy this handle's cached document or edit draft without I/O or validation.

    The result is deeply isolated, uses working units and preserves the typed
    document's field visibility. Mutating it does not change the handle or disk.
    """
    document = _view_read(view).document
    if isinstance(document, DocumentStore):
        return document.snapshot()
    return document.model_copy(deep=True)


def manual_update(
    document: PointDocument,
    path: str,
    value: YamlValue,
    *,
    stderr: float | None = None,
) -> PointDocument:
    """Copy and validate a manual working-unit write to an actual point field.

    path is a dotted component/general field, without role expansion. Invalid
    paths, missing components, schema values or non-finite/negative stderr
    raise before returning. The input and caller's value remain isolated.
    Record manual provenance on the accepted path and its leaves, clearing
    clone origin. Preserve general.created_at unless explicitly assigned.
    No file, ledger, notification or default timestamp creation is involved.
    """
    if stderr is not None and (not isfinite(stderr) or stderr < 0):
        raise ValueError("stderr must be finite and non-negative, or None")
    candidate = document.model_copy(deep=True)
    metadata = _manual_source()
    metadata["stderr"] = stderr
    _set_document_value(candidate, path, value, metadata=metadata)
    return PointDocument.model_validate(_logical_value(candidate))


def prepare_point_update(
    document: PointDocument,
    path: str,
    value: YamlValue,
    *,
    source: Provenance | None,
    ledger: Path,
    entry_id: str,
) -> PointDocument:
    """Prepare an isolated working-unit point candidate for the owning entry.

    document is the validated memory point; path is a concrete component/general
    dotted field and value is YAML data. ledger is this entry's ledger.jsonl path
    and entry_id its UUID. source=None records manual provenance without reading
    ledger; explicit Provenance uses existing local source validation and clears
    accepted clone origin. Preserve other fields/created_at and all inputs.
    Invalid path/schema/source raises before return; explicit source ledger,
    attachment and I/O failures propagate. No point/YAML/store creation, commit,
    event append or notification.
    """
    if source is None:
        return manual_update(document, path, value)
    metadata = _accepted_source(source, ledger, entry_id)
    candidate = document.model_copy(deep=True)
    _set_document_value(candidate, path, value, metadata=metadata)
    return PointDocument.model_validate(_logical_value(candidate))


def accept_source(view: EditView, path: str, *, provenance: Provenance) -> None:
    """Attach a local source to an existing draft value without changing it.

    path is an actual dotted component/general field. Containers accept all
    current leaves and clear clone origin on the path and descendants.
    Missing values/invalid paths or invalid local ledger sources raise without
    changing the source table. Ledger schema and I/O errors propagate.
    Never validate the value, run model hooks or commit; view.edit() owns commit.
    """
    parts = _path_parts(path)
    if len(parts) < 2:
        raise ValueError(f"{path!r}: expected a dotted field path")
    binding = _view_read(view)
    draft = binding.document
    if isinstance(draft, DocumentStore) or binding.source_metadata is None:
        raise TypeError("Source acceptance requires an edit draft")
    value = document_value(draft, path)
    metadata = binding.source_metadata(provenance)
    draft.provenance = _sources_after_write(draft.provenance, path, value, metadata)


def source_path(view: PointView | SetupView) -> Path:
    """Return the YAML path bound to this handle, without I/O.

    EditView has no source getter; its caller captures the owning handle's path.
    """
    source = _view_read(view).source
    if source is None:
        raise TypeError("Edit drafts have no bound source path")
    return source


@dataclass(frozen=True)
class DocumentField:
    """One detached logical leaf, list or empty mapping in document order.

    segments retains literal keys without a components wrapper. editable_path
    is their dotted path, or None if any segment is empty or contains a dot.
    value uses working units; provenance is the full recorded source or None,
    including cloned_from when present. Rows never expose opaque future data.
    """

    segments: tuple[str, ...]
    editable_path: str | None
    value: YamlValue
    provenance: Provenance | None


def document_fields(
    document: SetupDocument | PointDocument,
) -> tuple[DocumentField, ...]:
    """List isolated logical rows without I/O, validation or serialization hooks.

    Expand models/mappings in data order; lists, null and empty mappings remain
    single rows. Preserve unaddressable literal keys with editable_path=None
    and no dotted provenance lookup. Invalid logical YAML values raise TypeError
    or Pydantic ValidationError rather than hiding the document.
    """
    rows: list[DocumentField] = []

    def append_rows(segments: tuple[str, ...], value: YamlValue) -> None:
        if isinstance(value, dict) and value:
            for key, child in value.items():
                append_rows((*segments, key), child)
            return
        path = (
            ".".join(segments)
            if all(part and "." not in part for part in segments)
            else None
        )
        source = document.provenance.get(path) if path is not None else None
        provenance = (
            TypeAdapter(Provenance).validate_python(deepcopy(source))
            if source is not None
            else None
        )
        rows.append(DocumentField(segments, path, value, provenance))

    append_rows(("general",), _logical_value(document.general))
    for name, component in document.components.items():
        append_rows((name,), _logical_value(component))
    return tuple(rows)


def document_value(document: SetupDocument | PointDocument, path: str) -> YamlValue:
    """Read an actual dotted component/general path from this document only.

    Return an isolated working-unit YAML value, including explicit null and
    logical containers. Never run validators, serializers or read another file.
    Invalid or unaddressable syntax raises ValueError. Missing components,
    fields, unset optional values or non-container ancestors raise AttributeError.
    Model fields and extras take precedence over class attributes; hidden
    forward-minor data stays hidden. Roles are not expanded here.
    """
    parts = _path_parts(path)
    node: object = document if parts[0] == "general" else document.components
    for index, part in enumerate(parts):
        prefix = ".".join(parts[: index + 1])
        if isinstance(node, BaseModel):
            node = _read_field(node, part, ".".join(parts[:index]))
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise AttributeError(f"{prefix}: path {path!r} is not readable")
    return _logical_value(node)


def resolve_path(
    document: PointDocument,
    path: str,
    *,
    roles: Mapping[str, str],
    reserved_names: Collection[str] = (),
) -> str:
    """Resolve a dotted path's first segment using captured role-to-component names.

    Read only this document. Preserve the remaining segments without checking
    leaf or library-key existence. Component names and the general namespace
    stay unchanged. Empty paths or segments raise ValueError. A prefix shared
    by roles, components, logical general fields or reserved_names raises
    RoleResolutionError identifying the namespaces. Unknown prefixes and roles
    bound to missing components also raise RoleResolutionError. General fields
    require the general prefix; bare metadata keys are not supported.
    No I/O, model validation, serialization or mutation occurs.
    """
    parts = _path_parts(path)
    prefix = parts[0]
    namespaces: list[str] = []
    if prefix in roles:
        namespaces.append("role")
    if prefix in document.components:
        namespaces.append("component")
    if (
        prefix == "general"
        or prefix in type(document.general).model_fields
        or prefix in (document.general.model_extra or {})
    ):
        namespaces.append("general")
    if prefix in reserved_names:
        namespaces.append("reserved")
    if len(namespaces) > 1:
        raise RoleResolutionError(
            prefix,
            None,
            None,
            f"Ambiguous path prefix {prefix!r}: namespaces {', '.join(namespaces)}",
        )
    if prefix in roles:
        target = roles[prefix]
        if target not in document.components:
            raise RoleResolutionError(
                prefix, None, None, f"Role targets missing component {target!r}"
            )
        parts[0] = target
    elif prefix not in document.components and prefix != "general":
        if "general" in namespaces:
            reason = f"General field {prefix!r} requires the general namespace"
        elif "reserved" in namespaces:
            reason = f"Path prefix {prefix!r} is a reserved expression name"
        else:
            reason = f"Unknown path prefix {prefix!r}"
        raise RoleResolutionError(prefix, None, None, reason)
    return ".".join(parts)


def _path_parts(path: str) -> list[str]:
    parts = path.split(".")
    if not all(parts):
        raise ValueError(f"{path!r}: expected a nonempty dotted path")
    return parts


def _logical_value(node: object) -> YamlValue:
    """Copy logical data without calling model serializers or validators."""
    if isinstance(node, BaseModel):
        result: YamlMap = {}
        for name in type(node).model_fields:
            value = _model_field_value(node, name)
            if name in node.model_fields_set or value is not None:
                result[name] = _logical_value(value)
        for name, value in (node.model_extra or {}).items():
            result[name] = _logical_value(value)
        return result
    if isinstance(node, dict):
        mapping: YamlMap = {}
        for name, value in node.items():
            if not isinstance(name, str):
                raise TypeError("Logical YAML mappings require string keys")
            mapping[name] = _logical_value(value)
        return mapping
    if isinstance(node, list):
        return [_logical_value(value) for value in node]
    return TypeAdapter[YamlValue](YamlValue).validate_python(deepcopy(node))


def _retain_component_defaults(model: BaseModel) -> None:
    """Persist generated defaults without turning absent optional leaves into null.

    DocumentStore projects field presence. Capture native Pydantic defaults
    when adding a component so reload and seed do not regenerate factory values.
    This changes presence only, never validation or values.
    """
    for name, field in type(model).model_fields.items():
        value = getattr(model, name)
        if field.default_factory is not None or value is not None:
            model.model_fields_set.add(name)
        if isinstance(value, BaseModel):
            _retain_component_defaults(value)


def add_component_to_draft(
    draft: SetupDocument | PointDocument,
    name: str,
    kind: str,
    fields: YamlMap,
    source: Path,
) -> None:
    """Add one original registered model to an independent document draft.

    Internal entry helper, not a package export. Inputs are working-unit YAML
    fields. The original model owns validation; the store owns commit.
    Invalid names, fields, values or duplicates leave the draft unchanged.
    """
    validate_component_name(name, source=source)
    model = component_registry.get(kind, source=source, component=name)
    if name in draft.components:
        raise ValueError(f"Component {name!r} already exists")
    component = model.model_validate({"kind": kind, **fields})
    _retain_component_defaults(component)
    sources = _sources_after_write(draft.provenance, name, component, _manual_source())
    draft.components[name] = component
    draft.provenance = sources


def _manual_source() -> YamlMap:
    return {
        "source": "manual",
        "kind": None,
        "run_id": None,
        "at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "stderr": None,
    }


def _model_field_value(node: BaseModel, name: str) -> object:
    """Read logical data, never a class attribute shadowed by an extra key."""
    if name in type(node).model_fields:
        return getattr(node, name)
    return (node.model_extra or {}).get(name)


def _field_value(node: object, path: str) -> object:
    """Read a dotted logical model/mapping path; missing or null returns None."""
    for part in path.split("."):
        if isinstance(node, BaseModel):
            node = _model_field_value(node, part)
        elif isinstance(node, dict):
            node = node.get(part)
        else:
            return None
    return node


def _sources_after_write(
    sources: dict[str, YamlMap], path: str, value: object, metadata: YamlMap
) -> dict[str, YamlMap]:
    """Prepare sources for an accepted field and its written leaves without mutation."""
    result = deepcopy(sources)
    if isinstance(value, BaseModel):
        value = value.model_dump(
            exclude_unset=True, serialize_as_any=True, warnings=False
        )
    tree = TypeAdapter[YamlValue](YamlValue).validate_python(value)

    def include(field: str, node: YamlValue) -> None:
        result.setdefault(field, {})
        if isinstance(node, dict):
            for name, child in node.items():
                include(f"{field}.{name}", child)

    include(path, tree)
    for field, source in result.items():
        if field == path or field.startswith(f"{path}."):
            source.update(deepcopy(metadata))
            source.pop("cloned_from", None)
    return result


def document_meta(
    document: SetupDocument | PointDocument, path: str
) -> Provenance | None:
    """Return an isolated source for this document's actual dotted path.

    Missing values, null or missing sources return None. Invalid empty paths or
    empty path segments raise ValueError. Reads use only this document, without
    file I/O, role expansion or model hooks; stderr stays in working units.
    """
    parts = _path_parts(path)
    root = document if parts[0] == "general" else document.components
    if _field_value(root, path) is None or path not in document.provenance:
        return None
    return TypeAdapter(Provenance).validate_python(deepcopy(document.provenance[path]))


@contextmanager
def stage_component(
    draft: SetupDocument | PointDocument, name: str, field: str
) -> Generator[ComponentSchema]:
    """Yield an independent working-unit component candidate for a draft edit.

    draft is a complete setup or point draft. name identifies an existing
    component; an unknown name raises KeyError. field is the accepted relative
    dotted path within it, such as t1 or wiring.ch, used for source bookkeeping.
    Normal exit validates the original complete model and accepts its value
    with a fresh manual source, clearing cloned_from on the path and descendants
    even for an unchanged value.
    A body exception or validation failure leaves the draft and sources intact.
    The owning store commits the complete document. No I/O occurs here.
    """
    candidate = draft.components[name].model_copy(deep=True)
    yield candidate
    accepted = type(candidate).model_validate(
        candidate.model_dump(exclude_unset=True, serialize_as_any=True, warnings=False)
    )
    sources = _sources_after_write(
        draft.provenance,
        f"{name}.{field}",
        _field_value(accepted, field),
        _manual_source(),
    )
    draft.components[name] = accepted
    draft.provenance = sources


@overload
def stage_general(
    draft: SetupDocument, field: str
) -> AbstractContextManager[SetupGeneral]: ...


@overload
def stage_general(
    draft: PointDocument, field: str
) -> AbstractContextManager[PointGeneral]: ...


@contextmanager
def stage_general(
    draft: SetupDocument | PointDocument, field: str
) -> Generator[SetupGeneral | PointGeneral]:
    """Yield independent metadata, accepting it into draft on normal exit.

    draft is a complete setup or point draft. The candidate is SetupGeneral for
    setup, PointGeneral for a point. Successful validation accepts the field
    with its manual source; a body exception or ValidationError leaves both unchanged.
    This does no I/O and does not enforce entry identity or document references.
    """
    candidate = draft.general.model_copy(deep=True)
    yield candidate
    fields = candidate.model_dump(exclude_unset=True, warnings=False)
    if isinstance(draft, SetupDocument):
        general = SetupGeneral.model_validate(fields)
        draft.provenance = _sources_after_write(
            draft.provenance,
            f"general.{field}",
            _field_value(general, field),
            _manual_source(),
        )
        draft.general = general
    else:
        point_general = PointGeneral.model_validate(fields)
        draft.provenance = _sources_after_write(
            draft.provenance,
            f"general.{field}",
            _field_value(point_general, field),
            _manual_source(),
        )
        draft.general = point_general


def _read_field(node: _FieldNode, name: str, path: str) -> object:
    if isinstance(node, BaseModel):
        if name not in type(node).model_fields and name not in (node.model_extra or {}):
            raise AttributeError(f"{path}.{name}: field is not set")
        value = _model_field_value(node, name)
        if name not in node.model_fields_set and value is None:
            raise AttributeError(f"{path}.{name}: field is not set")
        return value
    return node[name]


def _field_container(value: object, path: str) -> _FieldNode:
    # Preserve models inside mappings rather than projecting them to YAML.
    if isinstance(value, (BaseModel, dict)):
        return value
    raise AttributeError(f"{path}: not a field container")


def _write_field(node: _FieldNode, name: str, value: object) -> None:
    copied = deepcopy(value)
    if isinstance(node, BaseModel):
        if name in type(node).model_fields:
            setattr(node, name, copied)
        elif type(node).model_config.get("extra") == "allow":
            extras = node.model_extra
            if extras is None:
                raise ValueError(f"{name!r}: model has no extra field storage")
            extras[name] = copied
            node.model_fields_set.add(name)
        else:
            # The original model applies forbid/ignore, not a registry precheck.
            type(node).model_validate(
                {
                    **node.model_dump(exclude_unset=True, serialize_as_any=True),
                    name: copied,
                }
            )
    else:
        node[name] = copied


class FieldView:
    _model: Callable[[], _FieldNode]
    _edit: Callable[[str], AbstractContextManager[_FieldNode]]
    _path: str

    def __init__(
        self,
        model: Callable[[], _FieldNode],
        edit: Callable[[str], AbstractContextManager[_FieldNode]],
        path: str,
    ) -> None:
        """Bind a container to memory reads and atomic candidate edits.

        model returns the latest container without I/O. edit(relative_path)
        yields an independent candidate and validates the complete owning model
        on normal exit. Rejection leaves values and sources unchanged. path is
        the full logical container path used in errors.
        """
        self._model = model
        self._edit = edit
        self._path = path

    def __getattr__(self, name: str) -> YamlValue | FieldView:
        """Read one field/key; missing fields raise AttributeError, without I/O."""
        try:
            return self[name]
        except KeyError as cause:
            raise AttributeError(f"{self._path}.{name}: field is not set") from cause

    def __getitem__(self, name: str) -> YamlValue | FieldView:
        """Read one literal key, returning a live child view for dict/model values.

        Scalars and lists are independent YAML values. Missing mapping keys raise
        KeyError; unset null-default model fields raise AttributeError. Empty or
        dotted keys cannot be addressed by set/meta.
        """
        value = _read_field(self._model(), name, self._path)
        if isinstance(value, (BaseModel, dict)):
            return FieldView(
                lambda: _field_container(
                    _read_field(self._model(), name, self._path), f"{self._path}.{name}"
                ),
                lambda field: self._edit_child(name, field),
                f"{self._path}.{name}",
            )
        serialized = TypeAdapter(object).dump_python(value, serialize_as_any=True)
        return deepcopy(TypeAdapter(YamlValue).validate_python(serialized))

    @contextmanager
    def _edit_child(self, name: str, field: str) -> Generator[_FieldNode]:
        with self._edit(f"{name}.{field}") as parent:
            child = _field_container(
                _read_field(parent, name, self._path), f"{self._path}.{name}"
            )
            if isinstance(parent, BaseModel):
                parent.model_fields_set.add(name)
            yield child

    def __setattr__(self, name: str, value: object) -> None:
        """Write one field/key through the complete owning model's validation."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def __setitem__(self, name: str, value: object) -> None:
        """Copy into an atomic candidate; validation/body failures reject the write."""
        with self._edit(name) as draft:
            _write_field(draft, name, value)


class GeneralView(FieldView):
    _general_model: Callable[[], SetupGeneral | PointGeneral]
    _extension: FieldView

    def __init__(
        self,
        model: Callable[[], SetupGeneral | PointGeneral],
        edit: Callable[[str], AbstractContextManager[SetupGeneral | PointGeneral]],
    ) -> None:
        super().__init__(model, edit, "general")
        self._general_model = model

        @contextmanager
        def edit_extension(name: str) -> Generator[_FieldNode]:
            with edit(f"ext.{name}") as draft:
                yield _field_container(draft.ext, "general.ext")
                draft.model_fields_set.add("ext")

        self._extension = FieldView(
            lambda: _field_container(model().ext, "general.ext"),
            edit_extension,
            "general.ext",
        )

    @property
    def ext(self) -> FieldView:
        return self._extension

    @property
    def description(self) -> str | None:
        return self._general_model().description

    @description.setter
    def description(self, value: str | None) -> None:
        self["description"] = value


class ComponentView:
    _model: Callable[[], ComponentSchema]
    _fields: FieldView

    def __init__(
        self,
        model: Callable[[], ComponentSchema],
        edit: Callable[[str], AbstractContextManager[ComponentSchema]],
        path: str,
    ) -> None:
        """Bind a component to memory reads and complete-model field edits.

        model supplies the latest snapshot. edit(relative_path) yields a detached
        candidate, validates it on normal exit, and accepts its value and source.
        path is the component name. Dict/model reads return live child views
        using the same callback; scalar/list reads are independent YAML values.
        No read performs I/O.
        """
        self._model = model
        self._fields = FieldView(model, edit, path)

    @property
    def kind(self) -> str:
        """Return the component's declared kind."""
        return self._model().kind

    def __getattr__(self, name: str) -> YamlValue | FieldView:
        """Read a field; missing/unset fields raise AttributeError."""
        return self._fields.__getattr__(name)

    def __setattr__(self, name: str, value: object) -> None:
        """Validate and accept a field and source, or leave both unchanged."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._fields[name] = value


def _accepted_source(provenance: Provenance, ledger: Path, entry_id: str) -> YamlMap:
    """Validate and isolate source fields before touching any draft."""
    accepted = TypeAdapter(Provenance).validate_python(asdict(provenance))
    validate_source(accepted, ledger, entry_id)
    metadata = TypeAdapter[YamlMap](YamlMap).validate_python(asdict(accepted))
    metadata.pop("cloned_from", None)
    return metadata


def _set_document_value(
    draft: SetupDocument | PointDocument,
    path: str,
    value: YamlValue,
    *,
    metadata: YamlMap | None,
) -> None:
    """Share field validation and acceptance across persistent and memory edits."""
    parts = _path_parts(path)
    if len(parts) < 2:
        raise ValueError(f"{path!r}: expected a dotted field path")
    edit_root: AbstractContextManager[BaseModel]
    if parts[0] == "general":
        edit_root = stage_general(draft, ".".join(parts[1:]))
    else:
        if parts[0] not in draft.components:
            raise AttributeError(f"Unknown component {parts[0]!r}")
        edit_root = stage_component(draft, parts[0], ".".join(parts[1:]))
    with edit_root as root:
        node: _FieldNode = root
        for index, name in enumerate(parts[1:-1], start=1):
            parent_path = ".".join(parts[:index])
            if isinstance(node, BaseModel):
                child = _read_field(node, name, parent_path)
                node.model_fields_set.add(name)
            else:
                child = node[name]
            if not isinstance(child, (BaseModel, dict)):
                raise AttributeError(f"{parent_path}.{name}: not a field container")
            node = child

        @contextmanager
        def edit_node(_name: str) -> Generator[_FieldNode]:
            yield node

        FieldView(lambda: node, edit_node, ".".join(parts[:-1]))[parts[-1]] = value
    if metadata is not None:
        root = draft if parts[0] == "general" else draft.components
        draft.provenance = _sources_after_write(
            draft.provenance, path, _field_value(root, path), metadata
        )


class EditView:
    def __init__(
        self, draft: SetupDocument | PointDocument, *, ledger: Path, entry_id: str
    ) -> None:
        self._draft = draft
        self._ledger = ledger
        self._entry_id = entry_id
        register_view_read(
            self,
            draft,
            None,
            source_metadata=lambda provenance: _accepted_source(
                provenance, ledger, entry_id
            ),
        )

    @property
    def general(self) -> GeneralView:
        return GeneralView(lambda: self._draft.general, self._edit_general)

    @contextmanager
    def _edit_general(self, field: str) -> Generator[SetupGeneral | PointGeneral]:
        with stage_general(self._draft, field) as candidate:
            yield candidate

    def set(
        self, path: str, value: YamlValue, *, provenance: Provenance | None = None
    ) -> None:
        """Accept a logical dotted path and working-unit value into this draft.

        Omitted provenance records manual acceptance. An explicit source keeps
        its five fixed fields and clears clone origin; non-manual event ids must
        belong to this entry's ledger. Written containers accept their leaves
        together. Invalid paths, schema values or source references raise before
        acceptance. The enclosing view.edit() validates the complete document
        and owns commit, conflict and publication. Empty or dotted mapping keys
        cannot be addressed through this dotted syntax.
        """
        metadata = (
            None
            if provenance is None
            else _accepted_source(provenance, self._ledger, self._entry_id)
        )
        _set_document_value(self._draft, path, value, metadata=metadata)

    @property
    def description(self) -> str | None:
        return self._draft.general.description

    @description.setter
    def description(self, value: str | None) -> None:
        self.general.description = value

    def __getattr__(self, name: str) -> ComponentView:
        if name not in self._draft.components:
            raise AttributeError(f"Unknown component {name!r}")
        return ComponentView(
            lambda: self._draft.components[name],
            lambda field: self._edit_component(name, field),
            name,
        )

    @contextmanager
    def _edit_component(self, name: str, field: str) -> Generator[ComponentSchema]:
        with stage_component(self._draft, name, field) as candidate:
            yield candidate


class SetupView:
    def __init__(
        self,
        store: DocumentStore[SetupDocument],
        source: Path,
        *,
        ledger: Path,
        entry_id: str,
    ) -> None:
        """Bind one complete template Store and its source path without I/O.

        Reads use the cached working-unit snapshot. All edits commit only this
        template; existing points neither change nor participate in validation.
        ResultEntry supplies the validated store, source, identity and ledger path.
        """
        self._store = store
        self._source = source
        self._ledger = ledger
        self._entry_id = entry_id
        self._edit_document = store.edit
        register_view_read(self, store, source)

    @property
    def general(self) -> GeneralView:
        return GeneralView(lambda: self._store.snapshot().general, self._edit_general)

    @contextmanager
    def _edit_general(self, field: str) -> Generator[SetupGeneral]:
        with self._edit_document() as draft, stage_general(draft, field) as candidate:
            yield candidate

    @property
    def description(self) -> str | None:
        return self._store.snapshot().general.description

    @description.setter
    def description(self, value: str | None) -> None:
        with self.edit() as draft:
            draft.description = value

    @contextmanager
    def edit(self) -> Generator[EditView]:
        with self._edit_document() as draft:
            yield EditView(draft, ledger=self._ledger, entry_id=self._entry_id)

    def meta(self, path: str) -> Provenance | None:
        """Return an independent cached source in working units, or None."""
        return document_meta(self._store.snapshot(), path)

    def refresh(self) -> None:
        self._store.refresh()

    def add_component(self, name: str, *, kind: str, **fields: YamlValue) -> None:
        with self._edit_document() as draft:
            add_component_to_draft(draft, name, kind, fields, self._source)

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
            self._edit_document() as draft,
            stage_component(draft, name, field) as candidate,
        ):
            yield candidate
