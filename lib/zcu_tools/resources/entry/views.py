"""Parameter document reads and transaction views over one typed store.

Copy incoming values before assignment: notebook models can validate assignment.
"""

from __future__ import annotations

from collections.abc import Collection, Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from operator import is_
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, TypeAdapter

from zcu_tools.format_version import YamlMap, YamlValue
from zcu_tools.resources.context.library import ModuleLibrary, library_revision
from zcu_tools.resources.document_store import DocumentRevision, DocumentStore
from zcu_tools.resources.parameter_fields import stage_model
from zcu_tools.resources.parameter_metadata import read_metadata, write_metadata

from ._parameter_origin import ParameterOrigin
from .aliases import AliasFile, AliasValue
from .errors import RoleResolutionError
from .field_views import (
    ComponentView as ComponentView,
)
from .field_views import (
    FieldNode,
    model_field_value,
    read_field,
)
from .field_views import (
    FieldView as FieldView,
)
from .field_views import (
    GeneralView as GeneralView,
)
from .registry import component_registry
from .schema import (
    ComponentSchema,
    ParameterContent,
    ParameterDocument,
    ParameterGeneral,
    validate_component_name,
)
from .schema_fields import declared_unit

if TYPE_CHECKING:
    from .parameter_view import ParameterView


type _ViewDocument = DocumentStore[ParameterDocument] | ParameterDocument


@dataclass
class _ViewRead:
    """Package-owned access for parameter and edit views without reserving method names.

    document is the store/draft owned by this view.
    source is the bound YAML path, or None for an edit draft with no source getter.
    origin is the parameter view's constructor identity, or None for edit drafts.
    """

    document: _ViewDocument
    source: Path | None
    origin: ParameterOrigin | None
    content: ParameterContent | None = None
    revision: DocumentRevision | None = None


def _validate_view_component_name(name: str, *, source: Path | None) -> None:
    # Local import avoids a cycle: ParameterView uses the shared view machinery.
    from .parameter_view import ParameterView

    if any(
        name in dir(view_type) and not name.startswith("_")
        for view_type in (ParameterView, EditView)
    ):
        raise ValueError(
            f"{source}: invalid component name {name!r}; would shadow a view member {name!r}"
        )


def validate_view_components(
    document: ParameterDocument | ParameterContent, *, source: Path | None
) -> None:
    """Reject components that shadow any ParameterView/EditView public member.

    Entry-internal helper, not a package export. document is a typed cached
    document or store validation candidate; source identifies its YAML path,
    or None for an unbound edit draft. Raise ValueError naming the conflicting
    member before binding/publication. No mutation, model hooks or I/O occurs.
    Public class members are discovered on each call, without a reserved table.
    """
    names = (
        document.component_names()
        if isinstance(document, ParameterContent)
        else document.components
    )
    for name in names:
        validate_component_name(name, source=source)
        _validate_view_component_name(name, source=source)


def register_view_read(
    view: object,
    document: _ViewDocument,
    source: Path | None,
    *,
    origin: ParameterOrigin | None = None,
) -> None:
    """Bind one entry-internal read record during view construction, without I/O.

    Not a package export. document is the view's validated store/draft, source
    its YAML path (None for EditView), and origin its full parameter identity
    (None for edit drafts). The private attribute cannot collide with legal components.
    Already bound handles or components shadowing any public view member raise
    ValueError; objects without an instance dictionary raise TypeError.
    Validate the cached document without I/O before binding. No model hook runs.
    """
    if isinstance(document, DocumentStore):
        validate_view_components(ParameterContent(document.content()), source=source)
    else:
        validate_view_components(document, source=source)
    state = vars(view)
    if "_entry_read" in state:
        raise ValueError("Entry view is already bound")
    state["_entry_read"] = _ViewRead(document, source, origin)


def _view_read(view: object) -> _ViewRead:
    # One owner-local reflective boundary keeps private state off the view namespace.
    binding: object = vars(view)["_entry_read"]
    if not isinstance(binding, _ViewRead):
        raise TypeError("Invalid entry view read binding")
    return binding


def parameter_origin(view: ParameterView) -> ParameterOrigin:
    """Return a global or point view's immutable constructor identity without I/O.

    Entry-internal helper, not a package export. An edit draft binding raises
    ValueError; no destination identity is used to fill missing information.
    """
    origin = _view_read(view).origin
    if origin is None:
        raise ValueError("View has no parameter origin")
    return origin


def snapshot(
    view: ParameterView | EditView,
) -> ParameterDocument:
    """Project the complete cached document or edit draft without I/O.

    Bound content is validated as the complete document, including model-level
    transformations and checks. Drafts are copied without revalidation. The
    result is isolated and uses working units. Mutating it changes no file.
    Use parameter_content() for reads that must leave unused fields opaque.
    """
    document = _view_read(view).document
    if isinstance(document, DocumentStore):
        return document.snapshot()
    return document.model_copy(deep=True)


def parameter_content(view: ParameterView) -> ParameterContent:
    """Copy cached resource content without validating unused component fields."""
    return cached_parameter_content(view).model_copy(deep=True)


def cached_parameter_content(view: ParameterView) -> ParameterContent:
    """Owner-internal cached content; defaults stay stable until the file changes."""
    binding = _view_read(view)
    document = binding.document
    if not isinstance(document, DocumentStore):
        raise TypeError("Parameter content requires a bound document")
    if binding.content is None or binding.revision != document.document_revision:
        binding.content = ParameterContent.from_document(
            document.content(), source=document.source_path
        )
        binding.revision = document.document_revision
    return binding.content


def document_revision(
    view: ParameterView | ModuleLibrary | AliasFile,
) -> DocumentRevision:
    """Return a parameter, library or aliases handle's held revision without I/O.

    Missing library/aliases files have revision=None. An aliases revision can
    identify invalid bytes; it does not imply format or session syntax validity.
    Edit drafts have no installed version.
    """
    if isinstance(view, ModuleLibrary):
        return library_revision(view)
    if isinstance(view, AliasFile):
        return view.read().revision
    document = _view_read(view).document
    if not isinstance(document, DocumentStore):
        raise TypeError("Edit drafts have no installed revision")
    return document.document_revision


def source_path(view: ParameterView) -> Path:
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
    value uses working units. metadata is a detached copy of the caller's
    opaque mapping stored at editable_path, or None when absent or the path is
    unaddressable. Rows never expose opaque future data.
    """

    segments: tuple[str, ...]
    editable_path: str | None
    value: YamlValue
    metadata: YamlMap | None


def document_fields(
    document: ParameterDocument | ParameterContent,
) -> tuple[DocumentField, ...]:
    """List isolated logical rows without I/O, validation or serialization hooks.

    Expand models/mappings in data order; lists, null and empty mappings remain
    single rows. Preserve unaddressable literal keys with editable_path=None
    and no dotted metadata lookup. Invalid logical YAML values raise TypeError
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
        metadata = read_metadata(document.metadata, path) if path is not None else None
        rows.append(DocumentField(segments, path, value, metadata))

    if document.general is not None:
        append_rows(("general",), _logical_value(document.general))
    for name, component in document.components.items():
        append_rows((name,), _logical_value(component))
    return tuple(rows)


def document_unit(
    document: ParameterDocument | ParameterContent, path: str
) -> str | None:
    """Read a fixed field's static schema unit at an actual dotted path.

    Return the nonempty json_schema_extra.unit label without converting values.
    Roles are not expanded. The final declared BaseModel field supplies its
    label regardless of whether its value is a list, mapping or unset optional.
    Ext namespaces, child paths inside mapping/list data, missing ancestors or
    fields, and fields without static unit metadata return None. Existing nested
    BaseModel fields can supply labels. No leaf value is read, and no schema
    callback, model hook or I/O runs.
    Invalid/unaddressable dotted syntax or malformed unit metadata raises
    ValueError. Labels are not inferred from field names or other values.
    """
    parts = _path_parts(path)
    node: object = (
        document.general if parts[0] == "general" else document.components.get(parts[0])
    )
    for index, part in enumerate(parts[1:], start=1):
        if part == "ext" or not isinstance(node, BaseModel):
            return None
        field = type(node).model_fields.get(part)
        if field is None:
            return None
        if index == len(parts) - 1:
            return declared_unit(field, path)
        node = model_field_value(node, part)
    return None


def document_value(
    document: ParameterDocument | ParameterContent, path: str
) -> YamlValue:
    """Read an actual dotted component/general path from this document only.

    Return an isolated working-unit YAML value, including explicit null and
    logical containers. Never run validators, serializers or read another file.
    Invalid or unaddressable syntax raises ValueError. Missing components,
    fields, unset optional values or non-container ancestors raise AttributeError.
    Model fields and extras take precedence over class attributes; hidden
    forward-minor data stays hidden. Roles are not expanded here.
    """
    parts = _path_parts(path)
    if isinstance(document, ParameterContent):
        return document.value(path)
    node: object = document if parts[0] == "general" else document.components
    for index, part in enumerate(parts):
        prefix = ".".join(parts[: index + 1])
        if isinstance(node, BaseModel):
            node = read_field(node, part, ".".join(parts[:index]))
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise AttributeError(f"{prefix}: path {path!r} is not readable")
    return _logical_value(node)


def _content_path(
    document: ParameterContent, parts: list[str], roles: Mapping[str, AliasValue]
) -> str:
    prefix = parts[0]
    if prefix == "general":
        return ".".join(parts)
    names = document.component_names()
    if prefix in names:
        return ".".join(parts)
    if prefix in roles:
        target = roles[prefix]
        if isinstance(target, str) and target in names:
            return ".".join((target, *parts[1:]))
    raise RoleResolutionError(
        prefix, None, None, f"Unknown parameter namespace {prefix!r}"
    )


def resolve_path(
    document: ParameterDocument | ParameterContent,
    path: str,
    *,
    roles: Mapping[str, AliasValue],
    reserved_names: Collection[str] = (),
) -> str:
    """Resolve a dotted path's first segment using captured role-to-component names.

    Read only this document. Preserve the remaining segments without checking
    leaf or library-key existence. Component names and the general namespace
    stay unchanged. Empty paths or segments raise ValueError. A prefix shared
    by roles, components, logical general fields or reserved_names raises
    RoleResolutionError identifying the namespaces. Unknown prefixes and roles
    bound to lists or missing components also raise RoleResolutionError. General fields
    require the general prefix; bare metadata keys are not supported.
    No I/O, model validation, serialization or mutation occurs.
    """
    parts = _path_parts(path)
    if isinstance(document, ParameterContent):
        return _content_path(document, parts, roles)
    return _typed_path(document, parts, roles, reserved_names)


def _typed_path(
    document: ParameterDocument,
    parts: list[str],
    roles: Mapping[str, AliasValue],
    reserved_names: Collection[str],
) -> str:
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
        if isinstance(target, list):
            raise RoleResolutionError(
                prefix,
                None,
                None,
                "Alias is a list",
                message=f"${prefix} is a list; a path requires one component name",
            )
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
            value = model_field_value(node, name)
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
    draft: ParameterDocument,
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
    _validate_view_component_name(name, source=source)
    model = component_registry.get(kind, source=source, component=name)
    if name in draft.components:
        raise ValueError(f"Component {name!r} already exists")
    component = model.model_validate({"kind": kind, **fields})
    _retain_component_defaults(component)
    draft.components[name] = component


def document_meta(
    document: ParameterDocument | ParameterContent, path: str
) -> YamlMap | None:
    """Return this document's isolated opaque metadata for a dotted path, or None.

    Invalid empty paths or empty path segments raise ValueError. The value at
    path is not looked up: metadata may exist for an absent value and an empty
    stored mapping returns {}. No key is interpreted and no file I/O, role
    expansion or model hook runs.
    """
    _path_parts(path)
    return read_metadata(document.metadata, path)


@contextmanager
def stage_component(draft: ParameterDocument, name: str) -> Generator[ComponentSchema]:
    """Yield an independent working-unit component candidate for a draft edit.

    draft is a complete global or point draft. name identifies an existing
    component; an unknown name raises KeyError. Normal exit validates the
    original complete model and replaces the draft component. Metadata is
    neither read nor changed. A body exception or validation failure leaves the
    draft intact. The owning store commits the complete document. No I/O occurs.
    """

    def publish(accepted: ComponentSchema) -> None:
        draft.components[name] = accepted

    with stage_model(draft.components[name], publish) as candidate:
        yield candidate


@contextmanager
def stage_general(draft: ParameterDocument) -> Generator[ParameterGeneral]:
    """Yield independent general fields, accepting them into draft on normal exit.

    draft is a complete parameter draft. Successful validation replaces its
    general fields; a body exception or ValidationError leaves it unchanged.
    Metadata is untouched. This does no I/O and does not enforce entry identity.
    """

    def publish(accepted: ParameterGeneral) -> None:
        draft.general = accepted

    with stage_model(draft.general, publish) as candidate:
        yield candidate


def _set_document_value(
    draft: ParameterDocument,
    path: str,
    value: YamlValue | complex,
) -> None:
    """Validate and accept one dotted value into a draft, leaving metadata unchanged."""
    parts = _path_parts(path)
    if len(parts) < 2:
        raise ValueError(f"{path!r}: expected a dotted field path")
    edit_root: AbstractContextManager[BaseModel]
    if parts[0] == "general":
        edit_root = stage_general(draft)
    else:
        if parts[0] not in draft.components:
            raise AttributeError(f"Unknown component {parts[0]!r}")
        edit_root = stage_component(draft, parts[0])
    with edit_root as root:
        node: FieldNode = root
        freeform = False
        for index, name in enumerate(parts[1:-1], start=1):
            parent_path = ".".join(parts[:index])
            if isinstance(node, BaseModel):
                child = read_field(node, name, parent_path)
                field = type(node).model_fields.get(name)
                # Pydantic's annotation type omits runtime PEP 695 aliases.
                freeform = field is not None and is_(field.annotation, YamlMap)
                node.model_fields_set.add(name)
            else:
                if freeform and name not in node:
                    node[name] = {}
                child = node[name]
            if not isinstance(child, (BaseModel, dict)):
                raise AttributeError(f"{parent_path}.{name}: not a field container")
            node = child

        @contextmanager
        def edit_node(_name: str) -> Generator[FieldNode]:
            yield node

        FieldView(lambda: node, edit_node, ".".join(parts[:-1]))[parts[-1]] = value


class EditView:
    """An isolated value/metadata draft whose components cannot shadow view members.

    Read or set component and general values, add registered components with
    add_component, and replace metadata with set_meta in one edit session.
    All operations share one draft. The enclosing edit validates the complete
    document and commits only on normal exit. Value writes keep metadata.
    """

    def __init__(self, draft: ParameterDocument, *, source: Path) -> None:
        """Bind an edit session to draft, rejecting view-shadowing component names.

        source is the owning parameters.yaml path, used as the error origin
        when adding a component. Binding does not run model hooks, read disk,
        mutate the draft or commit; the enclosing edit owns publication.
        Shadowing raises ValueError naming the member.
        """
        self._source = source
        self._draft = draft
        register_view_read(self, draft, None)

    def add_component(self, name: str, *, kind: str, **fields: YamlValue) -> None:
        """Add a new named component to this edit session without committing.

        kind must be registered by the caller. fields supplies working-unit
        values; omitted fields use the registered model's defaults. Invalid
        names, unknown kinds, duplicate names or invalid fields raise without
        changing the draft. The enclosing edit commits all changes together.
        """
        add_component_to_draft(self._draft, name, kind, fields, self._source)

    @property
    def general(self) -> GeneralView:
        return GeneralView(lambda: self._draft.general, self._edit_general)

    @contextmanager
    def _edit_general(self, _field: str) -> Generator[ParameterGeneral]:
        with stage_general(self._draft) as candidate:
            yield candidate

    def set(self, path: str, value: YamlValue | complex) -> None:
        """Accept a logical dotted path and working-unit value into this draft.

        Native complex scalars encode through the core codec before the original
        schema validation; the stored value remains a YAML tag. Containers
        replace their whole value. Inside declared YamlMap fields, missing
        intermediate keys become empty mappings. Existing non-container values
        and unknown typed paths raise without changing the draft, as do invalid
        schema values. Metadata at every path is kept unchanged, including
        for same-value writes. Empty or dotted mapping keys cannot be addressed
        through this dotted syntax.
        """
        _set_document_value(self._draft, path, value)

    def meta(self, path: str) -> YamlMap | None:
        """Read isolated opaque draft metadata; absent paths return None.

        path uses nonempty dotted segments. No value lookup or key
        interpretation occurs; an empty stored mapping returns {}.
        """
        return read_metadata(self._draft.metadata, path)

    def set_meta(self, path: str, metadata: object) -> None:
        """Replace one dotted path's complete draft metadata with a detached copy.

        metadata accepts only string keys and YAML null/bool/int/float/str/list/
        mapping values, checked strictly. Invalid representation raises
        ValidationError before changing the draft. {} clears the path. No value
        lookup, merge, unit conversion or key interpretation occurs; values and
        other paths are unchanged. DocumentStore's mapping-leaf merge and
        conflict rules apply when the enclosing edit commits.
        """
        write_metadata(self._draft.metadata, path, metadata)
        self._draft.model_fields_set.add("metadata")

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
    def _edit_component(self, name: str, _field: str) -> Generator[ComponentSchema]:
        with stage_component(self._draft, name) as candidate:
            yield candidate
