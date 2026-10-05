"""Setup and transaction views over one typed document store.

Copy incoming values before assignment: notebook models can validate assignment.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import overload

from pydantic import BaseModel, TypeAdapter

from zcu_tools.format_version import YamlMap, YamlValue
from zcu_tools.resources.document_store import DocumentStore

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

type _FieldNode = BaseModel | dict[str, object]


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


def _field_value(node: object, path: str) -> object:
    for part in path.split("."):
        if isinstance(node, BaseModel):
            node = getattr(node, part, None)
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
    """Project one independent source only when its cached value exists."""
    root = document if path.split(".")[0] == "general" else document.components
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
        value: object = getattr(node, name)
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
        if (
            name not in type(node).model_fields
            and node.model_config.get("extra") != "allow"
        ):
            # The original model applies forbid/ignore, not a registry precheck.
            type(node).model_validate(
                {
                    **node.model_dump(exclude_unset=True, serialize_as_any=True),
                    name: copied,
                }
            )
            return
        setattr(node, name, copied)
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


class EditView:
    def __init__(
        self, draft: SetupDocument | PointDocument, *, ledger: Path, entry_id: str
    ) -> None:
        self._draft = draft
        self._ledger = ledger
        self._entry_id = entry_id

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
        metadata: YamlMap | None = None
        if provenance is not None:
            accepted = TypeAdapter(Provenance).validate_python(asdict(provenance))
            validate_source(accepted, self._ledger, self._entry_id)
            metadata = TypeAdapter[YamlMap](YamlMap).validate_python(asdict(accepted))
            metadata.pop("cloned_from", None)
        parts = path.split(".")
        if len(parts) < 2 or not all(parts):
            raise ValueError(f"{path!r}: expected a dotted field path")
        edit_root: AbstractContextManager[BaseModel]
        if parts[0] == "general":
            edit_root = self._edit_general(".".join(parts[1:]))
        else:
            if parts[0] not in self._draft.components:
                raise AttributeError(f"Unknown component {parts[0]!r}")
            edit_root = self._edit_component(parts[0], ".".join(parts[1:]))
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
            root = self._draft if parts[0] == "general" else self._draft.components
            self._draft.provenance = _sources_after_write(
                self._draft.provenance, path, _field_value(root, path), metadata
            )

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
