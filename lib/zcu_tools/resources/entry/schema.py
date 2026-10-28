"""Typed setup document at the persistence boundary."""

import keyword
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar, Literal, Self, get_args, get_origin, override

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    RootModel,
    TypeAdapter,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic_core import (
    PydanticUndefined,
    SchemaSerializer,
    SchemaValidator,
    core_schema,
)

from zcu_tools.format_version import YamlMap, YamlValue
from zcu_tools.resources.parameter_fields import (
    mapping_shape,
    model_field_schema,
    sequence_shape,
)
from zcu_tools.resources.parameter_metadata import (
    PARAMETER_FORMAT,
    PARAMETER_VERSION,
    validate_parameter_header,
)
from zcu_tools.resources.projection import (
    ProjectionPathError,
    projection_child,
    projection_value,
)

__all__ = (
    "PARAMETER_FORMAT",
    "PARAMETER_VERSION",
    "ComponentSchema",
    "PointDocument",
    "PointGeneral",
    "SetupDocument",
    "SetupGeneral",
    "is_forward_minor",
    "validate_component_name",
)


def is_forward_minor(document: Mapping[str, YamlValue], *, source: Path) -> bool:
    """Validate the shared parameter-container 2.x header; report a newer minor.

    Old majors and provenance tables raise with source and versions, exactly as
    the standalone parameter entry does. No data is changed or migrated.
    """
    version = validate_parameter_header(document, source=source)
    return version.minor > PARAMETER_VERSION.minor


def validate_component_name(name: str, *, source: Path | None = None) -> None:
    """Require a public Python identifier, excluding keywords and leading underscores.

    source is the optional document path for diagnostics. Invalid names raise
    ValueError. View namespace collisions belong to view binding, not schema.
    """
    if not name.isidentifier() or name.startswith("_") or keyword.iskeyword(name):
        raise ValueError(
            f"{source}: invalid component name {name!r}; expected a public identifier"
        )


def _ignore_legacy_identity(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: item for key, item in value.items() if key != "entry_id"}
    return value


class SetupGeneral(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: str
    description: str | None = None
    ext: YamlMap = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def ignore_legacy_identity(cls, value: object) -> object:
        """Discard obsolete entry identity without interpreting its value."""
        return _ignore_legacy_identity(value)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        if datetime.fromisoformat(value).utcoffset() != timedelta(0):
            raise ValueError("created_at must be a UTC timestamp")
        return value


class PointGeneral(BaseModel):
    """Independent metadata copied from the seed setup or source point.

    Legacy entry_id is ignored and omitted from explicit writes.
    created_at is renewed on creation, as an ISO-8601 UTC timestamp.
    description is optional human-readable text; None means no description.
    ext is a YAML mapping, empty by default, with no unit conversion.
    Empty keys or keys containing dots cannot be addressed by set/meta.
    Unknown metadata fields and invalid timestamps raise Pydantic ValidationError.
    """

    model_config = ConfigDict(extra="forbid")

    created_at: str
    description: str | None = None
    ext: YamlMap = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def ignore_legacy_identity(cls, value: object) -> object:
        """Discard obsolete entry identity without interpreting its value."""
        return _ignore_legacy_identity(value)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return SetupGeneral.validate_created_at(value)


class ComponentSchema(BaseModel):
    """Base for registered components, requiring only a string kind.

    Definitions own extra policy, validators, hooks and optional containers,
    independently of the parameter-container version. Forbid rejects unknown
    fields, ignore discards them, and allow retains them at every minor version.
    Loading uses model_validate and saving uses model_dump. Mapping keys with
    dots or an empty name cannot be addressed by set/meta dotted paths.
    """

    kind: str


class _ParameterDocument(BaseModel):
    """Shared typed component boundary for complete parameter documents."""

    model_config = ConfigDict(extra="forbid")
    _source: ClassVar[Path] = Path("setup.yaml")
    _upgrade_target: ClassVar[Path | None] = None

    format: str
    format_version: str
    schema_version: int = Field(strict=True, ge=1)
    components: dict[str, ComponentSchema] = Field(default_factory=dict)
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
        # Lookup errors must retain their public entry error type rather than
        # becoming Pydantic field-validator ValueErrors.
        from .registry import component_registry

        if isinstance(obj, Mapping):
            document = TypeAdapter(YamlMap).validate_python(obj)
            is_forward_minor(document, source=cls._source)
            component_registry.require_schema_policy().require_current(
                document, source=cls._source, upgrade_target=cls._upgrade_target
            )
            components = TypeAdapter(dict[str, YamlMap]).validate_python(
                document.get("components", {})
            )
            for name, fields in components.items():
                validate_component_name(name, source=cls._source)
                kind = fields.get("kind")
                if isinstance(kind, str):
                    component_registry.get(kind, source=cls._source, component=name)
        return super().model_validate(
            obj,
            strict=strict,
            extra=extra,
            from_attributes=from_attributes,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @field_validator("components", mode="before")
    @classmethod
    def validate_components(cls, value: object) -> dict[str, ComponentSchema]:
        # Import locally because registered models derive from ComponentSchema.
        from .registry import component_registry

        # A separate validation call keeps the container's unknown-field policy
        # from overriding the original component and nested models' extra policies.
        components = TypeAdapter(dict[str, YamlMap]).validate_python(value)
        result: dict[str, ComponentSchema] = {}
        for name, fields in components.items():
            validate_component_name(name, source=cls._source)
            kind = fields.get("kind")
            model = (
                component_registry.get(kind, source=cls._source, component=name)
                if isinstance(kind, str)
                else ComponentSchema
            )
            # Validators may mutate mapping inputs; keep the caller isolated.
            adapter = TypeAdapter[dict[str, ComponentSchema]](dict[str, model])
            result[name] = adapter.validate_python({name: deepcopy(fields)})[name]
        return result

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, value: object) -> dict[str, YamlMap]:
        return TypeAdapter(dict[str, YamlMap]).validate_python(value, strict=True)


type _DefaultOmissions = (
    Mapping[str, bool | _DefaultOmissions] | Mapping[int, bool | _DefaultOmissions]
)


def _default_omissions(value: object) -> _DefaultOmissions:
    """Keep factory outputs but omit model nulls that were never supplied."""
    if isinstance(value, (list, tuple)):
        return {
            index: child
            for index, item in enumerate(value)
            if (child := _default_omissions(item))
        }
    fields: dict[str, bool | _DefaultOmissions] = {}
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            item = getattr(value, name)
            if item is None and name not in value.model_fields_set:
                fields[name] = True
            elif child := _default_omissions(item):
                fields[name] = child
    elif isinstance(value, dict):
        for name, item in value.items():
            if isinstance(name, str) and (child := _default_omissions(item)):
                fields[name] = child
    return fields


class ParameterContent(RootModel[YamlValue]):
    """Detached working-unit parameter content, decoded only when requested.

    root retains the captured components, general and opaque metadata sections,
    including unknown or malformed unused values. An empty mapping has no general
    or timestamps. Requested leaves use their registered field annotation; whole
    section projections explicitly validate that section. No file I/O occurs.
    """

    root: YamlValue = Field(default_factory=dict)
    _defaults: dict[str, YamlValue] = PrivateAttr(default_factory=dict)
    _source: Path = PrivateAttr(default=Path("snapshot"))

    @classmethod
    def from_document(cls, document: YamlValue, *, source: Path) -> Self:
        """Detach cached YAML with its diagnostic source, never serialized as data."""
        content = cls(deepcopy(document))
        content._source = source
        return content

    @model_serializer
    def serialized_content(self) -> YamlValue:
        """Copy recorded content for output, omitting only legacy general.entry_id.

        Unused and unknown values remain opaque, including general extensions.
        Serialization does not validate sections or mutate the captured source.
        """
        document = deepcopy(self.root)
        if isinstance(document, dict):
            general = document.get("general")
            if isinstance(general, dict):
                general.pop("entry_id", None)
        return document

    def section(self, name: str) -> YamlValue:
        """Copy a recorded section; missing sections raise AttributeError."""
        if not isinstance(self.root, dict):
            raise ValueError("parameter content must be a mapping")
        if name not in self.root:
            raise AttributeError(f"Missing parameter section: {name}")
        return deepcopy(self.root[name])

    def component(self, name: str) -> ComponentSchema:
        """Validate only this complete component, including model-level invariants.

        Unknown names raise AttributeError. Malformed fields and cross-field
        checks raise ValueError naming the component. Other components stay opaque.
        """
        node, model = self._root_field(name)
        if name == "general" or not issubclass(model, ComponentSchema):
            raise ValueError("Expected a component name")
        values = TypeAdapter(YamlMap).validate_python(deepcopy(node))
        for field_name, field in model.model_fields.items():
            if field_name not in values and (
                field.default_factory is not None
                or (not field.is_required() and field.default is not None)
            ):
                values[field_name] = self._default_value(
                    f"{name}.{field_name}", model_field_schema(model, field_name)
                )
        try:
            return model.model_validate(values)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc

    def captured(self) -> Self:
        """Detach raw content with already consumed defaults recorded as values.

        Legacy general.entry_id is omitted. Other unused fields stay opaque and
        no new defaults or validators run. Reusing this serialized content
        therefore cannot regenerate a consumed factory.
        """
        content = self.model_copy(deep=True)
        content.root = self.serialized_content()
        for path, value in sorted(
            self._defaults.items(), key=lambda item: item[0].count(".")
        ):
            # Omitted optional nulls remain omitted: a field-level before rule may
            # reject an explicitly supplied null even though its default is None.
            if value is None:
                continue
            parts = path.split(".")
            if parts[0] != "general":
                parts.insert(0, "components")
            node = content.root
            for part in parts[:-1]:
                if not isinstance(node, dict):
                    raise ValueError(f"{path}: default ancestor must be a mapping")
                node = node[part]
            if not isinstance(node, dict):
                raise ValueError(f"{path}: default parent must be a mapping")
            node[parts[-1]] = deepcopy(value)
        return content

    def project(self) -> "PointDocument":
        """Validate the complete recorded document, including all model checks.

        Requires a complete point document, unlike leaf reads. Missing sections,
        malformed components/general and model invariants raise owner errors.
        No file is read and no missing identity or timestamp is synthesized.
        """
        return PointDocument.model_validate(self.captured().root)

    def projection(self, segments: tuple[str, ...]) -> object:
        """Project one native document subtree, validating only that selection.

        Empty segments explicitly validate the complete document. Components,
        general and nested models retain their full projection checks when chosen
        as a whole. Leaves use field checks; unselected siblings stay opaque.
        Missing keys or invalid indices raise ProjectionPathError naming the first
        bad segment. No files are read and no values are replaced on failure.
        """
        if not segments:
            return self.project().model_dump(mode="python", serialize_as_any=True)
        section, *suffix = segments
        if section == "components":
            return self._component_projection(tuple(suffix))
        if section == "general":
            if not suffix:
                general = self.general
                return None if general is None else general.model_dump(mode="python")
            return self._projection_field((section, *suffix))
        return self._section_projection(section, tuple(suffix))

    def _component_projection(self, segments: tuple[str, ...]) -> object:
        if not segments:
            return {
                name: component.model_dump(mode="python", serialize_as_any=True)
                for name, component in self.components.items()
            }
        name, *fields = segments
        if name not in self.component_names():
            raise ProjectionPathError(name)
        if not fields:
            return self.component(name).model_dump(mode="python", serialize_as_any=True)
        return self._projection_field((name, *fields))

    def _section_projection(self, section: str, suffix: tuple[str, ...]) -> object:
        if section == "metadata":
            # Metadata has no component schema; only the selected opaque subtree
            # is consumed, unless the complete section is explicitly requested.
            return projection_value(
                self.metadata if not suffix else self.section(section), tuple(suffix)
            )
        if section not in PointDocument.model_fields:
            raise ProjectionPathError(section)
        raw = self.section(section)
        if suffix:
            return projection_value(raw, tuple(suffix))
        schema = model_field_schema(PointDocument, section)
        value = SchemaValidator(schema).validate_python(raw)
        return SchemaSerializer(schema).to_python(value, mode="python")

    def _projection_field(self, segments: tuple[str, ...]) -> YamlValue:
        for length in range(2, len(segments) + 1):
            if not segments[length - 1]:
                raise ProjectionPathError("")
            try:
                self.node(".".join(segments[:length]))
            except AttributeError as exc:
                raise ProjectionPathError(segments[length - 1]) from exc
        return self.value(".".join(segments))

    def summary(self, describe: Callable[[object], object]) -> dict[str, object]:
        """Describe sequence leaves and project only the scalar fields emitted.

        describe receives each sequence without consuming its elements. Other
        output leaves retain their field validation. Nested model dictionaries
        are traversed by declared fields, not validated as complete components.
        Missing or malformed fields needed for the output raise owner errors.
        No wire conversion, file access or empty-on-error substitution occurs.
        """
        result: dict[str, object] = {
            name: self.projection((name,))
            for name in ("format", "format_version", "schema_version")
        }
        result["general"] = self._summary_model(PointGeneral, "general", describe)
        result["components"] = {
            name: self._summary_model(self._root_field(name)[1], name, describe)
            for name in self.component_names()
        }
        result["metadata"] = self._summary_opaque(self.metadata, describe)
        return result

    def _summary_model(
        self, model: type[BaseModel], path: str, describe: Callable[[object], object]
    ) -> dict[str, object]:
        node = self.node(path)
        if not isinstance(node, dict):
            raise ValueError(f"{path}: expected a model mapping")
        fields = [
            name for name, field in model.model_fields.items() if not field.exclude
        ]
        extras = [name for name in node if name not in model.model_fields]
        if model.model_config.get("extra") == "allow":
            fields.extend(extras)
        return {
            name: self._summary_field(f"{path}.{name}", model, name, describe)
            for name in fields
        }

    def _summary_field(
        self,
        path: str,
        parent: type[BaseModel],
        name: str,
        describe: Callable[[object], object],
    ) -> object:
        node = self.node(path)
        field = parent.model_fields.get(name)
        annotation = None if field is None else field.annotation
        nested = next(
            (
                item
                for item in (annotation, *get_args(annotation))
                if isinstance(item, type) and issubclass(item, BaseModel)
            ),
            None,
        )
        if nested is not None and isinstance(node, dict):
            return self._summary_model(nested, path, describe)
        if isinstance(node, list):
            return describe(sequence_shape(node, self._field(path)[1]))
        if isinstance(node, dict):
            mapping_shape(node, self._field(path)[1])
            return self._summary_mapping(path, node, describe)
        return self.value(path)

    def _summary_mapping(
        self,
        path: str,
        node: dict[str, YamlValue],
        describe: Callable[[object], object],
    ) -> object:
        if "__complex__" in node:
            return self.value(path)
        return {
            key: self._summary_mapping(f"{path}.{key}", value, describe)
            if isinstance(value, dict)
            else describe(value)
            if isinstance(value, list)
            else self.value(f"{path}.{key}")
            for key, value in node.items()
        }

    @staticmethod
    def _summary_opaque(value: object, describe: Callable[[object], object]) -> object:
        if isinstance(value, dict):
            if "__complex__" in value:
                return deepcopy(value)
            return {
                key: ParameterContent._summary_opaque(child, describe)
                for key, child in value.items()
            }
        return describe(value) if isinstance(value, list) else value

    def component_names(self) -> tuple[str, ...]:
        """List recorded names without decoding any component or general fields."""
        if isinstance(self.root, dict) and "components" not in self.root:
            return ()
        components = self.section("components")
        if not isinstance(components, dict):
            raise ValueError("parameter components must be a mapping")
        return tuple(components)

    def component_kind(self, name: str) -> str:
        """Read only a component's kind; malformed/missing content raises."""
        value = self.value(f"{name}.kind")
        if not isinstance(value, str):
            raise ValueError(f"{name}.kind must be a string")
        return value

    @property
    def components(self) -> dict[str, ComponentSchema]:
        """Decode the complete components section on explicit whole-section use."""
        if self.root == {}:
            return {}
        return {name: self.component(name) for name in self.component_names()}

    @property
    def general(self) -> PointGeneral | None:
        """Decode captured general metadata, or None when absent."""
        if isinstance(self.root, dict) and "general" not in self.root:
            return None
        value = self.section("general")
        return None if value is None else PointGeneral.model_validate(value)

    @property
    def metadata(self) -> dict[str, YamlMap]:
        """Decode the opaque metadata section when explicitly requested."""
        if isinstance(self.root, dict) and "metadata" not in self.root:
            return {}
        return _ParameterDocument.validate_metadata(self.section("metadata"))

    def value(self, path: str) -> YamlValue:
        """Read a concrete field; reject malformed used values with its full path.

        Unrelated component/general fields stay opaque. Missing fields raise
        AttributeError. Explicit whole-model reads validate that requested model.
        """
        node, schema = self._field(path)
        try:
            validated = SchemaValidator(schema).validate_python(deepcopy(node))
        except ValueError as exc:
            raise ValueError(f"{self._source}: {path}: {exc}") from exc
        serialized = SchemaSerializer(schema).to_python(validated, mode="python")
        return TypeAdapter(YamlValue).validate_python(serialized)

    def node(self, path: str) -> YamlValue:
        """Copy one declared node for field navigation without decoding its children.

        Ancestors must be mappings. Requested component kinds must be registered.
        Scalar values still require value() before consumption.
        """
        return deepcopy(self._field(path)[0])

    def _root_field(self, name: str) -> tuple[YamlValue, type[BaseModel]]:
        from .registry import component_registry

        if name == "general":
            return self.section("general"), PointGeneral
        components = self.section("components")
        if not isinstance(components, dict):
            raise ValueError("parameter components must be a mapping")
        if name not in components:
            raise AttributeError(name)
        node = components[name]
        if not isinstance(node, dict):
            raise ValueError(f"{name}: component must be a mapping")
        kind = node.get("kind")
        if not isinstance(kind, str):
            raise ValueError(f"{name}.kind must be a string")
        return node, component_registry.get(kind, source=self._source, component=name)

    def _default_value(self, path: str, schema: core_schema.CoreSchema) -> YamlValue:
        if path not in self._defaults:
            default = SchemaValidator(schema).validate_python(PydanticUndefined)
            self._defaults[path] = TypeAdapter(YamlValue).validate_python(
                SchemaSerializer(schema).to_python(
                    default, mode="python", exclude=_default_omissions(default)
                )
            )
        return self._defaults[path]

    def _field(self, path: str) -> tuple[YamlValue, core_schema.CoreSchema]:
        parts = path.split(".")
        if not all(parts):
            raise ValueError(f"Invalid parameter path: {path!r}")
        node, root_model = self._root_field(parts[0])
        model: type[BaseModel] | None = root_model
        annotation: object = YamlValue
        schema = TypeAdapter(YamlValue).core_schema
        defaulted = False
        for index, part in enumerate(parts[1:], start=1):
            defaulted = False
            if model is not None:
                self._require_model_mapping(node, annotation, path, part)
            if isinstance(node, list):
                arguments = get_args(annotation)
                node = TypeAdapter(YamlValue).validate_python(
                    projection_child(node, part)
                )
                annotation = arguments[0] if arguments else YamlValue
                schema = TypeAdapter(annotation).core_schema
                model = self._nested_model(annotation)
                continue
            if not isinstance(node, dict):
                raise ProjectionPathError(part)
            field = None if model is None else model.model_fields.get(part)
            if (
                field is None
                and model is not None
                and model.model_config.get("extra") != "allow"
            ):
                raise AttributeError(path)
            if field is not None and model is not None:
                annotation = field.rebuild_annotation()
                schema = model_field_schema(model, part)
            else:
                arguments = get_args(annotation)
                annotation = (
                    arguments[1]
                    if get_origin(annotation) is dict and len(arguments) == 2
                    else YamlValue
                )
                schema = TypeAdapter(annotation).core_schema
            if part in node:
                node = node[part]
            elif field is not None and not field.is_required():
                prefix = ".".join(parts[: index + 1])
                node = self._default_value(prefix, schema)
                defaulted = True
            else:
                raise AttributeError(f"{path}: field is not set")
            model = self._nested_model(annotation)
        # A default has already passed its own schema. Revalidating its serialized
        # value would turn an omitted optional field into an explicit null input.
        return node, TypeAdapter(YamlValue).core_schema if defaulted else schema

    @staticmethod
    def _require_model_mapping(
        node: YamlValue, annotation: object, path: str, segment: str
    ) -> None:
        if isinstance(node, dict):
            return
        if node is None and type(None) in get_args(annotation):
            raise ProjectionPathError(segment)
        raise ValueError(f"{path}: model ancestor must be a mapping")

    @staticmethod
    def _nested_model(annotation: object) -> type[BaseModel] | None:
        nested = (
            ()
            if get_origin(annotation) is dict
            else (annotation, *get_args(annotation))
        )
        return next(
            (
                item
                for item in nested
                if isinstance(item, type) and issubclass(item, BaseModel)
            ),
            None,
        )


class SetupDocument(_ParameterDocument):
    """Entry identity and complete component template in working units.

    schema_version is the caller policy's current whole-component version,
    distinct from the container format_version. Missing or noncurrent versions
    fail at model_validate with a path and upgrade command, before kind lookup.
    """

    general: SetupGeneral


class PointDocument(_ParameterDocument):
    """One complete, independent point document in working units.

    format and format_version are the parameter-container header, written as 2.0.
    schema_version is the strict positive integer from the caller's registered
    ParameterSchemaPolicy, independent of that container version. Runtime reads
    require the current version and never classify or upgrade legacy documents.
    The store validates header compatibility. Values remain in working units;
    general is point-local.
    components maps public identifiers to original registered ComponentSchema models,
    including kind; each original model owns validation. Schema loading does not
    reserve view members; binding a view checks those namespace collisions. No value or source falls back to setup.
    metadata maps logical dotted paths to opaque caller YAML mappings.
    Component and nested models choose extra policy at every minor version:
    forbid rejects unknown fields, ignore discards them, and allow retains them.
    Forward-minor preservation applies only to unknown container fields.
    """

    _source: ClassVar[Path] = Path("point.yaml")
    general: PointGeneral
