"""Partial input schema projection, retaining Pydantic's field constraints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import get_args, get_origin

from pydantic import BaseModel, JsonValue, PydanticInvalidForJsonSchema
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import PydanticOmit, core_schema

from ._json import copy_object
from ._sources import local_choices
from ._structure import (
    accepts_expression,
    alternatives,
    branch_tags,
    check_form_shape,
    discriminator,
    fixed_tag,
    intrinsic_tags,
    library_key,
    model_types,
    schema_branch_model,
    selected,
)
from .model import (
    Display,
    FormEnv,
    FormModel,
    FormPath,
    LinRange,
    ModuleInput,
)
from .options import Option


class _InputJsonSchema(GenerateJsonSchema):
    """Preserve native tagged branch order, including typed tag collisions.

    Pydantic owns traversal and refs. Iterate native choices rather than the
    string-keyed discriminator mapping, where numeric and string tags collide.
    """

    def tagged_union_schema(
        self, schema: core_schema.TaggedUnionSchema
    ) -> dict[str, JsonValue]:
        """Emit ordered concrete choices, preserving native skipped branches.

        Other generation errors propagate. Intrinsic tags remain in the model
        declaration; no consumer-specific branch metadata is emitted.
        """
        generated = copy_object(super().tagged_union_schema(schema))
        branches: list[dict[str, JsonValue]] = []
        for choice in schema["choices"].values():
            try:
                choice_schema = self.generate_inner(choice)
            except (PydanticOmit, PydanticInvalidForJsonSchema):
                # The native generator already skipped this choice and owns warnings.
                continue
            branch = copy_object(choice_schema)
            if branch not in branches:
                branches.append(branch)
        generated["oneOf"] = [branch for branch in branches]
        return generated


_EXPR: dict[str, JsonValue] = {
    "type": "object",
    "properties": {"expr": {"type": "string"}},
    "required": ["expr"],
    "additionalProperties": False,
}
_COMPLEX: dict[str, JsonValue] = {
    "type": "object",
    "properties": {
        "__complex__": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
        }
    },
    "required": ["__complex__"],
    "additionalProperties": False,
}


def _partial(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Copy schema without required lists, with an Expr alternative per text node.

    A text node is an integer, number or string node without enum or const:
    Literal and Enum values are picked, not typed. Boolean nodes get none.
    This places the alternative by node shape only; _decorate then removes
    it from each field that accepts_expression, the core's one text-field
    rule, does not accept, and from choice leaves and library keys.
    """
    node = copy_object(schema)
    node.pop("required", None)
    # Only schema positions are traversed: property names and defaults are data.
    for key in ("properties", "$defs", "patternProperties", "dependentSchemas"):
        children = node.get(key)
        if isinstance(children, dict):
            node[key] = {
                name: _partial(child) if isinstance(child, dict) else child
                for name, child in children.items()
            }
    for key in (
        "items",
        "additionalProperties",
        "not",
        "contains",
        "propertyNames",
        "if",
        "then",
        "else",
    ):
        child = node.get(key)
        if isinstance(child, dict):
            node[key] = _partial(child)
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        branches = node.get(key)
        if isinstance(branches, list):
            node[key] = [
                _partial(branch) if isinstance(branch, dict) else branch
                for branch in branches
            ]
    if (
        node.get("type") in ("integer", "number", "string")
        and "enum" not in node
        and "const" not in node
    ):
        return {"anyOf": [node, copy_object(_EXPR)]}
    return node


def _without_expr(node: dict[str, JsonValue]) -> None:
    """Drop the Expr alternatives of node's anyOf in place, nested anyOf included.

    When one ordinary alternative remains, node collapses to it and keeps its
    outer metadata; a nested anyOf, such as the inner node of a nullable
    leaf, collapses the same way. Model properties and container items are
    not entered. A node without anyOf is unchanged.
    """
    branches = node.get("anyOf")
    if not isinstance(branches, list):
        return
    for branch in branches:
        if isinstance(branch, dict):
            _without_expr(branch)
    ordinary = [branch for branch in branches if branch != _EXPR]
    if len(ordinary) == 1 and isinstance(ordinary[0], dict):
        metadata = {key: value for key, value in node.items() if key != "anyOf"}
        node.clear()
        node.update(ordinary[0])
        node.update(metadata)
    else:
        node["anyOf"] = ordinary


def _resolve(
    node: dict[str, JsonValue], root: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        definitions = root.get("$defs")
        if isinstance(definitions, dict):
            target = definitions.get(ref.removeprefix("#/$defs/"))
            if isinstance(target, dict):
                return target
    return node


def _inline_ref(node: dict[str, JsonValue], root: dict[str, JsonValue]) -> None:
    """Copy a shared definition into this occurrence, retaining field metadata."""
    resolved = _resolve(node, root)
    if resolved is not node:
        metadata = {key: value for key, value in node.items() if key != "$ref"}
        node.clear()
        node.update(copy_object(resolved))
        node.update(metadata)


def _field_metadata(info: FieldInfo, target: dict[str, JsonValue]) -> None:
    if complex in alternatives(info.annotation):
        variants: list[JsonValue] = [
            {"type": "number"},
            copy_object(_COMPLEX),
            copy_object(_EXPR),
        ]
        if type(None) in alternatives(info.annotation):
            variants.append({"type": "null"})
        target.pop("type", None)
        target["anyOf"] = variants
    for metadata in info.metadata:
        if isinstance(metadata, Display):
            if metadata.decimals is not None and float not in alternatives(
                info.annotation
            ):
                raise ValueError(
                    f"Display decimals requires a float field, not {info.annotation!r}"
                )
            for key, value in (
                ("title", metadata.label),
                ("description", metadata.description),
                ("unit", metadata.unit),
                ("placeholder", metadata.placeholder),
                ("decimals", metadata.decimals),
            ):
                if value is not None:
                    target[key] = value
            target["collapsed"] = metadata.collapsed
            # Only declared: forms without hidden fields keep their schema.
            if metadata.hidden:
                target["hidden"] = True
            if metadata.group is not None:
                target["x-form-group"] = metadata.group
    for child_info, child_target in _container_nodes(info, target):
        _field_metadata(child_info, child_target)


def _label(info: FieldInfo) -> str | None:
    """Return the field's Display label, or None when it declares none."""
    return next(
        (
            metadata.label
            for metadata in info.metadata
            if isinstance(metadata, Display) and metadata.label is not None
        ),
        None,
    )


def _section_title(name: str) -> str:
    """Title a model field by name: underscores as spaces, first letter upper.

    For example ro_cfg becomes "Ro cfg" and module becomes "Module".
    """
    text = name.replace("_", " ")
    return text[:1].upper() + text[1:]


def _hide_fixed_tag(
    model: type[BaseModel],
    node: dict[str, JsonValue],
    tags: Mapping[type[BaseModel], str],
) -> None:
    """Mark model's one-value structural tag hidden in its schema node.

    The caller reached model without a union selector, so its tag can hold
    only one value and the editor does not draw it. Models without such a tag
    (see fixed_tag) are left unchanged.
    """
    key = fixed_tag(model, tags)
    properties = node.get("properties")
    if key is None or not isinstance(properties, dict):
        return
    selector = properties.get(key)
    if isinstance(selector, dict):
        selector["hidden"] = True


def _non_null(target: dict[str, JsonValue]) -> dict[str, JsonValue]:
    variants = target.get("anyOf")
    if isinstance(variants, list):
        return next(
            (v for v in variants if isinstance(v, dict) and v.get("type") != "null"),
            target,
        )
    return target


def _container_nodes(
    info: FieldInfo, target: dict[str, JsonValue]
) -> tuple[tuple[FieldInfo, dict[str, JsonValue]], ...]:
    types = tuple(t for t in alternatives(info.annotation) if t is not type(None))
    if len(types) != 1:
        return ()
    annotation = types[0]
    args = get_args(annotation)
    if not args:
        return ()
    target = _non_null(target)
    if get_origin(annotation) in (dict, Mapping):
        items = target.get("additionalProperties")
        return (
            ((FieldInfo.from_annotation(args[1]), items),)
            if isinstance(items, dict)
            else ()
        )
    if get_origin(annotation) in (list, tuple):
        items = target.get("items")
        if isinstance(items, dict):
            return ((FieldInfo.from_annotation(args[0]), items),)
        fixed = target.get("prefixItems")
        if isinstance(fixed, list):
            return tuple(
                (FieldInfo.from_annotation(t), node)
                for t, node in zip(args, fixed, strict=False)
                if isinstance(node, dict)
            )
    return ()


def _tagged_branches(
    variants: list[JsonValue], tag_key: str, root: dict[str, JsonValue]
) -> list[tuple[list[JsonValue], dict[str, JsonValue]]]:
    """Inline emitted tagged branches as (native tags, model branch) pairs.

    Pydantic emits a named union member (a ``type`` alias) that is tagged on
    the same key as one branch carrying all of its members' tags. Replace that
    branch with its own branches, recursively, so every branch is one concrete
    model with its own tags, as the field annotation's flattened union is.
    Raise ValueError for a non-object branch or one without emitted tags.
    """
    branches: list[tuple[list[JsonValue], dict[str, JsonValue]]] = []
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError("Tagged schema branch must be an object")
        _inline_ref(variant, root)
        nested = variant.get("oneOf")
        marker = variant.get("discriminator")
        if (
            isinstance(nested, list)
            and isinstance(marker, dict)
            and marker.get("propertyName") == tag_key
        ):
            branches.extend(_tagged_branches(nested, tag_key, root))
        else:
            props = variant.get("properties")
            selector = props.get(tag_key) if isinstance(props, dict) else None
            if not isinstance(selector, dict):
                raise ValueError("Tagged schema branch lacks a selector")
            branches.append((_native_tags(selector, root), variant))
    return branches


def _native_tags(
    node: dict[str, JsonValue], root: dict[str, JsonValue]
) -> list[JsonValue]:
    """Read an undecorated selector to associate its schema with the annotation.

    This runs before choices alter enums. Tags used by options come from that
    annotation owner, not from this schema association or the wire mapping.
    """
    node = _resolve(node, root)
    if "const" in node:
        return [node["const"]]
    values = node.get("enum")
    if isinstance(values, list):
        return values
    variants = node.get("anyOf", node.get("allOf"))
    if isinstance(variants, list):
        return [
            tag
            for variant in variants
            if isinstance(variant, dict)
            for tag in _native_tags(variant, root)
        ]
    raise ValueError("Tagged schema selector lacks native Literal values")


def _child_nodes(
    info: FieldInfo, target: dict[str, JsonValue], root: dict[str, JsonValue]
) -> tuple[tuple[type[BaseModel], dict[str, JsonValue]], ...]:
    _inline_ref(target, root)
    node = _non_null(target)
    _inline_ref(node, root)
    models = model_types(info.annotation)
    variants = node.get("oneOf")
    if isinstance(variants, list):
        if (tag_key := discriminator(info)) is not None:
            tagged = _tagged_branches(variants, tag_key, root)
            node["oneOf"] = [variant for _, variant in tagged]
            children: list[tuple[type[BaseModel], dict[str, JsonValue]]] = []
            for tags, variant in tagged:
                children.append((schema_branch_model(info, tags), variant))
            return tuple(children)
        return tuple(
            (model, variant)
            for model, variant in zip(models, variants, strict=False)
            if isinstance(variant, dict)
        )
    if len(models) == 1:
        return ((models[0], node),)
    return tuple(
        pair
        for child_info, child_target in _container_nodes(info, target)
        for pair in _child_nodes(child_info, child_target, root)
    )


def _apply_choices(
    target: dict[str, JsonValue], options: tuple[JsonValue, ...]
) -> None:
    """Restrict a choice leaf's node to options; it takes no Expr envelope."""
    _without_expr(target)
    variants = target.get("anyOf")
    if isinstance(variants, list):
        for variant in variants:
            if isinstance(variant, dict):
                _apply_choices(variant, options)
    else:
        _enum(target, options)


def _enum(target: dict[str, JsonValue], options: tuple[JsonValue, ...]) -> None:
    if options:
        target["enum"] = list(options)
    else:
        target["not"] = {}


@dataclass(frozen=True)
class _Projection:
    """Inputs shared by one input_schema call.

    root is the partial schema being decorated in place. env is the caller's
    environment. branch_tags maps each tagged-union model to its tag key.
    owners associates occurrence-local schema objects with their annotation
    models before dynamic choices change selectors. Copies used for option
    fields walk alongside these originals; identity never enters wire schema.
    """

    root: dict[str, JsonValue]
    env: FormEnv
    branch_tags: Mapping[type[BaseModel], str]
    selections: dict[FormPath, tuple[Option, ...]] = field(default_factory=dict)
    active_fields: Mapping[FormPath, FieldInfo] | None = None
    owners: dict[int, type[BaseModel]] = field(default_factory=dict)


def _expression_rule(
    info: FieldInfo, target: dict[str, JsonValue], path: FormPath
) -> None:
    """Keep target's Expr alternatives only where accepts_expression allows.

    target is info's schema node at field path; whole-container items are
    checked against their item annotations too. A union of text and picked
    values raises ValueError naming path.
    """
    if not accepts_expression(info.annotation, path):
        _without_expr(target)
    for item_info, item_target in _container_nodes(info, target):
        _expression_rule(item_info, item_target, path)


def _option_path(model: type[BaseModel], path: FormPath) -> bool:
    if not path or not isinstance(path[0], str):
        return False
    info = model.model_fields.get(path[0])
    if info is None:
        return False
    return len(path) == 1 or any(
        _option_path(child, path[1:]) for child in model_types(info.annotation)
    )


def _local_options(
    model: type[BaseModel], env: FormEnv
) -> Mapping[FormPath, tuple[Option, ...]]:
    declared = model.options(env) if issubclass(model, FormModel) else {}
    for path in declared:
        if not _option_path(model, path):
            raise ValueError(f"Unknown option field path: {path!r}")
    return declared


def _automatic_options(
    info: FieldInfo,
    target: dict[str, JsonValue],
    choices: tuple[JsonValue, ...] | None,
    field_nodes: Mapping[type[BaseModel], dict[str, JsonValue]],
) -> tuple[Option, ...]:
    if choices is not None:
        return tuple(Option(str(value), value) for value in choices)
    tag = discriminator(info)
    node = _non_null(target)
    variants = node.get("oneOf")
    if tag is not None and isinstance(variants, list):
        result: list[Option] = []
        for model in field_nodes:
            # One option per intrinsic tag; aliases share the model's fields.
            result.extend(
                Option(str(value), {tag: value}, fields=model)
                for value in intrinsic_tags(model, tag)
            )
        if type(None) in alternatives(info.annotation):
            result.insert(0, Option("None", None))
        return tuple(result)
    values = node.get("enum", [node["const"]] if "const" in node else None)
    if isinstance(values, list):
        if type(None) in alternatives(info.annotation) and None not in values:
            values = [None, *values]
        return tuple(Option(str(value), value) for value in values)
    return ()


def _selected_fields(
    node: dict[str, JsonValue],
    original: dict[str, JsonValue],
    value: dict[str, JsonValue],
    projection: _Projection,
) -> dict[str, JsonValue]:
    """Select copied fields using the original occurrence's annotation owner."""
    marker = node.get("discriminator")
    variants = node.get("oneOf")
    if not isinstance(marker, dict) or not isinstance(variants, list):
        return original
    tag = marker.get("propertyName")
    if not isinstance(tag, str) or tag not in value:
        return original
    originals = original.get("oneOf")
    if not isinstance(originals, list):
        raise ValueError("Option fields lack their original union schema")
    for variant, source in zip(variants, originals, strict=True):
        if not isinstance(variant, dict) or not isinstance(source, dict):
            continue
        model = projection.owners[id(source)]
        if any(
            type(item) is type(value[tag]) and item == value[tag]
            for item in intrinsic_tags(model, tag)
        ):
            metadata = {
                key: item
                for key, item in node.items()
                if key not in ("oneOf", "discriminator", "x-form-options")
            }
            node.clear()
            node.update(copy_object(variant))
            node.update(metadata)
            return source
    raise ValueError(f"Option fields have no branch for {tag!r}: {value[tag]!r}")


def _omit_selected_tags(
    node: dict[str, JsonValue],
    original: dict[str, JsonValue],
    value: JsonValue,
    projection: _Projection,
) -> None:
    if not isinstance(value, dict):
        return
    node = _non_null(node)
    original = _selected_fields(node, _non_null(original), value, projection)
    props = node.get("properties")
    originals = original.get("properties")
    if not isinstance(props, dict) or not isinstance(originals, dict):
        return
    for name, selected_value in value.items():
        target = props.get(name)
        source = originals.get(name)
        if isinstance(target, dict) and isinstance(source, dict):
            if "const" in target or "enum" in target:
                props.pop(name)
            else:
                _omit_selected_tags(target, source, selected_value, projection)


def _project_option(
    option: Option,
    projection: _Projection,
    field_nodes: Mapping[type[BaseModel], dict[str, JsonValue]],
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {"label": option.label, "value": option.value}
    if option.fields is not None:
        model = option.fields
        if model in field_nodes:
            original = field_nodes[model]
            owner = projection
        else:
            raw = copy_object(
                model.model_json_schema(schema_generator=_InputJsonSchema)
            )
            original = _partial(raw)
            owner = _Projection(original, projection.env, projection.branch_tags)
            _decorate(model, original, owner, set(), _Declarations(), ())
        root = copy_object(original)
        _omit_selected_tags(root, original, option.value, owner)
        result["fields"] = root
    if option.commands:
        result["commands"] = [
            {"name": command.name, "label": command.label}
            for command in option.commands
        ]
    return result


@dataclass(frozen=True)
class _Declarations:
    """Occurrence-local declarations and active ancestry passed to nested models."""

    choices: Mapping[FormPath, tuple[JsonValue, ...]] = field(default_factory=dict)
    options: Mapping[FormPath, tuple[Option, ...]] = field(default_factory=dict)
    active: bool = True

    def child(self, name: str) -> _Declarations:
        return _Declarations(
            {
                path[1:]: values
                for path, values in self.choices.items()
                if len(path) > 1 and path[0] == name
            },
            {
                path[1:]: values
                for path, values in self.options.items()
                if len(path) > 1 and path[0] == name
            },
            self.active,
        )


def _selection_schema(
    info: FieldInfo,
    target: dict[str, JsonValue],
    projection: _Projection,
    declarations: _Declarations,
    path: FormPath,
    field_nodes: Mapping[type[BaseModel], dict[str, JsonValue]],
) -> None:
    name = path[-1]
    values = declarations.choices.get((name,))
    if values is not None:
        _apply_choices(target, values)
    selections = declarations.options.get((name,))
    if selections is None:
        selections = _automatic_options(info, target, values, field_nodes)
    if selections or (name,) in declarations.options or values is not None:
        if declarations.active:
            projection.selections[path] = selections
        target["x-form-options"] = [
            _project_option(item, projection, field_nodes) for item in selections
        ]


def _decorate(
    model: type[BaseModel],
    node: dict[str, JsonValue],
    projection: _Projection,
    seen: set[type[BaseModel]],
    inherited: _Declarations,
    prefix: FormPath,
) -> None:
    """Decorate every schema branch, collecting selections only on active ancestry."""
    if model in seen:
        return
    # A reused child class cannot make an inactive ancestor active again.
    active = inherited.active and (
        projection.active_fields is None
        or all(
            projection.active_fields.get((*prefix, name)) is info
            for name, info in model.model_fields.items()
        )
    )
    # Track only this ancestry; reused sibling occurrences have distinct choices.
    seen = seen | {model}
    root = projection.root
    _inline_ref(node, root)
    projection.owners[id(node)] = model
    declarations = _Declarations(
        {**inherited.choices, **local_choices(model, projection.env)},
        {**inherited.options, **_local_options(model, projection.env)},
        active,
    )
    props = node.get("properties")
    if not isinstance(props, dict):
        return
    for name, info in model.model_fields.items():
        target = props.get(name)
        if not isinstance(target, dict):
            continue
        _inline_ref(target, root)
        path = (*prefix, name)
        _expression_rule(info, target, path)
        _field_metadata(info, target)
        models = model_types(info.annotation)
        if models and _label(info) is None:
            # The inlined definition carries its class name as title.
            target["title"] = _section_title(name)
        if library_key(model, name):
            # A library key is a plain string, never an Expr envelope.
            _without_expr(target)
        # Without a union selector, a branch model's tag has only one value.
        fixed = bool(models) and discriminator(info) is None
        field_nodes = dict(_child_nodes(info, target, root))
        for child, variant in field_nodes.items():
            _decorate(child, variant, projection, seen, declarations.child(name), path)
            if fixed:
                _hide_fixed_tag(child, variant, projection.branch_tags)
        _selection_schema(info, target, projection, declarations, path, field_nodes)


def input_schema(model: type[FormModel], env: FormEnv) -> dict[str, JsonValue]:
    """Project a Pydantic schema to partial tagged JSON input for this environment."""
    return _project(model, env).root


def effective_options(
    model: type[FormModel], env: FormEnv, data: Mapping[str, JsonValue]
) -> Mapping[FormPath, tuple[Option, ...]]:
    """Resolve schema selections only for current active fields, without caching."""
    return _project(model, env, data).selections


def _active_fields(
    model: type[BaseModel], data: Mapping[str, JsonValue], prefix: FormPath = ()
) -> dict[FormPath, FieldInfo]:
    result: dict[FormPath, FieldInfo] = {}
    for name, info in model.model_fields.items():
        path = (*prefix, name)
        result[path] = info
        value = data.get(name)
        child = selected(info, value)
        if child is not None and isinstance(value, dict):
            result.update(_active_fields(child, value, path))
    return result


def _project(
    model: type[FormModel], env: FormEnv, data: Mapping[str, JsonValue] | None = None
) -> _Projection:
    check_form_shape(model)
    raw = copy_object(model.model_json_schema(schema_generator=_InputJsonSchema))
    root = _partial(raw)
    tags = branch_tags((model, ModuleInput))
    # The stored range kind stays structural without a range/list selector.
    tags[LinRange] = "kind"
    projection = _Projection(
        root,
        env,
        tags,
        active_fields=_active_fields(model, data) if data is not None else None,
    )
    _decorate(model, root, projection, set(), _Declarations(), ())
    _hide_fixed_tag(model, root, tags)
    return projection
