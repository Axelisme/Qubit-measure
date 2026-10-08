"""Partial input schema projection, retaining Pydantic's field constraints."""

from collections.abc import Mapping
from enum import Enum
from typing import get_args, get_origin

from pydantic import BaseModel, JsonValue, PydanticInvalidForJsonSchema
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import PydanticOmit, core_schema

from ._json import copy_object, default_json
from ._sources import local_choices
from ._structure import alternatives, discriminator, model_types, schema_branch_model
from .model import Display, FormEnv, FormModel, FormPath


class _InputJsonSchema(GenerateJsonSchema):
    """Annotate native tagged schemas, including containers, aliases and definitions.

    Pydantic owns traversal and refs. Core choice keys supply intrinsic typed
    tags; native choice schemas identify branches without string-key collisions.
    """

    def tagged_union_schema(
        self, schema: core_schema.TaggedUnionSchema
    ) -> dict[str, JsonValue]:
        """Extend native oneOf with typed tags, preserving native skipped choices.

        Other generation errors propagate. Non-finite/non-JSON tags raise
        ValueError through the shared JSON codec.
        """
        generated = copy_object(super().tagged_union_schema(schema))
        branches: list[dict[str, JsonValue]] = []
        identities: list[list[JsonValue]] = []
        # Group aliases before adding metadata so it does not change schema equality.
        for tag, choice in schema["choices"].items():
            try:
                choice_schema = self.generate_inner(choice)
            except (PydanticOmit, PydanticInvalidForJsonSchema):
                # The native generator already skipped this choice and owns warnings.
                continue
            branch = copy_object(choice_schema)
            if branch not in branches:
                branches.append(branch)
                identities.append([])
            index = branches.index(branch)
            native_tag = tag.value if isinstance(tag, Enum) else tag
            identities[index].append(default_json(native_tag))
        for branch, tags in zip(branches, identities, strict=True):
            branch["x-form-discriminator-values"] = tags
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
    if node.get("type") in ("integer", "number"):
        return {"anyOf": [node, copy_object(_EXPR)]}
    return node


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
            for key, value in (
                ("title", metadata.label),
                ("description", metadata.description),
                ("unit", metadata.unit),
                ("placeholder", metadata.placeholder),
            ):
                if value is not None:
                    target[key] = value
            target["collapsed"] = metadata.collapsed
    for child_info, child_target in _container_nodes(info, target):
        _field_metadata(child_info, child_target)


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


def _child_nodes(
    info: FieldInfo, target: dict[str, JsonValue], root: dict[str, JsonValue]
) -> tuple[tuple[type[BaseModel], dict[str, JsonValue]], ...]:
    _inline_ref(target, root)
    node = _non_null(target)
    _inline_ref(node, root)
    models = model_types(info.annotation)
    if len(models) == 1:
        return ((models[0], node),)
    variants = node.get("oneOf")
    if isinstance(variants, list):
        if discriminator(info) is not None:
            children: list[tuple[type[BaseModel], dict[str, JsonValue]]] = []
            for variant in variants:
                if not isinstance(variant, dict):
                    raise ValueError("Tagged schema branch must be an object")
                tags = variant.get("x-form-discriminator-values")
                if not isinstance(tags, list):
                    raise ValueError("Tagged schema branch lacks intrinsic identities")
                children.append((schema_branch_model(info, tags), variant))
            return tuple(children)
        return tuple(
            (model, variant)
            for model, variant in zip(models, variants, strict=False)
            if isinstance(variant, dict)
        )
    return tuple(
        pair
        for child_info, child_target in _container_nodes(info, target)
        for pair in _child_nodes(child_info, child_target, root)
    )


def _apply_choices(
    target: dict[str, JsonValue], options: tuple[JsonValue, ...]
) -> None:
    variants = target.get("anyOf")
    if isinstance(variants, list):
        for variant in variants:
            if isinstance(variant, dict) and variant != _EXPR:
                _apply_choices(variant, options)
    else:
        _enum(target, options)


def _enum(target: dict[str, JsonValue], options: tuple[JsonValue, ...]) -> None:
    if options:
        target["enum"] = list(options)
    else:
        target["not"] = {}


def _decorate(
    model: type[BaseModel],
    node: dict[str, JsonValue],
    root: dict[str, JsonValue],
    env: FormEnv,
    seen: set[type[BaseModel]],
    inherited: Mapping[FormPath, tuple[JsonValue, ...]],
) -> None:
    if model in seen:
        return
    # Track only this ancestry; reused sibling occurrences have distinct choices.
    seen = seen | {model}
    _inline_ref(node, root)
    options = {**inherited, **local_choices(model, env)}
    props = node.get("properties")
    if not isinstance(props, dict):
        return
    for name, info in model.model_fields.items():
        target = props.get(name)
        if not isinstance(target, dict):
            continue
        _inline_ref(target, root)
        _field_metadata(info, target)
        child_options = {
            path[1:]: values
            for path, values in options.items()
            if len(path) > 1 and path[0] == name
        }
        for child, variant in _child_nodes(info, target, root):
            _decorate(child, variant, root, env, seen, child_options)
        values = options.get((name,))
        if values is not None:
            _apply_choices(target, values)


def input_schema(model: type[FormModel], env: FormEnv) -> dict[str, JsonValue]:
    """Project a Pydantic schema to partial tagged JSON input for this environment."""
    raw = copy_object(model.model_json_schema(schema_generator=_InputJsonSchema))
    root = copy_object(_partial(raw))
    _decorate(model, root, root, env, set(), {})
    return root
