"""Partial input schema projection, retaining Pydantic's field constraints."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import get_args, get_origin

from pydantic import BaseModel, JsonValue, PydanticInvalidForJsonSchema
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import PydanticOmit, core_schema

from ._json import copy_object, default_json
from ._sources import local_choices
from ._structure import (
    accepts_expression,
    alternatives,
    branch_tags,
    discriminator,
    fixed_tag,
    key_or_inline,
    library_key,
    model_types,
    schema_branch_model,
)
from .model import (
    Display,
    FormEnv,
    FormModel,
    FormPath,
    ModuleInline,
    ModuleInput,
    ModuleRef,
    RangeInput,
)


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
    """Inline emitted tagged branches as (intrinsic tags, model branch) pairs.

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
        tags = variant.get("x-form-discriminator-values")
        if not isinstance(tags, list):
            raise ValueError("Tagged schema branch lacks intrinsic identities")
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
            branches.append((tags, variant))
    return branches


def _inline_variant(target: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Return the model alternative of a str-or-model field schema.

    Pydantic emits that alternative as a $ref, an object or a tagged oneOf; the
    key alternative is a string. A schema without one is returned unchanged.
    """
    variants = target.get("anyOf")
    return next(
        (
            variant
            for variant in (variants if isinstance(variants, list) else ())
            if isinstance(variant, dict)
            and ("$ref" in variant or "properties" in variant or "oneOf" in variant)
        ),
        target,
    )


def _child_nodes(
    info: FieldInfo, target: dict[str, JsonValue], root: dict[str, JsonValue]
) -> tuple[tuple[type[BaseModel], dict[str, JsonValue]], ...]:
    _inline_ref(target, root)
    node = (
        _inline_variant(target) if key_or_inline(info.annotation) else _non_null(target)
    )
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
    environment. library_specs holds each captured library key with its Spec, read
    once per call, or is None when env has no captured library. branch_tags maps
    each tagged-union branch model of this form, ModuleInput and RangeInput to
    its tag key (see branch_tags).
    """

    root: dict[str, JsonValue]
    env: FormEnv
    library_specs: tuple[tuple[str, BaseModel], ...] | None
    branch_tags: Mapping[type[BaseModel], str]


def _library_specs(env: FormEnv) -> tuple[tuple[str, BaseModel], ...] | None:
    library = env.library
    if library is None:
        return None
    return (
        *((key, library.module_spec(key)) for key in library.module_keys()),
        *((key, library.waveform_spec(key)) for key in library.waveform_keys()),
    )


def _key_branch(
    info: FieldInfo, target: dict[str, JsonValue], projection: _Projection
) -> None:
    """List a str-or-model field's offered library keys on its string branch.

    The branch is already a plain key without Expr (see accepts_expression).
    With a captured library, the branch's x-form-library-keys lists the keys
    whose Spec is one of the field's models, as suggestions; any string
    remains acceptable input.
    """
    variants = target.get("anyOf")
    models = model_types(info.annotation)
    for variant in variants if isinstance(variants, list) else ():
        if isinstance(variant, dict):
            specs = projection.library_specs
            if variant.get("type") == "string" and specs is not None:
                variant["x-form-library-keys"] = [
                    key for key, spec in specs if isinstance(spec, models)
                ]


def _module_input(info: FieldInfo) -> bool:
    """Whether info is a ModuleInput field: exactly ModuleRef | ModuleInline.

    Like a str-or-model field it is a library key or an inline model, but its
    kind tag selects the two: ref with a name key, or inline with a module.
    A nullable ModuleInput is not included and stays a plain tagged union.
    """
    return set(alternatives(info.annotation)) == {ModuleRef, ModuleInline}


def _ref_keys(
    target: dict[str, JsonValue],
    projection: _Projection,
    options: tuple[JsonValue, ...] | None,
) -> None:
    """List a ModuleRef name's offered library keys in x-form-library-keys.

    target is the name property node. With a captured library the list is
    every library module key, because an inline ModuleSpec accepts every
    module type; options, the name's declared choices or None, keeps only
    the keys among them, so each listed key passes both the choices and the
    library check. Without a captured library the node gets no list.
    """
    library = projection.env.library
    if library is None:
        return
    target["x-form-library-keys"] = [
        key for key in library.module_keys() if options is None or key in options
    ]


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


def _decorate(
    model: type[BaseModel],
    node: dict[str, JsonValue],
    projection: _Projection,
    seen: set[type[BaseModel]],
    inherited: Mapping[FormPath, tuple[JsonValue, ...]],
    prefix: FormPath,
) -> None:
    """Decorate model's schema node in place; prefix is model's field path."""
    if model in seen:
        return
    # Track only this ancestry; reused sibling occurrences have distinct choices.
    seen = seen | {model}
    root = projection.root
    _inline_ref(node, root)
    options = {**inherited, **local_choices(model, projection.env)}
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
        if key_or_inline(info.annotation):
            # Core owns the str-or-model rule; views read this marker only.
            target["x-form-key-or-inline"] = True
            _key_branch(info, target, projection)
        elif _module_input(info):
            target["x-form-key-or-inline"] = True
        if library_key(model, name):
            # A library key is a plain string, like a str-or-model key.
            _without_expr(target)
            _ref_keys(target, projection, options.get((name,)))
        child_options = {
            leaf[1:]: values
            for leaf, values in options.items()
            if len(leaf) > 1 and leaf[0] == name
        }
        # Without a union selector, a branch model's tag has only one value.
        fixed = bool(models) and discriminator(info) is None
        for child, variant in _child_nodes(info, target, root):
            _decorate(child, variant, projection, seen, child_options, path)
            if fixed:
                _hide_fixed_tag(child, variant, projection.branch_tags)
        values = options.get((name,))
        if values is not None:
            _apply_choices(target, values)


def input_schema(model: type[FormModel], env: FormEnv) -> dict[str, JsonValue]:
    """Project a Pydantic schema to partial tagged JSON input for this environment."""
    raw = copy_object(model.model_json_schema(schema_generator=_InputJsonSchema))
    root = copy_object(_partial(raw))
    tags = branch_tags((model, ModuleInput, RangeInput))
    projection = _Projection(root, env, _library_specs(env), tags)
    _decorate(model, root, projection, set(), {}, ())
    _hide_fixed_tag(model, root, tags)
    return root
