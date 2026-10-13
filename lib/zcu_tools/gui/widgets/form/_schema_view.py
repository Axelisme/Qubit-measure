"""Project the public partial-input schema into typed, active view rows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from pydantic import JsonValue

from zcu_tools.gui.form import FormPreview, FormValue, FormWritePath

type _Kind = Literal[
    "text", "integer", "number", "complex", "boolean", "json", "model", "key"
]


@dataclass(frozen=True)
class KeyOrInline:
    """How a row the schema marks x-form-key-or-inline reads and writes.

    Such a field holds a library key or an inline model. selector is None for
    an untagged str-or-model field: its JSON value is the key string or the
    inline object, and key_tag, inline_tag and name are unused (None, None
    and ""). For a tagged field (a ModuleInput), selector is the tag property,
    key_tag and inline_tag the tags of the key branch and the inline branch,
    and name the key branch's key property. keys are the schema's
    x-form-library-keys strings in schema order, the captured library keys
    offered for this field; empty when none are offered.
    """

    selector: str | None
    key_tag: JsonValue
    inline_tag: JsonValue
    name: str
    keys: tuple[str, ...]

    def inline(self, raw: JsonValue) -> bool:
        """Whether raw, the field's JSON input, is an inline model."""
        if self.selector is None:
            return isinstance(raw, dict)
        return isinstance(raw, dict) and raw.get(self.selector) == self.inline_tag

    def key(self, raw: JsonValue) -> JsonValue:
        """Return the key raw holds, or None when raw holds no key.

        raw holds no key when it is inline, unfilled (None) or another tag,
        or when a tagged key branch lacks its name. A held key may be any
        JSON the input contains, not only a listed or even a string key.
        """
        if self.selector is None:
            return None if isinstance(raw, dict) else raw
        if isinstance(raw, dict) and raw.get(self.selector) == self.key_tag:
            return raw.get(self.name)
        return None

    def key_write(
        self, path: FormWritePath, key: str
    ) -> dict[FormWritePath, JsonValue]:
        """Return the write batch that selects key for the field at path.

        An untagged field takes the key string itself. A tagged field takes
        its key tag and the key in one batch, so FormState switches the
        branch first and the explicit key wins.
        """
        if self.selector is None:
            return {path: key}
        return {(*path, self.selector): self.key_tag, (*path, self.name): key}

    def inline_write(self, path: FormWritePath) -> dict[FormWritePath, JsonValue]:
        """Return the write batch that switches the field at path to inline.

        An untagged field takes {}, the core's inline model defaults. A
        tagged field takes its inline tag; the inline branch's defaults (for
        a ModuleInput, a copy of the referenced library module) fill it.
        """
        if self.selector is None:
            return {path: {}}
        return {(*path, self.selector): self.inline_tag}


@dataclass(frozen=True)
class FieldView:
    """One active schema row; no validation or writable model mirror.

    path is the canonical model-field path. kind selects a Qt representation.
    label, description, unit and placeholder are display text; collapsed is
    the initial model-group fold hint. nullable permits null; active identifies
    an enabled nullable model. expression permits scalar Expr input: the
    schema offers an Expr alternative, which the core does only for text
    fields, so a row with options never has it. options is current choice
    data, or None for free input (empty means no choices).
    decimals is the schema's
    display digit count for a number, or None to show the full value. fixed
    marks a one-value Literal (schema const, not nullable): its row shows the
    value as read-only text, so options is None and expression is False.
    raw is current JSON or a schema static default when present is False.
    value is the ordinary preview leaf when has_value is True; otherwise None
    is only a placeholder. key_or_inline describes a field the schema marks
    x-form-key-or-inline (a str-or-model field or a ModuleInput), or is None
    for other fields: its row offers Inline and library keys in one
    dropdown, kind is model while the input is inline and key otherwise
    (a key or unfilled), and expression is False. group is the schema's
    x-form-group name: the editor draws
    the row in that foldable group of its model level, or ungrouped for
    None. children are current model rows,
    never container keys/indexes. Only kind=model can have children.
    """

    path: FormWritePath
    kind: _Kind
    label: str
    description: str
    unit: str
    placeholder: str
    collapsed: bool
    nullable: bool
    active: bool
    expression: bool
    options: tuple[JsonValue, ...] | None
    raw: JsonValue
    present: bool
    value: FormValue
    has_value: bool
    key_or_inline: KeyOrInline | None
    decimals: int | None
    fixed: bool
    group: str | None
    children: tuple[FieldView, ...] = ()


def _objects(value: JsonValue) -> tuple[dict[str, JsonValue], ...]:
    return (
        tuple(item for item in value if isinstance(item, dict))
        if isinstance(value, list)
        else ()
    )


def _properties(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    properties = schema.get("properties")
    return properties if isinstance(properties, dict) else {}


def _text(schema: dict[str, JsonValue], key: str) -> str:
    value = schema.get(key)
    return value if isinstance(value, str) else ""


def _variants(schema: dict[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    # Nullable scalar fields wrap another scalar/Expr anyOf. Flatten only
    # alternatives, not model properties/container items; retain outer metadata.
    branches = _objects(schema.get("anyOf"))
    if not branches:
        return (schema,)
    metadata = {key: value for key, value in schema.items() if key != "anyOf"}
    return tuple(
        variant for branch in branches for variant in _variants({**metadata, **branch})
    )


def _direct(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    # Value constraints are inside the ordinary Expr alternative.
    return next(
        (
            node
            for node in _variants(schema)
            if "expr" not in _properties(node) and node.get("type") != "null"
        ),
        schema,
    )


def _options(schema: dict[str, JsonValue]) -> tuple[JsonValue, ...] | None:
    if schema.get("not") == {}:
        return ()
    values = schema.get("enum")
    if isinstance(values, list):
        return tuple(values)
    return (schema["const"],) if "const" in schema else None


def _key_and_inline(
    schema: dict[str, JsonValue], variants: tuple[dict[str, JsonValue], ...]
) -> tuple[KeyOrInline, dict[str, JsonValue]] | None:
    """Read a field the schema marks x-form-key-or-inline: its rule and inline node.

    Core decides which fields are key-or-inline; unmarked fields return None.
    A tagged field (with a discriminator) is read by _tagged_key_or_inline.
    Otherwise variants are the field's flattened alternatives: the string
    node is the library key, whose x-form-library-keys are the offered keys,
    and the other non-null node the inline model. A marked field without
    exactly one of each raises ValueError.
    """
    if schema.get("x-form-key-or-inline") is not True:
        return None
    if "discriminator" in schema:
        return _tagged_key_or_inline(schema)
    keys = [node for node in variants if node.get("type") == "string"]
    inline = [node for node in variants if node.get("type") not in ("string", "null")]
    if len(keys) != 1 or len(inline) != 1:
        raise ValueError(
            "x-form-key-or-inline schema requires one key and one inline alternative"
        )
    rule = KeyOrInline(None, None, None, "", _library_keys(keys[0]))
    return rule, inline[0]


def _tagged_key_or_inline(
    schema: dict[str, JsonValue],
) -> tuple[KeyOrInline, dict[str, JsonValue]]:
    """Read a tagged key-or-inline field (a ModuleInput): its rule and inline branch.

    Each oneOf branch has one tag and one property besides the selector: a
    string in the key branch (whose x-form-library-keys are the offered
    keys) and a model in the inline branch. Any other shape raises
    ValueError.
    """
    discriminator = schema.get("discriminator")
    selector = (
        _text(discriminator, "propertyName") if isinstance(discriminator, dict) else ""
    )
    keys: list[tuple[JsonValue, str, dict[str, JsonValue]]] = []
    inline: list[tuple[JsonValue, dict[str, JsonValue]]] = []
    for branch in _objects(schema.get("oneOf")):
        tags = branch.get("x-form-discriminator-values")
        others = [
            (name, node)
            for name, node in _properties(branch).items()
            if name != selector and isinstance(node, dict)
        ]
        if not isinstance(tags, list) or len(tags) != 1 or len(others) != 1:
            raise ValueError(
                "Tagged x-form-key-or-inline branch requires one tag and one property"
            )
        ((name, node),) = others
        if _direct(node).get("type") == "string":
            keys.append((tags[0], name, node))
        elif _kind(_variants(node)) == "model":
            inline.append((tags[0], branch))
    if not selector or len(keys) != 1 or len(inline) != 1:
        raise ValueError(
            "Tagged x-form-key-or-inline schema requires one key and one inline branch"
        )
    ((key_tag, name, key_node),) = keys
    ((inline_tag, inline_branch),) = inline
    rule = KeyOrInline(selector, key_tag, inline_tag, name, _library_keys(key_node))
    return rule, inline_branch


def _shown(node: dict[str, JsonValue]) -> bool:
    """Whether a property schema is drawn, i.e. not marked hidden.

    The schema marks Display(hidden=True) fields and one-value structural
    tags outside a union; their input stays in state and is still validated.
    """
    return node.get("hidden") is not True


def _decimals(metadata: dict[str, JsonValue]) -> int | None:
    value = metadata.get("decimals")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _selector_title(schema: dict[str, JsonValue], key: str) -> str:
    """Title of the tag field key in the first branch declaring one, else key.

    A placeholder selector (no branch selected yet) is then labelled like
    the selector of a selected branch.
    """
    for branch in _objects(schema.get("oneOf")):
        node = _properties(branch).get(key)
        if isinstance(node, dict) and (title := _text(node, "title")):
            return title
    return key


def _library_keys(key: dict[str, JsonValue]) -> tuple[str, ...]:
    """The string keys of key's x-form-library-keys, or () without that list."""
    keys = key.get("x-form-library-keys")
    if not isinstance(keys, list):
        return ()
    return tuple(item for item in keys if isinstance(item, str))


def _kind(variants: tuple[dict[str, JsonValue], ...]) -> _Kind:
    if any("__complex__" in _properties(node) for node in variants):
        return "complex"
    for node in variants:
        if "expr" in _properties(node) or node.get("type") == "null":
            continue
        if "properties" in node or "discriminator" in node:
            return "model"
        kind: _Kind | None = None
        match node.get("type"):
            case "integer":
                kind = "integer"
            case "number":
                kind = "number"
            case "boolean":
                kind = "boolean"
            case "array" | "object":
                kind = "json"
            case _:
                pass
        if kind is not None:
            return kind
    return "text"


def _branch(
    schema: dict[str, JsonValue], raw: dict[str, JsonValue]
) -> tuple[dict[str, JsonValue], str, tuple[JsonValue, ...]]:
    discriminator = schema.get("discriminator")
    key = (
        _text(discriminator, "propertyName") if isinstance(discriminator, dict) else ""
    )
    options: list[JsonValue] = []
    selected: dict[str, JsonValue] = {}
    for branch in _objects(schema.get("oneOf")):
        tags = branch.get("x-form-discriminator-values")
        if not isinstance(tags, list):
            raise ValueError(
                "Tagged branch schema requires x-form-discriminator-values"
            )
        if raw.get(key) in tags:
            selected = branch
        selector = _properties(branch).get(key)
        if isinstance(selector, dict):
            allowed = _options(_direct(selector))
            options.extend(tag for tag in tags if allowed is None or tag in allowed)
    return selected, key, tuple(options)


def _field(
    schema: dict[str, JsonValue],
    path: FormWritePath,
    data: dict[str, JsonValue],
    preview: FormPreview,
) -> FieldView:
    variants = _variants(schema)
    raw = data.get(path[-1])
    present = path[-1] in data
    split = _key_and_inline(schema, variants)
    rule: KeyOrInline | None = None
    if split is None:
        direct = _direct(schema)
        kind: _Kind = _kind(variants)
        options = _options(direct)
        default = direct.get("default")
        expression = any("expr" in _properties(node) for node in variants)
    else:
        rule, inline = split
        default = schema.get("default")
        inline_mode = rule.inline(raw if present else default)
        # A key row has no input node of its own; the field node describes it.
        direct = inline if inline_mode else {}
        kind = "model" if inline_mode else "key"
        options = None
        expression = False
    metadata = {**direct, **schema}
    nullable = any(node.get("type") == "null" for node in variants)
    # One legal value leaves nothing to choose or compute.
    fixed = split is None and "const" in direct and not nullable
    field = FieldView(
        path=path,
        kind=kind,
        label=_text(metadata, "title") or path[-1],
        description=_text(metadata, "description"),
        unit=_text(metadata, "unit"),
        placeholder=_text(metadata, "placeholder"),
        collapsed=metadata.get("collapsed") is True,
        nullable=nullable,
        active=not nullable or (present and raw is not None),
        expression=expression,
        options=None if fixed else options,
        raw=raw if present else default,
        present=present,
        value=preview.values.get(path),
        has_value=path in preview.values,
        key_or_inline=rule,
        decimals=_decimals(metadata),
        fixed=fixed,
        group=_text(schema, "x-form-group") or None,
    )
    if kind != "model" or not field.active:
        return field
    # Recursion through _children: each nested model level projects its rows.
    return replace(field, children=_model_children(direct, path, raw, preview, rule))


def _model_children(
    schema: dict[str, JsonValue],
    path: FormWritePath,
    raw: JsonValue,
    preview: FormPreview,
    rule: KeyOrInline | None,
) -> tuple[FieldView, ...]:
    """Project a model row's child rows, as _children does.

    rule is the row's key-or-inline rule, or None. A tagged key-or-inline
    field's tag has no row: its dropdown selects the branch.
    """
    children = _children(schema, path, raw, preview)
    if rule is None or rule.selector is None:
        return children
    return tuple(child for child in children if child.path[-1] != rule.selector)


def _children(
    schema: dict[str, JsonValue],
    path: FormWritePath,
    raw: JsonValue,
    preview: FormPreview,
) -> tuple[FieldView, ...]:
    """Project the drawn property rows of the active model value raw at path.

    schema is the model's node. A tagged model shows its selected branch, its
    selector a plain tag dropdown of the branch choices; an unknown or missing
    tag shows only a placeholder selector, titled like the branch selectors.
    Properties the schema marks hidden have no rows.
    """
    nested = raw if isinstance(raw, dict) else {}
    selector = ""
    choices: tuple[JsonValue, ...] = ()
    if "discriminator" in schema:
        branch, selector, choices = _branch(schema, nested)
        properties = _properties(branch)
        # An unknown/missing tag exposes only its selector; do not guess a branch.
        if not properties:
            selector_schema: dict[str, JsonValue] = {
                "type": "string",
                "title": _selector_title(schema, selector),
            }
            properties = {selector: selector_schema}
    else:
        properties = _properties(schema)
    return tuple(
        replace(child, options=choices, expression=False, fixed=False)
        if child.path[-1] == selector
        else child
        for name, node in properties.items()
        if isinstance(node, dict) and _shown(node)
        for child in (_field(node, (*path, name), nested, preview),)
    )


def project_fields(
    schema: dict[str, JsonValue], data: dict[str, JsonValue], preview: FormPreview
) -> tuple[FieldView, ...]:
    """Project a public input schema and detached data/preview into active rows.

    Preserve declaration order and metadata. Inspect the discriminator only
    to choose its visible schema branch; unknown tags show a selector without
    guessing children. Fields the schema marks hidden have no rows; their
    input and errors stay with the state. Do not evaluate, validate, write
    or generate defaults.
    """
    return tuple(
        _field(node, (name,), data, preview)
        for name, node in _properties(schema).items()
        if isinstance(node, dict) and _shown(node)
    )


def flatten_fields(fields: tuple[FieldView, ...]) -> tuple[FieldView, ...]:
    """Return fields and all active children in declaration-order preorder."""
    return tuple(
        item for field in fields for item in (field, *flatten_fields(field.children))
    )
