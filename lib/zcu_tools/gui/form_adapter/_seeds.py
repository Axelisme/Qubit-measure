"""Template seed expansion uses public static defaults and captured source APIs."""

from collections.abc import Mapping
from typing import Literal, get_args, get_origin

from pydantic import BaseModel, JsonValue
from pydantic_core import PydanticUndefinedType

from zcu_tools.cfg_model import (
    ConfigBase,
    Expr,
    Hidden,
    Slot,
    Template,
    TemplateDef,
    TemplateSeed,
    expand_path,
)
from zcu_tools.gui.form import FormEnv, static_defaults
from zcu_tools.gui.session.parameters import ModuleSlotError
from zcu_tools.resources.entry.roles import RoleResolutionError

from ._types import members, type_tag


def slot_key(slot: Slot, env: FormEnv) -> str | None:
    path = expand_path(slot.path)
    if env.library is None:
        return None
    if not path.startswith("$") and path in env.library.module_keys():
        return path
    if env.parameters is None:
        return None
    try:
        key = env.parameters.library_key(path)
    except (ModuleSlotError, RoleResolutionError):
        return None
    return key if key in env.library.module_keys() else None


def scalar_seed(seed: TemplateSeed) -> JsonValue:
    if isinstance(seed, Hidden):
        if isinstance(seed.value, PydanticUndefinedType):
            raise ValueError("Hidden module seeds require an explicit literal value")
        return scalar_seed(seed.value)
    if isinstance(seed, Expr):
        return {"expr": seed.text}
    if isinstance(seed, complex):
        return {"__complex__": [seed.real, seed.imag]}
    if isinstance(seed, Slot):
        raise TypeError("Slot seeds require captured source resolution")
    if isinstance(seed, list):
        return [
            {name: scalar_seed(value) for name, value in item.items()} for item in seed
        ]
    return seed


def declaration_seed(seed: TemplateSeed) -> JsonValue:
    """Retain Hidden metadata only in the private model-declaration seed tree.

    Actual input uses scalar_seed instead, so this marker never becomes input.
    """
    value = scalar_seed(seed)
    return {"hidden": True, "value": value} if isinstance(seed, Hidden) else value


def set_path(data: dict[str, JsonValue], path: str, value: JsonValue) -> None:
    parts = path.split(".")
    if not all(parts):
        raise ValueError(f"Invalid seed path {path!r}")
    parent = data
    for part in parts[:-1]:
        if part not in parent:
            parent[part] = {}
        child = parent[part]
        if not isinstance(child, dict):
            raise ValueError(f"Seed path traverses a non-inline field: {path!r}")
        parent = child
    parent[parts[-1]] = value


def inline_wrapper(annotation: object) -> type[BaseModel] | None:
    """Find the generated tagged inline wrapper, not its persistent child kind."""
    for model in members(annotation):
        if isinstance(model, type) and issubclass(model, BaseModel):
            kind = model.model_fields.get("kind")
            if (
                kind is not None
                and get_args(kind.annotation) == ("inline",)
                and "module" in model.model_fields
            ):
                return model
    return None


def spec_annotation(annotation: object) -> object:
    """Return the persistent-content annotation behind a generated wrapper."""
    wrapper = inline_wrapper(annotation)
    return (
        wrapper.model_fields["module"].annotation if wrapper is not None else annotation
    )


def persistent_input(
    model: type[BaseModel], data: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    """Remove generated ref/inline wrappers from a validated input model dump.

    The result is persistent Spec JSON, with child refs restored to strings.
    No library, asset or native cfg is read or materialized.
    """
    result: dict[str, JsonValue] = {}
    for name, value in data.items():
        info = model.model_fields.get(name)
        if info is None:
            result[name] = value
            continue
        annotation = info.annotation
        if inline_wrapper(annotation) is not None and isinstance(value, dict):
            if value.get("kind") == "ref" and "name" in value:
                result[name] = value["name"]
                continue
            if value.get("kind") != "inline" or not isinstance(
                value.get("module"), dict
            ):
                result[name] = value
                continue
            value = value["module"]
            annotation = spec_annotation(annotation)
        if (
            isinstance(value, dict)
            and "expr" not in value
            and any(
                isinstance(member, type) and issubclass(member, BaseModel)
                for member in members(annotation)
            )
        ):
            try:
                child = selected_model(annotation, value)
            except ValueError:
                # Invalid branch selectors stay available to model validation.
                result[name] = value
            else:
                result[name] = persistent_input(child, value)
        else:
            result[name] = value
    return result


def selected_model(
    annotation: object, data: Mapping[str, JsonValue]
) -> type[BaseModel]:
    models = tuple(
        model
        for model in members(annotation)
        if isinstance(model, type) and issubclass(model, BaseModel)
    )
    if len(models) == 1:
        return models[0]
    matches: list[type[BaseModel]] = []
    for model in models:
        for tag in ("type", "style", "kind"):
            info = model.model_fields.get(tag)
            if (
                info is not None
                and get_origin(info.annotation) is Literal
                and data.get(tag) in get_args(info.annotation)
            ):
                matches.append(model)
                break
    if len(matches) != 1:
        raise ValueError(f"Expected one tagged inline Spec branch, got {dict(data)!r}")
    return matches[0]


def supported_input(
    model: type[BaseModel], data: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    """Project a validated persistent Spec dump onto its generated input model.

    data must dump a Spec of model's family. Fields absent from model are the
    generator's unsupported module fields; their stored values are discarded,
    so conversion restores Spec defaults. Inline children follow their tagged
    branch; string keys become kind=ref/name and inline children become
    kind=inline/module. Scalar leaves are copied. data is not modified.
    """
    result: dict[str, JsonValue] = {}
    for name, value in data.items():
        if name not in model.model_fields:
            continue
        annotation = model.model_fields[name].annotation
        wrapper = inline_wrapper(annotation)
        if wrapper is not None:
            if isinstance(value, str):
                result[name] = {"kind": "ref", "name": value}
                continue
            annotation = spec_annotation(annotation)
        if isinstance(value, dict) and any(
            isinstance(member, type) and issubclass(member, BaseModel)
            for member in members(annotation)
        ):
            child = supported_input(selected_model(annotation, value), value)
            result[name] = (
                {"kind": "inline", "module": child} if wrapper is not None else child
            )
        else:
            result[name] = value
    return result


def complete_input(
    model: type[BaseModel],
    authored: Mapping[str, JsonValue],
    *,
    path: tuple[str, ...] = (),
) -> dict[str, JsonValue]:
    """Overlay persistent-space seeds with defaults, before tagged projection.

    Report ambiguous inline field paths. Wrapper content remains in persistent
    space so authors keep Spec paths for seeds and Slots. Seeds for fields
    absent from the generated model, including unsupported module fields,
    raise ValueError with their dotted paths under path.
    """
    unknown = authored.keys() - model.model_fields.keys()
    if unknown:
        paths = sorted(".".join((*path, name)) for name in unknown)
        raise ValueError(f"Unknown or unsupported seeds for {model.__name__}: {paths}")
    data = persistent_input(model, static_defaults(model))
    data.update(authored)
    for name, value in tuple(data.items()):
        info = model.model_fields[name]
        annotation = spec_annotation(info.annotation)
        if get_origin(annotation) is list:
            element = get_args(annotation)[0]
            if (
                isinstance(element, type)
                and issubclass(element, BaseModel)
                and isinstance(value, list)
            ):
                data[name] = [
                    complete_input(element, item, path=(*path, name, str(index)))
                    if isinstance(item, dict)
                    else item
                    for index, item in enumerate(value)
                ]
        if (
            name in {"type", "style"}
            and get_origin(info.annotation) is Literal
            and value not in get_args(info.annotation)
        ):
            raise ValueError(f"Unknown {name} tag: {value!r}")
        if (
            isinstance(value, dict)
            and "expr" not in value
            and any(
                isinstance(member, type) and issubclass(member, BaseModel)
                for member in members(spec_annotation(info.annotation))
            )
        ):
            child_path = (*path, name)
            try:
                child = selected_model(spec_annotation(info.annotation), value)
            except ValueError as exc:
                raise ValueError(
                    f"{'.'.join(child_path)}: Cannot select inline Spec discriminator; "
                    "supply a tag when the Spec has no static default"
                ) from exc
            data[name] = complete_input(child, value, path=child_path)
    return data


def _captured_child(annotation: object, key: str, env: FormEnv) -> dict[str, JsonValue]:
    """Copy a child key into supported persistent JSON of its declared family.

    annotation is the generated child selector. Only its captured Spec is read;
    descendants remain keys and assets are not accessed. Missing capture/key,
    incompatible families and source failures propagate to the Form caller.
    """
    if env.library is None:
        raise ValueError("Missing captured library for Slot child copy")
    models = tuple(
        model
        for model in members(spec_annotation(annotation))
        if isinstance(model, type) and issubclass(model, BaseModel)
    )
    spec = (
        env.library.waveform_spec(key)
        if all("style" in model.model_fields for model in models)
        else env.library.module_spec(key)
    )
    model = next((model for model in models if issubclass(model, type(spec))), None)
    if model is None:
        raise ValueError(f"Incompatible library family for {key!r}")
    return persistent_input(
        model, supported_input(model, spec.model_dump(mode="json", exclude_none=True))
    )


def set_slot_path(
    model: type[BaseModel],
    data: dict[str, JsonValue],
    path: str,
    value: JsonValue,
    env: FormEnv,
) -> None:
    """Set a persistent Slot override, copying only traversed child keys inline.

    model describes data's generated editable family. path uses dotted Spec
    fields, never generated wrappers. Inline children keep their existing data;
    a captured child key is replaced by a detached supported Spec copy before
    descending. Other keys and the captured library remain unchanged. Invalid
    paths, missing keys, family mismatches and source failures propagate.
    """
    parts = path.split(".")
    if not all(parts):
        raise ValueError(f"Invalid seed path {path!r}")
    parent = data
    for part in parts[:-1]:
        if part not in model.model_fields:
            raise ValueError(f"Unknown seed path {path!r}")
        annotation = model.model_fields[part].annotation
        child = parent.get(part, {})
        if isinstance(child, str) and inline_wrapper(annotation) is not None:
            child = _captured_child(annotation, child, env)
        if not isinstance(child, dict):
            raise ValueError(f"Seed path traverses a non-inline field: {path!r}")
        parent[part] = child
        model = selected_model(spec_annotation(annotation), child)
        parent = child
    if parts[-1] not in model.model_fields:
        raise ValueError(f"Unknown seed path {path!r}")
    parent[parts[-1]] = value


def _path_annotation(
    model: type[BaseModel], data: Mapping[str, JsonValue], path: str
) -> object:
    head, separator, rest = path.partition(".")
    if head not in model.model_fields:
        raise ValueError(f"Unknown seed path {path!r}")
    annotation = spec_annotation(model.model_fields[head].annotation)
    if not separator:
        return annotation
    child = data.get(head, {})
    if not isinstance(child, dict):
        raise ValueError(f"Seed path requires inline object: {path!r}")
    return _path_annotation(selected_model(annotation, child), child, rest)


def _complete_template_input(
    model: type[BaseModel], authored: Mapping[str, JsonValue], template_id: str
) -> dict[str, JsonValue]:
    """Complete Spec input and identify its template in startup diagnostics."""
    try:
        return complete_input(model, authored)
    except ValueError as exc:
        raise ValueError(f"Template {template_id!r}: {exc}") from exc


def template_input(
    template: Template,
    catalog: Mapping[str, TemplateDef],
    expected: tuple[type[BaseModel], ...],
    env: FormEnv | None,
) -> dict[str, JsonValue]:
    if template.id not in catalog:
        raise ValueError(f"Unknown template {template.id!r}")
    definition = catalog[template.id]
    compatible = tuple(
        model for model in expected if issubclass(model, definition.module)
    )
    if len(compatible) != 1:
        raise ValueError(
            f"Template {template.id!r} has incompatible module {definition.module.__name__}"
        )
    model = compatible[0]
    # Catalog binding validates structure before input expansion.
    seeds = (
        dict(definition.seeds)
        if env is None
        else definition.expanded_seeds(bindings=template.bindings)
    )
    seeds.update(template.set)
    data: dict[str, JsonValue] = {}
    for path, seed in seeds.items():
        if not isinstance(seed, Slot):
            set_path(data, path, scalar_seed(seed))
    data = _complete_template_input(model, data, template.id)
    for path, seed in seeds.items():
        if isinstance(seed, Slot):
            annotation = _path_annotation(model, data, path)
            allowed = tuple(
                model
                for model in members(annotation)
                if isinstance(model, type) and issubclass(model, ConfigBase)
            )
            if not allowed:
                raise ValueError(f"Slot seed requires a module field: {path!r}")
            if (
                env is not None
                and (value := nested_slot(seed, allowed, env)) is not None
            ):
                set_path(data, path, value)
    return _complete_template_input(model, data, template.id)


def template_seed_tree(
    template: Template, catalog: Mapping[str, TemplateDef]
) -> dict[str, JsonValue]:
    """Describe authored defaults without reading sources.

    Call after template_input validates IDs, families and paths.
    This tree is only for deriving field requirements, never for Slot presence
    or actual input. Literal overrides replace earlier Expr seeds.
    """
    # Requirements depend on the authored seed kinds, not replacement values.
    seeds = dict(catalog[template.id].seeds)
    seeds.update(template.set)
    data: dict[str, JsonValue] = {}
    for path, seed in seeds.items():
        if isinstance(seed, Slot):
            nested: dict[str, JsonValue] = {}
            for child_path, value in seed.set.items():
                set_path(nested, child_path, declaration_seed(value))
            set_path(data, path, nested)
        else:
            set_path(data, path, declaration_seed(seed))
    return data


def nested_slot(
    slot: Slot,
    expected: tuple[type[BaseModel], ...],
    env: FormEnv,
) -> JsonValue:
    key = slot_key(slot, env)
    if key is not None and env.library is not None:
        spec = env.library.module_spec(key)
        if type_tag(type(spec)) not in {type_tag(model) for model in expected}:
            raise ValueError(f"Slot {slot.path!r} has incompatible module family")
        if not slot.copy:
            return key
        selected = next(
            model for model in expected if type_tag(model) == type_tag(type(spec))
        )
        data = persistent_input(
            selected,
            supported_input(selected, spec.model_dump(mode="json", exclude_none=True)),
        )
        for path, seed in slot.set.items():
            set_slot_path(selected, data, path, scalar_seed(seed), env)
        return data
    return None
