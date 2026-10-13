"""Template seed expansion uses public static defaults and captured source APIs."""

from collections.abc import Mapping
from typing import Literal, get_args, get_origin

from pydantic import BaseModel, JsonValue

from zcu_tools.cfg_model import (
    ConfigBase,
    Expr,
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
    if isinstance(seed, Expr):
        return {"expr": seed.text}
    if isinstance(seed, complex):
        return {"__complex__": [seed.real, seed.imag]}
    if isinstance(seed, Slot):
        raise TypeError("Slot seeds require captured source resolution")
    return seed


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
        for tag in ("type", "style"):
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
    branch; string keys and scalar leaves are copied. data is not modified.
    """
    result: dict[str, JsonValue] = {}
    for name, value in data.items():
        if name not in model.model_fields:
            continue
        annotation = model.model_fields[name].annotation
        if isinstance(value, dict) and any(
            isinstance(member, type) and issubclass(member, BaseModel)
            for member in members(annotation)
        ):
            result[name] = supported_input(selected_model(annotation, value), value)
        else:
            result[name] = value
    return result


def complete_input(
    model: type[BaseModel],
    authored: Mapping[str, JsonValue],
    *,
    path: tuple[str, ...] = (),
) -> dict[str, JsonValue]:
    """Overlay seeds with static defaults; report ambiguous inline field paths."""
    unknown = authored.keys() - model.model_fields.keys()
    if unknown:
        raise ValueError(f"Unknown seeds for {model.__name__}: {sorted(unknown)}")
    data = static_defaults(model)
    data.update(authored)
    for name, value in tuple(data.items()):
        info = model.model_fields[name]
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
                for member in members(info.annotation)
            )
        ):
            child_path = (*path, name)
            try:
                child = selected_model(info.annotation, value)
            except ValueError as exc:
                raise ValueError(
                    f"{'.'.join(child_path)}: Cannot select inline Spec discriminator; "
                    "supply a tag when the Spec has no static default"
                ) from exc
            data[name] = complete_input(child, value, path=child_path)
    return data


def _path_annotation(
    model: type[BaseModel], data: Mapping[str, JsonValue], path: str
) -> object:
    head, separator, rest = path.partition(".")
    if head not in model.model_fields:
        raise ValueError(f"Unknown seed path {path!r}")
    annotation = model.model_fields[head].annotation
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
    *,
    active: tuple[str, ...] = (),
) -> dict[str, JsonValue]:
    if template.id in active:
        raise ValueError(f"Recursive template fallback: {(*active, template.id)!r}")
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
    seeds = definition.expanded_seeds(
        qubit=template.qubit, resonator=template.resonator
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
            if seed.fallback is not None:
                template_input(
                    seed.fallback, catalog, allowed, None, active=(*active, template.id)
                )
            if (
                env is not None
                and (
                    value := nested_slot(
                        seed, catalog, allowed, env, active=(*active, template.id)
                    )
                )
                is not None
            ):
                set_path(data, path, value)
    for path, value in template.lock.items():
        _path_annotation(model, data, path)
        set_path(data, path, scalar_seed(value))
    return _complete_template_input(model, data, template.id)


def template_seed_tree(
    template: Template, catalog: Mapping[str, TemplateDef]
) -> dict[str, JsonValue]:
    """Describe authored defaults, including nested fallbacks, without reading sources.

    Call after template_input validates IDs, families, paths and recursion.
    This tree is only for deriving field requirements, never for Slot presence
    or actual input. Literal overrides and locks replace earlier Expr seeds.
    """
    seeds = catalog[template.id].expanded_seeds(
        qubit=template.qubit, resonator=template.resonator
    )
    seeds.update(template.set)
    data: dict[str, JsonValue] = {}
    for path, seed in seeds.items():
        if isinstance(seed, Slot):
            nested = (
                template_seed_tree(seed.fallback, catalog)
                if seed.fallback is not None
                else {}
            )
            for child_path, value in seed.set.items():
                set_path(nested, child_path, scalar_seed(value))
            for child_path, value in seed.lock.items():
                set_path(nested, child_path, scalar_seed(value))
            set_path(data, path, nested)
        else:
            set_path(data, path, scalar_seed(seed))
    for path, value in template.lock.items():
        set_path(data, path, scalar_seed(value))
    return data


def nested_slot(
    slot: Slot,
    catalog: Mapping[str, TemplateDef],
    expected: tuple[type[BaseModel], ...],
    env: FormEnv,
    *,
    active: tuple[str, ...] = (),
) -> JsonValue:
    key = slot_key(slot, env)
    if key is not None and env.library is not None:
        spec = env.library.module_spec(key)
        if type_tag(type(spec)) not in {type_tag(model) for model in expected}:
            raise ValueError(f"Slot {slot.path!r} has incompatible module family")
        if not slot.copy and not slot.lock:
            return key
        selected = next(
            model for model in expected if type_tag(model) == type_tag(type(spec))
        )
        data = supported_input(
            selected, spec.model_dump(mode="json", exclude_none=True)
        )
        for path, seed in slot.set.items():
            set_path(data, path, scalar_seed(seed))
        for path, value in slot.lock.items():
            set_path(data, path, scalar_seed(value))
        return data
    if slot.fallback is not None:
        return template_input(slot.fallback, catalog, expected, env, active=active)
    return None
