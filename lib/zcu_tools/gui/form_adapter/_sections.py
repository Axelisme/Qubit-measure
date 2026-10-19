"""Generate recursive cfg sections, their defaults and native value projections."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from functools import partial
from types import new_class
from typing import Annotated, get_args, get_origin

from pydantic import BaseModel, Field, JsonValue, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from zcu_tools.cfg_model import (
    AliasName,
    CfgInputValue,
    ConfigBase,
    Expr,
    FormNode,
    Hidden,
    Locked,
    ModuleSpec,
    Range,
    Slot,
    Template,
    TemplateDef,
    WaveformSpec,
    get_field_meta,
)
from zcu_tools.gui.form import (
    Display,
    FormEnv,
    FormModel,
    FormWritePath,
    InputConverter,
    Option,
    static_defaults,
)
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.resources.entry.roles import RoleResolutionError

from ._display import field_display
from ._modules import (
    ModuleField,
    editable_spec,
    module_specs,
)
from ._seeds import persistent_input
from ._types import literal_type, union_type
from ._types import members as _members
from .fixed_elements import FixedElementsForm
from .sweep import CenterSpan, LinRange, StepCenter, StepRange


def _sweep_branch(
    base: type[FormModel], node: FormNode, decimals: int | None
) -> type[FormModel]:
    """Specialize authored endpoint requirements and float display precision."""
    fields: dict[str, FieldInfo] = {}
    for name, info in base.model_fields.items():
        if info.annotation is not float:
            continue
        required = (
            isinstance(node, Range)
            and name in ("start", "stop")
            and (getattr(node, name) is None or isinstance(getattr(node, name), Expr))
        )
        if not required and decimals is None:
            continue
        endpoint = deepcopy(info)
        if required:
            endpoint.default = PydanticUndefined
            endpoint.default_factory = None
        if decimals is not None:
            endpoint.metadata.append(Display(decimals=decimals))
        fields[name] = endpoint
    model = new_class(
        f"Authored{base.__name__}",
        (base,),
        exec_body=lambda ns: ns.update(
            {
                "__module__": "zcu_tools.gui.form_adapter.form",
                "__annotations__": {name: float for name in fields},
                **fields,
            }
        ),
    )
    if not issubclass(model, FormModel):
        raise TypeError("Generated sweep model must implement FormModel")
    return model


def _sweep_model(node: FormNode, info: FieldInfo) -> object:
    """Build four active branches; Expr and None endpoints lose inherited defaults.

    node is the authored sweep; info is its core field declaration. Float
    precision applies to all coordinates and step, never n or the outer model.
    No expression is evaluated and no shared branch class is modified.
    """
    meta = get_field_meta(info)
    decimals = meta.decimals if meta is not None else None
    branches = tuple(
        _sweep_branch(base, node, decimals)
        for base in (LinRange, CenterSpan, StepRange, StepCenter)
    )
    return Annotated[union_type(branches), Field(discriminator="kind")]


def _section_type(annotation: object) -> type[BaseModel] | None:
    members = tuple(
        member for member in _members(annotation) if member is not type(None)
    )
    if (
        len(members) == 1
        and isinstance(model := members[0], type)
        and issubclass(model, BaseModel)
        and not issubclass(model, SweepCfg)
    ):
        return model
    return None


def _seed_json(seed: FormNode) -> JsonValue:
    if isinstance(seed, list):
        return [_seed_json(item) for item in seed]
    if isinstance(seed, Expr):
        return {"expr": seed.text}
    if isinstance(seed, complex):
        return {"__complex__": [seed.real, seed.imag]}
    if seed is None or isinstance(seed, (str, int, float, bool)):
        return seed
    raise TypeError(f"Not a scalar form seed: {seed!r}")


def _check_position_seed(node: FormNode, path: FormWritePath) -> None:
    """Position trees supply values, never alter the shared element declaration."""
    if isinstance(node, (Locked, Hidden)) or (
        isinstance(node, (Slot, Template)) and node.lock
    ):
        raise ValueError(
            f"{path!r}: position seed cannot change the shared element declaration "
            f"with {type(node).__name__}; declare shared constraints on the element type"
        )
    if isinstance(node, Mapping):
        for name, child in node.items():
            _check_position_seed(child, (*path, name))
    elif isinstance(node, list):
        for index, child in enumerate(node):
            _check_position_seed(child, (*path, index))


def _alias_name(node: AliasName, env: FormEnv) -> str | None:
    if env.parameters is None:
        return None
    path = node.alias if node.alias.startswith("$") else f"${node.alias}"
    try:
        return env.parameters.resolve_path(path)
    except RoleResolutionError:
        return None


def _field_info(info: FieldInfo, *, locked: bool) -> FieldInfo:
    result = deepcopy(info)
    display = field_display(info, locked=locked)
    if SweepCfg in _members(info.annotation) and not locked:
        display = replace(display or Display(), decimals=None, selector=False)
    if display is not None:
        result.metadata.append(display)
    return result


PERSISTENT_SPECS = frozenset(
    model
    for model in (*_members(ModuleSpec), *_members(WaveformSpec))
    if isinstance(model, type)
)


class SpecSection:
    """A persistent module or waveform Spec root converts input back to that Spec.

    The generated model subclasses both FormModel and the Spec's input model, so
    hosts use it as a FormState root or as a ModuleInline module field. Child
    library keys use tagged inputs and lower back to strings; nothing is
    looked up or materialized.
    spec is a concrete persistent ConfigBase type; tree must be empty or
    construction raises ValueError. editable is its input projection; model is
    the generated root class. children and modules are empty mappings, exposing
    the same traversal interface as Section. No sources are read at construction.
    """

    def __init__(self, spec: type[ConfigBase], tree: Mapping[str, FormNode]) -> None:
        if tree:
            raise ValueError(f"{spec.__name__}: Spec forms take no authored tree")
        self.editable = editable_spec(spec, {}, {})
        model = new_class(
            f"{spec.__name__}Form",
            (self.editable,),
            exec_body=lambda ns: ns.update(
                {"__module__": "zcu_tools.gui.form_adapter.form"}
            ),
        )
        if not issubclass(model, FormModel):
            raise TypeError("Generated Spec model must implement FormModel")
        self.model: type[FormModel] = model
        self.children: dict[str, Section] = {}
        self.modules: dict[str, ModuleField] = {}

    def bind_templates(self, catalog: Mapping[str, TemplateDef], adapter: str) -> None:
        """Accept the host catalog and adapter name; Spec roots have no seed nodes."""
        # The section contains no Template or Slot node to resolve.
        del catalog, adapter

    def to_raw(
        self, value: BaseModel, env: FormEnv, errors: dict[tuple[str | int, ...], str]
    ) -> dict[str, CfgInputValue]:
        """Project a validated model back to Spec fields, preserving child keys.

        env and errors follow Section.to_raw but remain unused: Spec projection
        neither reads sources nor materializes native modules.
        """
        del env, errors
        return {
            name: item
            for name, item in persistent_input(
                type(value), value.model_dump(mode="json")
            ).items()
        }


def _module_converters(
    fields: Mapping[str, ModuleField],
    _cls: type[FormModel],
    env: FormEnv,
) -> Mapping[tuple[str | int, ...], InputConverter]:
    return {(name,): field.converter(env) for name, field in fields.items()}


class Section:
    """Generate a recursive input section from a native model and authored tree.

    core supplies field annotations; tree supplies seeds and declaration nodes;
    path is the enclosing model-field path for diagnostics. Unknown fields,
    unsupported declarations and position-specific list constraints raise at
    construction without reading sources.

    model is the generated FormModel class. core, tree and path retain those
    declarations. children maps ordinary model fields to sections; lists maps
    model-list fields to their shared element section. element_seeds maps list
    fields to per-position initial-value sections. modules maps module fields
    to their conversion declarations. These mappings remain owned here.
    """

    def __init__(
        self, core: type[BaseModel], tree: Mapping[str, FormNode], path: tuple[str, ...]
    ) -> None:
        unknown = tree.keys() - core.model_fields.keys()
        if unknown:
            raise ValueError(
                f"{'.'.join(path) or core.__name__}: unknown fields {sorted(unknown)}"
            )
        self.core = core
        self.tree = dict(tree)
        self.path = path
        self.children: dict[str, Section] = {}
        self.lists: dict[str, Section] = {}
        self.element_seeds: dict[str, tuple[Section, ...]] = {}
        self.modules: dict[str, ModuleField] = {}
        namespace: dict[str, object] = {"__module__": "zcu_tools.gui.form_adapter.form"}
        annotations: dict[str, object] = {}
        program_fields = (
            ProgramV2Cfg.model_fields if issubclass(core, ProgramV2Cfg) else {}
        )
        names = [name for name in core.model_fields if name not in program_fields] + [
            name for name in core.model_fields if name in program_fields
        ]
        for name in names:
            info = core.model_fields[name]
            node = tree.get(name)
            if isinstance(node, Hidden):
                if info.is_required():
                    raise ValueError(
                        f"{'.'.join((*path, name))}: Hidden requires a core default"
                    )
                continue
            annotation = self._annotation(name, info, node)
            field_info = _field_info(info, locked=isinstance(node, Locked))
            if (name in self.modules and not self.modules[name].locks) or (
                not isinstance(node, Locked) and SweepCfg in _members(info.annotation)
            ):
                field_info.discriminator = "kind"
            if isinstance(node, Expr):
                field_info.default = PydanticUndefined
                field_info.default_factory = None
            if isinstance(node, Locked):
                field_info.default = node.value
            annotations[name] = annotation
            namespace[name] = field_info
        section = self

        def defaults(cls: type[FormModel], env: FormEnv) -> dict[str, JsonValue]:
            del cls
            return section.default_data(env)

        def choices(
            cls: type[FormModel], env: FormEnv
        ) -> Mapping[tuple[str | int, ...], tuple[JsonValue, ...]]:
            del cls
            return section._alias_choices(env)

        def options(
            cls: type[FormModel], env: FormEnv
        ) -> Mapping[tuple[str | int, ...], tuple[Option, ...]]:
            result: dict[tuple[str | int, ...], tuple[Option, ...]] = {
                (name,): field.options(env) for name, field in section.modules.items()
            }
            # Empty device defaults need a label; choices remains the value policy.
            result.update(
                {
                    path: tuple(Option(str(value) or "None", value) for value in values)
                    for path, values in cls.choices(env).items()
                }
            )
            return result

        namespace.update(
            converters=classmethod(partial(_module_converters, section.modules)),
            __annotations__=annotations,
            defaults=classmethod(defaults),
            choices=classmethod(choices),
            options=classmethod(options),
        )
        model = new_class(
            f"{core.__name__}Form",
            (FixedElementsForm,),
            exec_body=lambda ns: ns.update(namespace),
        )
        if not issubclass(model, FormModel):
            raise TypeError("Generated model must implement FormModel")
        self.model = model

    def _alias_choices(
        self, env: FormEnv
    ) -> dict[tuple[str | int, ...], tuple[JsonValue, ...]]:
        """Share core device defaults and captured names with descriptions and validation."""
        result: dict[tuple[str | int, ...], tuple[JsonValue, ...]] = {}
        for name, node in self.tree.items():
            if not isinstance(node, AliasName) or node.choices_source != "devices":
                continue
            info = self.core.model_fields[name]
            values: tuple[JsonValue, ...] = (
                (None, *env.devices)
                if type(None) in _members(info.annotation)
                else tuple(env.devices)
            )
            # Unresolved aliases use the core default without writing input.
            if isinstance(info.default, str) and info.default not in values:
                values = (info.default, *values)
            result[(name,)] = values
        return result

    def _annotation(self, name: str, info: FieldInfo, node: FormNode) -> object:
        if isinstance(node, Locked):
            return literal_type(node.value)
        if module_specs(info.annotation):
            field = ModuleField(
                info.annotation,
                (*self.path, name),
                node if isinstance(node, (Slot, Template)) else None,
            )
            self.modules[name] = field
            return (
                union_type((field.annotation, type(None)))
                if type(None) in _members(info.annotation)
                else field.annotation
            )
        if SweepCfg in _members(info.annotation):
            return _sweep_model(node, info)
        if get_origin(info.annotation) is list:
            element_type = get_args(info.annotation)[0]
            if isinstance(element_type, type) and issubclass(element_type, ConfigBase):
                child = Section(element_type, {}, (*self.path, name))
                self.lists[name] = child
                if node is not None:
                    if not isinstance(node, list) or any(
                        not isinstance(item, Mapping) for item in node
                    ):
                        raise TypeError(
                            f"{(*self.path, name)!r}: expected a list of element trees"
                        )
                    for index, item in enumerate(node):
                        _check_position_seed(item, (*self.path, name, index))
                    self.element_seeds[name] = tuple(
                        Section(element_type, item, (*self.path, name))
                        for item in node
                        if isinstance(item, Mapping)
                    )
                return list[child.model]
        child_core = _section_type(info.annotation)
        if child_core is None:
            return info.annotation
        child_tree = node if isinstance(node, Mapping) else {}
        child = Section(child_core, child_tree, (*self.path, name))
        self.children[name] = child
        return (
            child.model | None
            if type(None) in _members(info.annotation)
            else child.model
        )

    def _sweep_defaults(self) -> dict[str, JsonValue]:
        """Select Range + Expts for fresh, otherwise unauthored sweep fields."""
        return {
            name: static_defaults(LinRange)
            for name, info in self.core.model_fields.items()
            if name in self.model.model_fields and SweepCfg in _members(info.annotation)
        }

    def _list_defaults(self, data: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        for name, child in self.lists.items():
            native = data.get(name)
            if isinstance(native, list):
                result[name] = [
                    child.native_data(item) if isinstance(item, dict) else item
                    for item in native
                ]
        return result

    def default_data(self, env: FormEnv) -> dict[str, JsonValue]:
        """Build fresh raw defaults from seeds and captured env, without mutation.

        Missing aliases remain absent. Module/template source failures propagate;
        expression seeds remain raw for Core default evaluation.
        """
        data = {**self._sweep_defaults(), **static_defaults(self.model)}
        data.update(self._list_defaults(data))
        for name, node in self.tree.items():
            if isinstance(node, Hidden):
                continue
            if name in self.element_seeds:
                data[name] = [
                    child.default_data(env) for child in self.element_seeds[name]
                ]
            elif name in self.children:
                data[name] = self.children[name].default_data(env)
            elif name in self.modules:
                data.pop(name, None)
                if (module := self.modules[name].initial_input(env)) is not None:
                    data[name] = module
            elif isinstance(node, Range):
                data[name] = {
                    "kind": "linrange",
                    "start": _seed_json(node.start),
                    "stop": _seed_json(node.stop),
                    "n": node.expts,
                }
            elif isinstance(node, AliasName):
                data.pop(name, None)
                if (value := _alias_name(node, env)) is not None:
                    data[name] = value
            elif isinstance(node, Locked):
                data[name] = node.value
            else:
                data[name] = _seed_json(node)
        # Only core-disabled modules get an explicit selection. Existing initial
        # values win, while nullable fields without a None default stay missing.
        data = {
            name: None
            for name in self.modules
            if self.core.model_fields[name].default is None
        } | data
        data.update(self._required_children(data, env))
        return data

    def _required_children(
        self, data: Mapping[str, JsonValue], env: FormEnv
    ) -> dict[str, JsonValue]:
        return {
            name: child.default_data(env)
            for name, child in self.children.items()
            if name not in data and self.core.model_fields[name].is_required()
        }

    def native_data(self, data: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """Project a native element default recursively, without reading sources."""
        result = dict(data)
        for name, field in self.modules.items():
            item = result.get(name)
            if isinstance(item, dict):
                result[name] = field.native_input(item)
        for name, child in self.children.items():
            item = result.get(name)
            if isinstance(item, dict):
                result[name] = child.native_data(item)
        for name, child in self.lists.items():
            items = result.get(name)
            if isinstance(items, list):
                result[name] = [
                    child.native_data(item) if isinstance(item, dict) else item
                    for item in items
                ]
        for name, info in self.core.model_fields.items():
            item = result.get(name)
            if SweepCfg in _members(info.annotation) and isinstance(item, dict):
                result[name] = {
                    "kind": "linrange",
                    "start": item.get("start"),
                    "stop": item.get("stop"),
                    "n": item.get("expts"),
                }
        return result

    def bind_templates(self, catalog: Mapping[str, TemplateDef], adapter: str) -> None:
        """Validate and bind a detached catalog recursively, then rebuild model.

        adapter names the host definition in ValueError diagnostics. Unknown
        templates or malformed seeds raise before the section can be published.
        """
        for field in self.modules.values():
            try:
                field.validate_templates(catalog)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Adapter {adapter!r}, {'.'.join(field.path)}: {exc}"
                ) from exc
        for child in (*self.children.values(), *self.lists.values()):
            child.bind_templates(catalog, adapter)
        for seeds in self.element_seeds.values():
            for seed in seeds:
                seed.bind_templates(catalog, adapter)
        for field in self.modules.values():
            field.catalog = catalog
        self.model.model_rebuild(force=True)

    def _list_raw(
        self,
        value: BaseModel,
        env: FormEnv,
        errors: dict[FormWritePath, str],
        prefix: FormWritePath,
    ) -> dict[str, CfgInputValue]:
        raw: dict[str, CfgInputValue] = {}
        for name, child in self.lists.items():
            items: object = getattr(value, name)
            if not isinstance(items, list) or any(
                not isinstance(item, BaseModel) for item in items
            ):
                raise TypeError(
                    f"{(*prefix, name)!r}: expected validated model elements"
                )
            raw[name] = [
                child.to_raw(item, env, errors, (*prefix, name, index))
                for index, item in enumerate(items)
            ]
        return raw

    def to_raw(
        self,
        value: BaseModel,
        env: FormEnv,
        errors: dict[tuple[str | int, ...], str],
        path: FormWritePath | None = None,
    ) -> dict[str, CfgInputValue]:
        """Project a validated generated value into native cfg constructor data.

        env supplies captured module sources. Conversion errors accumulate in
        errors at indexed field paths; callers must reject a nonempty mapping.
        path overrides the declaration path during list-element traversal.
        Sources and value are not mutated; unrelated provider failures propagate.
        """
        prefix = self.path if path is None else path
        raw: dict[str, CfgInputValue] = value.model_dump(mode="python")
        raw.update(self._list_raw(value, env, errors, prefix))
        for name, child in self.children.items():
            item: object = getattr(value, name)
            if isinstance(item, BaseModel):
                raw[name] = child.to_raw(item, env, errors, (*prefix, name))
        for name, field in self.modules.items():
            item: object = getattr(value, name)
            if item is None:
                raw.pop(name, None)
            elif isinstance(item, BaseModel):
                try:
                    raw[name] = field.materialize(item, env)
                except (ValueError, TypeError) as exc:
                    errors[(*prefix, name)] = str(exc)
        for name in self.core.model_fields:
            if name not in value.__class__.model_fields:
                continue
            item: object = getattr(value, name)
            if isinstance(item, LinRange | CenterSpan | StepRange | StepCenter):
                try:
                    sweep = SweepCfg(
                        start=item.start,
                        stop=item.stop,
                        expts=item.n,
                        step=item.step
                        if isinstance(item, StepRange | StepCenter)
                        else (item.stop - item.start) / (item.n - 1)
                        if item.n > 1
                        else 0,
                    )
                except ValidationError as exc:
                    errors[(*prefix, name)] = str(exc)
                else:
                    raw[name] = sweep.model_dump(mode="json")
        return raw
