"""Translate core cfg declarations into Qt-free input models and typed cfgs."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from operator import methodcaller
from types import UnionType, new_class
from typing import Annotated, Literal, TypeAliasType, Union, get_args, get_origin

from pydantic import BaseModel, Field, JsonValue, ValidationError
from pydantic.fields import FieldInfo

from zcu_tools.cfg_model import (
    AliasName,
    ConfigBase,
    Expr,
    FormNode,
    Hidden,
    Locked,
    Range,
    get_field_meta,
)
from zcu_tools.gui.form import (
    Display,
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    LinRange,
    static_defaults,
)
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.resources.entry.roles import RoleResolutionError


class SweepRange(LinRange):
    """Core sweep input uses n internally, with the user-facing label Expts."""

    n: Annotated[int, Display(label="Expts")] = Field(default=3, ge=1)


def _members(annotation: object) -> tuple[object, ...]:
    if isinstance(annotation, TypeAliasType):
        return _members(annotation.__value__)
    args = get_args(annotation)
    if get_origin(annotation) is Annotated:
        return _members(args[0])
    if get_origin(annotation) in (UnionType, Union):
        return tuple(member for arg in args for member in _members(arg))
    return (annotation,)


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
    if isinstance(seed, Expr):
        return {"expr": seed.text}
    if isinstance(seed, complex):
        return {"__complex__": [seed.real, seed.imag]}
    if seed is None or isinstance(seed, (str, int, float, bool)):
        return seed
    raise TypeError(f"Not a scalar form seed: {seed!r}")


def _alias_name(node: AliasName, env: FormEnv) -> str | None:
    if env.parameters is None:
        return None
    path = node.alias if node.alias.startswith("$") else f"${node.alias}"
    try:
        return env.parameters.resolve_path(path)
    except RoleResolutionError:
        return None


def _field_info(info: FieldInfo) -> FieldInfo:
    result = deepcopy(info)
    meta = get_field_meta(info)
    if meta is not None:
        result.metadata.append(
            Display(label=meta.label, unit=meta.unit, description=meta.description)
        )
    return result


class _Section:
    """A derived section owns shape conversion, defaults and field-path errors."""

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
        self.children: dict[str, _Section] = {}
        namespace: dict[str, object] = {"__module__": __name__}
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
            field_info = _field_info(info)
            if isinstance(node, Locked):
                field_info.default = node.value
            annotations[name] = annotation
            namespace[name] = field_info
        section = self

        def defaults(
            cls: type[FormModel], env: FormEnv, previous: Mapping[str, JsonValue]
        ) -> dict[str, JsonValue]:
            del cls, previous
            return section.default_data(env)

        def choices(
            cls: type[FormModel], env: FormEnv
        ) -> Mapping[tuple[str | int, ...], tuple[JsonValue, ...]]:
            del cls
            return {
                (name,): tuple(env.devices)
                for name, node in section.tree.items()
                if isinstance(node, AliasName) and node.choices_source == "devices"
            }

        namespace.update(
            __annotations__=annotations,
            defaults=classmethod(defaults),
            choices=classmethod(choices),
        )
        model = new_class(
            f"{core.__name__}Form",
            (FormModel,),
            exec_body=lambda ns: ns.update(namespace),
        )
        if not issubclass(model, FormModel):
            raise TypeError("Generated model must implement FormModel")
        self.model = model

    def _annotation(self, name: str, info: FieldInfo, node: FormNode) -> object:
        if isinstance(node, Locked):
            # Dynamic Literal parameters are not expressible as static annotations.
            return methodcaller("__getitem__", node.value)(Literal)
        if SweepCfg in _members(info.annotation):
            return SweepRange
        child_core = _section_type(info.annotation)
        if child_core is None:
            return info.annotation
        child_tree = node if isinstance(node, Mapping) else {}
        child = _Section(child_core, child_tree, (*self.path, name))
        self.children[name] = child
        return (
            child.model | None
            if type(None) in _members(info.annotation)
            else child.model
        )

    def default_data(self, env: FormEnv) -> dict[str, JsonValue]:
        data = static_defaults(self.model)
        for name, node in self.tree.items():
            if isinstance(node, Hidden):
                continue
            if name in self.children:
                data[name] = self.children[name].default_data(env)
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
        for name, child in self.children.items():
            if name not in data and self.core.model_fields[name].is_required():
                data[name] = child.default_data(env)
        return data

    def to_raw(
        self, value: BaseModel, errors: dict[tuple[str | int, ...], str]
    ) -> dict[str, JsonValue]:
        raw = value.model_dump(mode="json")
        for name, child in self.children.items():
            item: object = getattr(value, name)
            if isinstance(item, BaseModel):
                raw[name] = child.to_raw(item, errors)
        for name in self.core.model_fields:
            if name not in value.__class__.model_fields:
                continue
            item: object = getattr(value, name)
            if isinstance(item, LinRange):
                try:
                    sweep = SweepCfg(
                        start=item.start,
                        stop=item.stop,
                        expts=item.n,
                        step=(item.stop - item.start) / (item.n - 1)
                        if item.n > 1
                        else 0,
                    )
                except ValidationError as exc:
                    errors[(*self.path, name)] = str(exc)
                else:
                    raw[name] = sweep.model_dump(mode="json")
        return raw


class Form[C: ConfigBase]:
    """Generate an input model from a core cfg and its authored starting tree.

    core_cfg is the validated result type; tree keys follow its nested field
    names. Unspecified fields derive shape and defaults from core declarations.
    Hidden requires a core default; unknown keys fail with ValueError at creation.
    model is a generated FormModel class, used only through the public form API.
    defaults resolve names once against the supplied captured FormEnv; hosts use
    the same env until Reset. Optional extras adds an 'extras' section and requires
    a named lower(extras, cfg) callable returning the concrete core cfg type.
    """

    def __init__[E: BaseModel](
        self,
        core_cfg: type[C],
        tree: Mapping[str, FormNode],
        *,
        extras: type[E] | None = None,
        lower: Callable[[E, C], C] | None = None,
    ) -> None:
        if (extras is None) != (lower is None):
            raise ValueError("extras and lower must be provided together")
        self.core_cfg = core_cfg
        self._section = _Section(core_cfg, tree, ())
        self.model: type[FormModel] = self._section.model
        self._lower: Callable[[BaseModel, C], C] | None = None
        if extras is not None and lower is not None:
            if "extras" in core_cfg.model_fields:
                raise ValueError("Core cfg already declares extras")

            def lower_model(value: BaseModel, cfg: C) -> C:
                return lower(extras.model_validate(value.model_dump()["extras"]), cfg)

            self._lower = lower_model
            parent = self.model
            namespace: dict[str, object] = {
                "__module__": __name__,
                "__annotations__": {"extras": extras},
            }
            namespace["extras"] = FieldInfo(annotation=extras, default_factory=extras)
            model = new_class(
                f"{core_cfg.__name__}WithExtrasForm",
                (parent,),
                exec_body=lambda ns: ns.update(namespace),
            )
            if not issubclass(model, FormModel):
                raise TypeError("Generated extras model must implement FormModel")
            self.model = model
            self._section.model = model

    def default_input(self, env: FormEnv) -> FormInput[FormModel]:
        """Build fresh input from captured env; required unresolved names stay absent.

        No sources are refreshed or mutated. Source and default factory failures
        propagate. Expressions remain envelopes until FormState.result evaluates
        them against this same env.
        """
        return self.model.default_input(env)

    def to_cfg(self, value: FormModel, env: FormEnv) -> C | FormErrors:
        """Convert this form's ordinary result into a validated concrete core cfg.

        value must be a result of this generated model; another model raises
        TypeError. env is the same captured source used by result. Core validation
        failures return FormErrors at their cfg field paths. Conversion does not
        mutate input or sources. Exceptions from the author-provided lower hook
        propagate; its result is validated again against core_cfg.
        """
        del env
        if not isinstance(value, self.model):
            raise TypeError("to_cfg requires this form's generated model result")
        errors: dict[tuple[str | int, ...], str] = {}
        raw = self._section.to_raw(value, errors)
        raw.pop("extras", None)
        if errors:
            return FormErrors(errors)
        try:
            cfg = self.core_cfg.model_validate(raw)
        except ValidationError as exc:
            return FormErrors(
                {tuple(error["loc"]): error["msg"] for error in exc.errors()}
            )
        if self._lower is not None:
            cfg = self.core_cfg.model_validate(self._lower(value, cfg).model_dump())
        return cfg
