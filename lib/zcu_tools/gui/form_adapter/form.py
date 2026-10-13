"""Translate core cfg declarations into Qt-free input models and typed cfgs."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from types import MappingProxyType, new_class
from typing import Annotated

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
    Range,
    Slot,
    Template,
    TemplateDef,
    get_field_meta,
)
from zcu_tools.gui.form import (
    Display,
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormWritePath,
    LinRange,
    static_defaults,
)
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.resources.entry.roles import RoleResolutionError

from ._modules import ModuleField, ModuleValues, module_specs
from ._types import literal_type, union_type
from ._types import members as _members


class SweepRange(LinRange):
    """Core sweep input uses n internally, with the user-facing label Expts."""

    n: Annotated[int, Display(label="Expts")] = Field(default=3, ge=1)


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
        self.modules: dict[str, ModuleField] = {}
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
            if name in self.modules and not self.modules[name].locks:
                field_info.discriminator = "kind"
            if isinstance(node, Expr):
                field_info.default = PydanticUndefined
                field_info.default_factory = None
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
                (name,): (
                    (None, *env.devices)
                    if type(None)
                    in _members(section.core.model_fields[name].annotation)
                    else tuple(env.devices)
                )
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
        for name, child in self.children.items():
            if name not in data and self.core.model_fields[name].is_required():
                data[name] = child.default_data(env)
        return data

    def bind_templates(self, catalog: Mapping[str, TemplateDef], adapter: str) -> None:
        for field in self.modules.values():
            try:
                field.validate_templates(catalog)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Adapter {adapter!r}, {'.'.join(field.path)}: {exc}"
                ) from exc
        for child in self.children.values():
            child.bind_templates(catalog, adapter)
        for field in self.modules.values():
            field.catalog = catalog
        self.model.model_rebuild(force=True)

    def to_raw(
        self, value: BaseModel, env: FormEnv, errors: dict[tuple[str | int, ...], str]
    ) -> dict[str, CfgInputValue]:
        raw: dict[str, CfgInputValue] = value.model_dump(mode="python")
        for name, child in self.children.items():
            item: object = getattr(value, name)
            if isinstance(item, BaseModel):
                raw[name] = child.to_raw(item, env, errors)
        for name, field in self.modules.items():
            item: object = getattr(value, name)
            if item is None:
                raw.pop(name, None)
            elif isinstance(item, BaseModel):
                try:
                    raw[name] = field.materialize(item, env)
                except (ValueError, TypeError) as exc:
                    errors[(*self.path, name)] = str(exc)
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
    default_input resolves names once against the supplied captured FormEnv;
    hosts use the same env until Reset. Register with Registry.register(form=...)
    or call bind_templates before resolving Template/fallback nodes. Slot copies
    and Template defaults are detached persistent Specs, not library-editor
    structural defaults. Only Expr nodes become expression envelopes. Ordinary
    strings, including $-prefixed strings, remain literal. Expr initial values
    remove static defaults in this form only; failed evaluation requires input
    instead of falling back to core defaults. Module locks remove
    the ref branch and use single-value Literal fields in inline content. Optional extras adds an 'extras' section and requires
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
        self._core_cfg = core_cfg
        self._section = _Section(core_cfg, tree, ())
        self.model: type[FormModel] = self._section.model
        self._lower: Callable[[BaseModel, ConfigBase], C] | None = None
        if extras is not None and lower is not None:
            if "extras" in core_cfg.model_fields:
                raise ValueError("Core cfg already declares extras")

            def lower_model(value: BaseModel, cfg: ConfigBase) -> C:
                return lower(
                    extras.model_validate(value.model_dump()["extras"]),
                    core_cfg.model_validate(cfg),
                )

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

    @property
    def core_cfg(self) -> type[C]:
        """Return the fixed validated cfg result type; the definition cannot retarget it."""
        return self._core_cfg

    def bind_templates(
        self, templates: Mapping[str, TemplateDef], *, adapter: str
    ) -> None:
        """Bind a detached catalog and validate authored templates before publication.

        Registry.register calls this explicitly. adapter is the registry name,
        included with the consuming field path in ValueError for unknown IDs,
        incompatible families or malformed seed trees. No sources are read.
        Call before default_input when the tree contains Template/fallback nodes.
        Definitions remain immutable and the caller's mapping is never retained.
        """
        self._section.bind_templates(MappingProxyType(dict(templates)), adapter)

    def default_input(self, env: FormEnv) -> FormInput[FormModel]:
        """Build fresh input from captured env; required unresolved names stay absent.

        No sources are refreshed or mutated. Source and default factory failures
        propagate. The shared core evaluates expression defaults once into direct
        input against this env. Unresolved or invalid defaults stay absent.
        """
        return self.model.default_input(env)

    def _module_field(self, path: tuple[str, ...]) -> ModuleField:
        section = self._section
        if not path or any(not part for part in path):
            raise ValueError("Module path must contain nonempty field names")
        for part in path[:-1]:
            if part not in section.children:
                raise ValueError(f"Not a core module field: {'.'.join(path)}")
            section = section.children[part]
        if path[-1] not in section.modules:
            raise ValueError(f"Not a core module field: {'.'.join(path)}")
        return section.modules[path[-1]]

    def module_inline_input(
        self, input: FormInput[FormModel], path: tuple[str, ...], env: FormEnv
    ) -> dict[str, JsonValue]:
        """Prepare inline input for a core module path against fixed env.

        input must belong to this Form; otherwise raise TypeError. path consists
        of nonempty core field names, not module descendants; invalid paths raise
        ValueError. Copy a captured library Spec through ModuleInline.defaults;
        absent library/key uses Template fallback or editable static blank.
        Wrong family/source failures propagate. Preserve existing inline values
        and locks. No mutation, validation of the whole draft, or publication
        occurs; blank may still require ordinary leaf edits before Run.
        """
        if input.model is not self.model:
            raise TypeError("module_inline_input requires this Form's input")
        field = self._module_field(path)
        raw: JsonValue = input.data
        for part in path:
            raw = raw.get(part) if isinstance(raw, dict) else None
        return field.inline_input(raw if isinstance(raw, dict) else {}, env)

    def module_inline_writes(
        self, input: FormInput[FormModel], path: tuple[str, ...], env: FormEnv
    ) -> dict[FormWritePath, JsonValue]:
        """Prepare one public leaf batch for required or nullable module input.

        input, path and env follow module_inline_input, with the same failures,
        locks and captured-library/Template/blank preparation. The generated
        inline branch defaults use that same preparation during discriminator
        switching. Return no writes for already-inline input. The resource must
        apply this batch once through guarded public write; preparation changes
        no input and publishes nothing. Callers need not know module layout.
        """
        if input.model is not self.model:
            raise TypeError("module_inline_writes requires this Form's input")
        field = self._module_field(path)
        raw: JsonValue = input.data
        for part in path:
            raw = raw.get(part) if isinstance(raw, dict) else None
        previous = raw if isinstance(raw, dict) else {}
        return (
            {}
            if previous.get("kind") == "inline"
            else field.inline_writes(previous, env)
        )

    def read_module(
        self, input: FormInput[FormModel], path: tuple[str, ...], env: FormEnv
    ) -> ModuleValues | FormErrors:
        """Read this module field's evaluated value without mutation or refresh.

        input must belong to this Form; other models raise TypeError. path uses
        nonempty core field names; unknown/nonmodule paths raise ValueError.
        env is the fixed capture used to evaluate input. Key mode materializes
        its captured library Spec; inline mode evaluates this field's input.
        Other invalid draft fields do not block this read. Only this module's
        validation/source failures return FormErrors at the consuming path.
        Values exclude full-cfg validation and extras lowering: this is not the
        native cfg after to_cfg. Unexpected source exceptions propagate.
        """
        if input.model is not self.model:
            raise TypeError("read_module requires this Form's input")
        field = self._module_field(path)
        raw: JsonValue = input.data
        for part in path:
            raw = raw.get(part) if isinstance(raw, dict) else None
        return field.read(raw, env)

    def to_cfg(self, value: FormModel, env: FormEnv) -> C | FormErrors:
        """Convert this form's ordinary result into a validated concrete core cfg.

        value must be a result of this generated model; another model raises
        TypeError. env is the same captured source used by result. Core validation
        failures return FormErrors at their cfg field paths. Conversion does not
        mutate input or sources. Exceptions from the author-provided lower hook
        propagate; its result is validated again against core_cfg.
        """
        if not isinstance(value, self.model):
            raise TypeError("to_cfg requires this form's generated model result")
        errors: dict[tuple[str | int, ...], str] = {}
        raw = self._section.to_raw(value, env, errors)
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
