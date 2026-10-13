"""Translate core cfg declarations into Qt-free input models and typed cfgs."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
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
    FormErrors,
    FormInput,
    FormModel,
    FormState,
    FormWritePath,
    LinRange,
    static_defaults,
)
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.resources.entry.roles import RoleResolutionError

from ._display import field_display
from ._modules import ModuleField, ModuleValues, editable_spec, module_specs
from ._seeds import supported_input, template_seed_tree
from ._seeds import template_input as _template_input
from ._types import literal_type, union_type
from ._types import members as _members


class SweepRange(LinRange):
    """Core sweep input uses n internally, with the user-facing label Expts."""

    n: Annotated[int, Display(label="Expts")] = Field(default=3, ge=1)


def _sweep_model(node: FormNode, info: FieldInfo) -> type[SweepRange]:
    """Generate endpoint requirements and float precision for one core sweep.

    node is the authored sweep value; info is its core field declaration.
    Only Expr endpoints lose the inherited LinRange default. FieldMeta decimals
    decorates start/stop, never n or the outer model. Literal or unauthored
    endpoints keep ordinary defaults. No expression is evaluated, and neither
    the core declaration nor the shared SweepRange class is modified.
    """
    meta = get_field_meta(info)
    decimals = meta.decimals if meta is not None else None
    fields: dict[str, FieldInfo] = {}
    for name in ("start", "stop"):
        required = isinstance(node, Range) and isinstance(getattr(node, name), Expr)
        if not required and decimals is None:
            continue
        endpoint = deepcopy(SweepRange.model_fields[name])
        if required:
            endpoint.default = PydanticUndefined
            endpoint.default_factory = None
        if decimals is not None:
            endpoint.metadata.append(Display(decimals=decimals))
        fields[name] = endpoint
    if not fields:
        return SweepRange
    namespace: dict[str, object] = {
        "__module__": __name__,
        "__annotations__": {name: float for name in fields},
        **fields,
    }
    model = new_class(
        "AuthoredSweepRange", (SweepRange,), exec_body=lambda ns: ns.update(namespace)
    )
    if not issubclass(model, SweepRange):
        raise TypeError("Generated sweep model must implement SweepRange")
    return model


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


def _field_info(info: FieldInfo, *, locked: bool) -> FieldInfo:
    result = deepcopy(info)
    if (display := field_display(info, locked=locked)) is not None:
        if SweepCfg in _members(info.annotation):
            display = replace(display, decimals=None)
        result.metadata.append(display)
    return result


_PERSISTENT_SPECS = frozenset(
    model
    for model in (*_members(ModuleSpec), *_members(WaveformSpec))
    if isinstance(model, type)
)


class _SpecSection:
    """A persistent module or waveform Spec root converts input back to that Spec.

    The generated model subclasses both FormModel and the Spec's input model, so
    hosts use it as a FormState root or as a ModuleInline module field. Child
    library keys stay ordinary strings; nothing is looked up or materialized.
    It has no core sections or module fields, and no authored tree.
    """

    def __init__(self, spec: type[ConfigBase], tree: Mapping[str, FormNode]) -> None:
        if tree:
            raise ValueError(f"{spec.__name__}: Spec forms take no authored tree")
        self.editable = editable_spec(spec, {}, {})
        model = new_class(
            f"{spec.__name__}Form",
            (FormModel, self.editable),
            exec_body=lambda ns: ns.update({"__module__": __name__}),
        )
        if not issubclass(model, FormModel):
            raise TypeError("Generated Spec model must implement FormModel")
        self.model: type[FormModel] = model
        self.children: dict[str, _Section] = {}
        self.modules: dict[str, ModuleField] = {}

    def bind_templates(self, catalog: Mapping[str, TemplateDef], adapter: str) -> None:
        # The section contains no Template or Slot node to resolve.
        del catalog, adapter

    def to_raw(
        self, value: BaseModel, env: FormEnv, errors: dict[tuple[str | int, ...], str]
    ) -> dict[str, CfgInputValue]:
        del env, errors
        return value.model_dump(mode="python")


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
            field_info = _field_info(info, locked=isinstance(node, Locked))
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
            return _sweep_model(node, info)
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
    hosts use the same env until Reset. The host calls bind_templates with its
    template catalog before resolving Template/fallback nodes. Slot copies
    and Template defaults are detached persistent Specs, not library-editor
    structural defaults. Only Expr nodes become expression envelopes. Ordinary
    strings, including $-prefixed strings, remain literal. Expr initial values
    remove static defaults in this form only; failed evaluation requires input
    instead of falling back to core defaults. Module locks remove
    the ref branch and use single-value Literal fields in inline content. Optional extras adds an 'extras' section and requires
    a named lower(extras, cfg) callable returning the concrete core cfg type.
    Generated module input omits optional desc, mask, outsel, ro_ch and mux_*
    fields; conversion restores their Spec defaults, and copying a library
    entry inline discards its stored values for them. Templates cannot seed
    them (see bind_templates); there is no override. FieldMeta label, unit,
    decimals, description and group become Display metadata; locked fields
    drop decimals because they are single-value Literals.
    A core_cfg that is a concrete persistent module or waveform Spec builds the
    same Spec form as module_form and requires an empty tree.
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
        self._section: _Section | _SpecSection = (
            _SpecSection(core_cfg, tree)
            if core_cfg in _PERSISTENT_SPECS
            else _Section(core_cfg, tree, ())
        )
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

        The host's registration entry calls this explicitly. adapter is the
        registered definition name, included with the consuming field path in ValueError for unknown IDs,
        incompatible families or malformed seed trees. A template seed for an
        unsupported module field (optional desc, mask, outsel, ro_ch or mux_*)
        raises the same ValueError, naming the template id and the dotted seed
        path; to_cfg fills those fields only from Spec defaults. Seeds for other
        fields, including $-prefixed strings, stay literal. No sources are read.
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

    def template_input(
        self,
        template: Template,
        catalog: Mapping[str, TemplateDef],
        env: FormEnv,
    ) -> FormInput[FormModel]:
        """Build fresh creation input for this persistent module or waveform Spec.

        template names a catalog ID, with optional qubit/resonator bindings,
        relative-path seed overrides and literal locks. catalog maps IDs to
        TemplateDef values. env supplies captured parameters and library for
        Expr initial values and Slot resolution. Partial seeds overlay Spec
        defaults; failed Expr initial values leave required fields unfilled.
        Bound Slots keep child library keys; unbound Slots use their fallback.
        No native cfg or asset is materialized. to_cfg returns the original Spec.

        input.model may be a subclass of this Form's model, specialized for
        this template's Expr requirements and locks. Build a new FormState from
        the returned input. Recompute the same template's initial values with
        input.model.default_input(env); changing templates requires a new state.
        The original Form.model, template, catalog and sources are not modified.
        The input model retains a detached catalog mapping, not the caller's map.

        Core cfg Forms raise TypeError. Unknown IDs, incompatible Spec families,
        unsupported or malformed seed/lock paths and recursive fallbacks raise
        ValueError naming the template or path. Captured-source and default
        factory failures propagate; they never silently use a fallback.
        """
        if not isinstance(self._section, _SpecSection):
            raise TypeError("template_input requires a persistent Spec form")
        bound = MappingProxyType(dict(catalog))
        _template_input(template, bound, (self.model,), None)
        editable = editable_spec(
            self.model, template.lock, template_seed_tree(template, bound)
        )

        def defaults(
            cls: type[FormModel], env: FormEnv, previous: Mapping[str, JsonValue]
        ) -> dict[str, JsonValue]:
            del previous
            return _template_input(template, bound, (cls,), env)

        model = new_class(
            f"{self.model.__name__}Template",
            (editable,),
            exec_body=lambda ns: ns.update(
                {"__module__": __name__, "defaults": classmethod(defaults)}
            ),
        )
        if not issubclass(model, FormModel):
            raise TypeError("Template input model must implement FormModel")
        return model.default_input(env)

    def input_from_spec(self, spec: ConfigBase, env: FormEnv) -> FormInput[FormModel]:
        """Project an existing persistent Spec into fresh input for this Spec form.

        Only Forms built for a persistent Spec accept this; a core cfg Form raises
        TypeError. spec accepts the shared ConfigBase boundary so Form remains
        covariant in its concrete result type. At runtime spec must be exactly
        core_cfg, not a subclass; otherwise raise
        TypeError. Ordinary values and child library keys are copied, and None
        values stay absent like static defaults. Values of unsupported module
        fields are discarded, so to_cfg restores Spec defaults instead of the
        stored values. env supplies the default input that the projection
        overlays; captured source failures propagate. spec is not modified and
        no library entry, asset or native cfg is read.
        """
        if not isinstance(self._section, _SpecSection):
            raise TypeError("input_from_spec requires a persistent Spec form")
        if type(spec) is not self.core_cfg:
            raise TypeError(
                f"Expected {self.core_cfg.__name__}, got {type(spec).__name__}"
            )
        state = FormState(self.default_input(env))
        dropped = state.write_compatible(
            supported_input(
                self._section.editable, spec.model_dump(mode="json", exclude_none=True)
            ),
            env,
        )
        if dropped:
            raise ValueError(f"Spec values do not fit generated input: {dropped}")
        return state.input

    def to_cfg(self, value: FormModel, env: FormEnv) -> C | FormErrors:
        """Convert this form's ordinary result into a validated concrete core cfg.

        value must be a result of this generated model or its template_input
        subclass; another model raises TypeError. env is the same captured
        source used by result. Core validation
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


def module_form[S: ConfigBase](spec_type: type[S]) -> Form[S]:
    """Build the Form for one concrete persistent module or waveform Spec class.

    spec_type is a member class of ModuleSpec or WaveformSpec, such as PulseSpec
    or DirectReadoutCfg. Union aliases, materialized or native runtime cfgs and
    other classes raise TypeError. The model is a FormModel usable as a
    FormState root or a ModuleInline module field; hosts add choices through
    their own choices(env). to_cfg returns spec_type and keeps child library
    keys without reading the library, assets or building native cfg. Generated
    input omits optional desc, mask, outsel, ro_ch and mux_* fields, whose Spec
    defaults return in to_cfg. Use input_from_spec to edit an existing Spec.
    """
    if spec_type not in _PERSISTENT_SPECS:
        raise TypeError(
            f"Not a concrete persistent module or waveform Spec: {spec_type!r}"
        )
    return Form(spec_type, {})
