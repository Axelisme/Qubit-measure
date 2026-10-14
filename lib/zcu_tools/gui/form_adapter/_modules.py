"""Family-limited persistent module inputs and native cfg materialization."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from types import UnionType, new_class
from typing import (
    Annotated,
    ClassVar,
    Literal,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, Field, JsonValue, create_model, model_validator
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from qick.asm_v2 import QickParam

from zcu_tools.cfg_model import (
    AbsModuleCfg,
    ConfigBase,
    ModuleCfgFactory,
    ModuleSpec,
    PulseReadoutSpec,
    Slot,
    Template,
    TemplateDef,
    TemplateLiteral,
    WaveformSpec,
)
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormState,
    KeyRef,
    ModuleInline,
    ModuleRef,
    Option,
    OptionCommand,
    static_defaults,
)
from zcu_tools.program.v2.modules import AbsModuleCfg as NativeModuleBase
from zcu_tools.program.v2.modules import ModuleCfgFactory as NativeModuleFactory

from ._display import field_display
from ._seeds import (
    complete_input,
    inline_wrapper,
    persistent_input,
    scalar_seed,
    selected_model,
    set_path,
    slot_key,
    spec_annotation,
    supported_input,
    template_input,
    template_seed_tree,
)
from ._types import annotated_type, literal_type, members, type_tag, union_type

_SPEC_BY_TAG = {
    type_tag(model): model
    for model in members(ModuleSpec)
    if isinstance(model, type) and issubclass(model, ConfigBase)
}


def module_specs(annotation: object) -> tuple[type[ConfigBase], ...]:
    real = tuple(member for member in members(annotation) if member is not type(None))
    if not real or not all(
        isinstance(member, type)
        and issubclass(member, (AbsModuleCfg, NativeModuleBase))
        for member in real
    ):
        return ()
    specs: list[type[ConfigBase]] = []
    for model in real:
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise TypeError("Module member must be a Pydantic model")
        tag = type_tag(model)
        if tag not in _SPEC_BY_TAG:
            raise ValueError(f"No persistent Spec for module {model.__name__}")
        specs.append(_SPEC_BY_TAG[tag])
    return tuple(specs)


_PERSISTENT_SPECS = tuple(
    model
    for model in (*members(ModuleSpec), *members(WaveformSpec))
    if isinstance(model, type) and issubclass(model, ConfigBase)
)


def _key_specs(annotation: object) -> tuple[type[ConfigBase], ...]:
    wrapper = inline_wrapper(annotation)
    if wrapper is None:
        return ()
    annotation = spec_annotation(annotation)
    return tuple(
        spec
        for spec in _PERSISTENT_SPECS
        if any(
            isinstance(member, type) and issubclass(member, spec)
            for member in members(annotation)
        )
    )


def _spec_tag(model: type[BaseModel]) -> tuple[str, str]:
    for name in ("type", "style"):
        info = model.model_fields.get(name)
        if info is not None and get_origin(info.annotation) is Literal:
            tags = get_args(info.annotation)
            if len(tags) == 1 and isinstance(tags[0], str):
                return name, tags[0]
    raise ValueError(f"Persistent Spec needs a unique type/style tag: {model.__name__}")


def _inline_kinds(annotation: object) -> tuple[type[BaseModel], ...]:
    return tuple(
        model
        for model in members(annotation)
        if isinstance(model, type)
        and issubclass(model, BaseModel)
        and model.model_fields.keys() - {"type", "style"}
    )


def _module_options(
    keys: tuple[tuple[str, OptionCommand], ...],
    kinds: list[tuple[type[BaseModel], type[BaseModel]]],
    *,
    nullable: bool,
) -> tuple[Option, ...]:
    """Format ordered None/ref/inline selections from caller-owned fields/commands."""
    options = [Option("None", None)] if nullable else []
    options.extend(
        Option(key, {"kind": "ref", "name": key}, commands=(command,))
        for key, command in keys
    )
    for kind, fields in kinds:
        tag, value = _spec_tag(kind)
        options.append(
            Option(
                "Inline" if len(kinds) == 1 else f"Inline: {value}",
                {"kind": "inline", "module": {tag: value}},
                fields=fields,
            )
        )
    return tuple(options)


def _spec_options(
    model: type[BaseModel],
    env: FormEnv,
    inline_fields: dict[tuple[type[BaseModel], type[BaseModel]], type[BaseModel]],
) -> Mapping[tuple[str | int, ...], tuple[Option, ...]]:
    declared: dict[tuple[str | int, ...], tuple[Option, ...]] = {}
    for name, info in model.model_fields.items():
        specs = _key_specs(info.annotation)
        if not specs:
            continue
        key_commands: tuple[tuple[str, OptionCommand], ...] = ()
        if env.library is not None and any(
            isinstance(member, type) and issubclass(member, KeyRef)
            for member in members(info.annotation)
        ):
            waveform = all(spec in members(WaveformSpec) for spec in specs)
            keys = (
                env.library.waveform_keys() if waveform else env.library.module_keys()
            )
            read = env.library.waveform_spec if waveform else env.library.module_spec
            key_commands = tuple(
                (
                    key,
                    OptionCommand(
                        "to_inline",
                        "Convert to inline",
                        partial(
                            _copy_spec,
                            key,
                            _inline_kinds(spec_annotation(info.annotation)),
                            waveform=waveform,
                        ),
                    ),
                )
                for key in keys
                if isinstance(read(key), specs)
            )
        kinds = _inline_kinds(spec_annotation(info.annotation))
        wrapper = inline_wrapper(info.annotation)
        if wrapper is None:
            raise TypeError("Spec options require a tagged inline wrapper")
        kind_fields: list[tuple[type[BaseModel], type[BaseModel]]] = []
        for kind in kinds:
            key = (wrapper, kind)
            fields = inline_fields.get(key)
            if fields is None:
                fields = create_model(
                    f"{kind.__name__}Fields", __base__=wrapper, module=(kind, ...)
                )
                inline_fields[key] = fields
            kind_fields.append((kind, fields))
        declared[(name,)] = _module_options(
            key_commands, kind_fields, nullable=type(None) in members(info.annotation)
        )
    return declared


def _copy_spec(
    key: str, models: tuple[type[BaseModel], ...], env: FormEnv, *, waveform: bool
) -> dict[str, JsonValue]:
    """Copy one captured persistent key into supported detached input.

    models are the permitted generated editable branches. No materialization or
    asset read occurs; child keys stay keys. Missing capture/key, wrong family
    and source faults propagate before the caller can publish a write.
    """
    if env.library is None:
        raise ValueError("Missing captured library for inline conversion")
    spec = env.library.waveform_spec(key) if waveform else env.library.module_spec(key)
    model = next((model for model in models if issubclass(model, type(spec))), None)
    if model is None:
        raise ValueError(f"Incompatible library family for {key!r}")
    return {
        "kind": "inline",
        "module": supported_input(
            model, spec.model_dump(mode="json", exclude_none=True)
        ),
    }


_UNSUPPORTED_FIELDS = frozenset({"desc", "mask", "outsel", "ro_ch"})


def _unsupported(name: str, info: FieldInfo) -> bool:
    """Optional hardware-detail fields stay out of generated module input.

    Spec defaults restore them when input is validated as the Spec. A required
    field, such as DirectReadoutCfg.ro_ch, has no default to restore and stays.
    """
    return not info.is_required() and (
        name in _UNSUPPORTED_FIELDS or name.startswith("mux_")
    )


def _editable(
    annotation: object,
    locks: Mapping[str, TemplateLiteral],
    seeds: Mapping[str, JsonValue],
) -> object:
    wrapper = inline_wrapper(annotation)
    if wrapper is not None or (
        str in members(annotation)
        and any(
            isinstance(member, type) and issubclass(member, BaseModel)
            for member in members(annotation)
        )
    ):
        return _tagged_annotation(annotation, locks, seeds)
    if isinstance(annotation, TypeAliasType):
        return _editable(annotation.__value__, locks, seeds)
    args = get_args(annotation)
    origin = get_origin(annotation)
    if origin is Annotated:
        result = _editable(args[0], locks, seeds)
        for metadata in args[1:]:
            result = annotated_type(result, metadata)
        return result
    if origin in (Union, UnionType):
        items = tuple(
            _editable(member, locks, seeds)
            for member in args
            if member is not QickParam
            and (
                not locks
                or any(
                    isinstance(model, type) and issubclass(model, BaseModel)
                    for model in members(member)
                )
            )
        )
        return union_type(items)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return editable_spec(annotation, locks, seeds)
    return annotation


def _tagged_annotation(
    annotation: object,
    locks: Mapping[str, TemplateLiteral],
    seeds: Mapping[str, JsonValue],
) -> object:
    """Derive one family-limited tagged selector, retaining nullable/locked shape.

    Re-derivation for a template unwraps the existing input annotation first;
    the persistent seed and lock paths do not contain wrapper segments.
    """
    models = tuple(
        editable_spec(member, locks, seeds)
        for member in members(spec_annotation(annotation))
        if isinstance(member, type) and issubclass(member, BaseModel)
    )
    if not models:
        raise TypeError("Tagged inline input needs a persistent child model")
    first = next(iter(models))
    module = union_type(models)
    if len(models) > 1:
        module = annotated_type(module, Field(discriminator=_spec_tag(first)[0]))
    inline = create_model(
        f"{first.__name__}Inline",
        __base__=FormModel,
        kind=(Literal["inline"], "inline"),
        module=(module, ...),
    )
    tagged = (
        inline
        if locks
        else annotated_type(union_type((KeyRef, inline)), Field(discriminator="kind"))
    )
    return (
        union_type((tagged, type(None)))
        if type(None) in members(annotation)
        else tagged
    )


def editable_spec(
    model: type[BaseModel],
    locks: Mapping[str, TemplateLiteral],
    seeds: Mapping[str, JsonValue],
) -> type[BaseModel]:
    """Derive the input subclass of a persistent Spec and its nested models.

    locks maps relative leaf paths to single-value Literal fields with that
    default. seeds marks Expr initial values, whose fields lose static defaults.
    QickParam alternatives are removed. FieldMeta becomes Display metadata.
    str-or-model children become tagged ref/name or inline/module wrappers;
    persistent seed and lock paths still name Spec fields, without envelopes.
    Optional desc, mask, outsel, ro_ch and mux_* fields are removed, so input,
    schema and writes cannot set them; validating as the Spec restores defaults.
    The Spec's validators are inherited; the Spec class is not modified.
    """
    annotations: dict[str, object] = {}
    namespace: dict[str, object] = {
        "__module__": __name__,
        "__annotations__": annotations,
    }
    for name, info in model.model_fields.items():
        if _unsupported(name, info):
            # A ClassVar annotation removes the inherited Pydantic field.
            annotations[name] = ClassVar[None]
            continue
        nested = {
            path.partition(".")[2]: value
            for path, value in locks.items()
            if path.partition(".")[0] == name and "." in path
        }
        seed = seeds.get(name)
        annotations[name] = (
            literal_type(locks[name])
            if name in locks
            else _editable(
                info.annotation, nested, seed if isinstance(seed, dict) else {}
            )
        )
        derived = deepcopy(info)
        if inline_wrapper(annotations[name]) is not None:
            derived.discriminator = (
                "kind"
                if any(
                    isinstance(member, type) and issubclass(member, KeyRef)
                    for member in members(annotations[name])
                )
                else None
            )
        if isinstance(seed, dict) and "expr" in seed:
            derived.default = PydanticUndefined
            derived.default_factory = None
        if name in locks:
            derived.default = locks[name]
        if (display := field_display(info, locked=name in locks)) is not None:
            derived.metadata.append(display)
        namespace[name] = derived

    # Each generated model owns its classes; option values and sources stay fresh.
    inline_fields: dict[tuple[type[BaseModel], type[BaseModel]], type[BaseModel]] = {}

    def options(
        cls: type[FormModel], env: FormEnv
    ) -> Mapping[tuple[str | int, ...], tuple[Option, ...]]:
        return _spec_options(cls, env, inline_fields)

    def defaults(cls: type[FormModel], env: FormEnv) -> dict[str, JsonValue]:
        del env
        return supported_input(cls, complete_input(cls, {}))

    namespace["options"] = classmethod(options)
    namespace["defaults"] = classmethod(defaults)
    if issubclass(model, PulseReadoutSpec):
        namespace["_autoderive_ro_fields"] = model_validator(mode="before")(
            _derive_readout_input
        )
    bases = (model,) if issubclass(model, FormModel) else (FormModel, model)
    result = new_class(
        f"{model.__name__}Input", bases, exec_body=lambda ns: ns.update(namespace)
    )
    if not issubclass(result, BaseModel):
        raise TypeError("Editable Spec must be a Pydantic model")
    return result


def _derive_readout_input(cls: type[BaseModel], data: object) -> object:
    """Reuse readout owner derivation while retaining malformed generated leaves."""
    if not isinstance(data, dict):
        return data
    raw = persistent_input(cls, data)
    derived = PulseReadoutSpec.derive_input_fields(raw)
    # The owner only changes ro_cfg, whose persistent/input shapes are identical.
    # Overlay those fields without the supported-field filter: invalid extras
    # and unchanged malformed wrappers must still reach Pydantic for errors.
    changed = {
        name: value
        for name, value in derived.items()
        if name not in raw or value != raw[name]
    }
    return {**data, **changed}


def _template_subtree(
    data: dict[str, JsonValue], ancestry: tuple[tuple[str, str, str], ...]
) -> dict[str, JsonValue] | None:
    """Select an authored position only when every persistent branch tag matches."""
    for name, tag, value in ancestry:
        if name:
            child = data.get(name)
            if not isinstance(child, dict):
                return None
            data = child
        if tag and data.get(tag) != value:
            return None
    return data


def bind_template_defaults(
    models: tuple[type[BaseModel], ...],
    template: Template,
    catalog: Mapping[str, TemplateDef],
) -> None:
    """Bind existing defaults on this template's dedicated generated model tree.

    Each branch uses fresh template input only at the same relative path and
    matching ancestry tags; other branches use their static type defaults.
    Models must be freshly derived input types, never shared persistent Specs.
    The existing template resolver owns Slot presence and captured-source errors.
    """

    def bind(
        model: type[BaseModel],
        ancestry: tuple[tuple[str, str, str], ...],
        *,
        wrapper: bool = False,
    ) -> None:
        if not issubclass(model, FormModel):
            raise TypeError("Template defaults require generated FormModel types")

        def defaults(cls: type[FormModel], env: FormEnv) -> dict[str, JsonValue]:
            data = template_input(template, catalog, models, env)
            subtree = _template_subtree(data, ancestry)
            if subtree is None:
                return static_defaults(cls)
            data = subtree
            if wrapper:
                content = cls.model_fields["module"].annotation
                child = selected_model(content, data)
                return {
                    "kind": "inline",
                    "module": supported_input(child, complete_input(child, data)),
                }
            return supported_input(cls, complete_input(cls, data))

        # Internal kind switches call the child defaults, not the root wrapper.
        # Bind the same path/branch policy on each separately generated child.
        # Install the descriptor, not a bound method; typed assignment expects the latter.
        type.__setattr__(model, "defaults", classmethod(defaults))
        for name, info in () if wrapper else model.model_fields.items():
            inline = inline_wrapper(info.annotation)
            if inline is not None:
                bind(inline, (*ancestry, (name, "", "")), wrapper=True)
            for child in _inline_kinds(spec_annotation(info.annotation)):
                tag, value = _spec_tag(child)
                bind(child, (*ancestry, (name, tag, value)))

    for model in models:
        tag, value = _spec_tag(model)
        bind(model, (("", tag, value),))


@dataclass(frozen=True)
class ModuleValues:
    """Detached execution module and its authored input origin.

    value contains this field's materialized module JSON, not the full native
    cfg after to_cfg or extras lowering. mode is key or inline, not a mode for
    its descendants.
    source is library:<actual-key> for key mode, otherwise the dotted Form path.
    No container aliases input or captured library; this is not a preview.
    """

    value: dict[str, JsonValue]
    mode: Literal["key", "inline"]
    source: str


class ModuleValueForm(FormModel):
    """Typed single-field evaluation shell for generated family-limited input."""

    value: ModuleRef | ModuleInline


class ModuleField:
    """A core module field's family-limited ref/inline input and lowering policy."""

    def __init__(
        self,
        annotation: object,
        path: tuple[str, ...],
        node: Slot | Template | None,
        *,
        specs: tuple[type[ConfigBase], ...] | None = None,
        ref_specs: tuple[type[ConfigBase], ...] | None = None,
    ) -> None:
        self.specs = module_specs(annotation) if specs is None else specs
        self.ref_specs = self.specs if ref_specs is None else ref_specs
        self.path = path
        self.node = node
        self.nullable = type(None) in members(annotation)
        self.catalog: Mapping[str, TemplateDef] = {}
        self.locks = node.lock if node is not None else {}
        self.editable = tuple(
            editable_spec(spec, self.locks, {}) for spec in self.specs
        )
        self.native = any(
            isinstance(member, type) and issubclass(member, NativeModuleBase)
            for member in members(annotation)
        )
        field = self

        def choices(
            cls: type[ModuleRef], env: FormEnv
        ) -> Mapping[tuple[str | int, ...], tuple[JsonValue, ...]]:
            del cls
            keys: tuple[JsonValue, ...] = (
                ()
                if env.library is None
                else tuple(
                    key
                    for key in env.library.module_keys()
                    if isinstance(env.library.module_spec(key), field.ref_specs)
                )
            )
            return {("name",): keys}

        ref = new_class(
            f"{''.join(path)}Ref",
            (ModuleRef,),
            exec_body=lambda ns: ns.update(
                {"__module__": __name__, "choices": classmethod(choices)}
            ),
        )
        if not issubclass(ref, ModuleRef):
            raise TypeError("Generated reference model must implement ModuleRef")
        self.ref = ref
        module = union_type(self.editable)
        if len(self.specs) > 1:
            module = annotated_type(module, Field(discriminator="type"))
        namespace: dict[str, object] = {
            "__module__": __name__,
            "__annotations__": {"module": module},
        }

        def defaults(cls: type[ModuleInline], env: FormEnv) -> dict[str, JsonValue]:
            del cls
            return field.fresh_inline_input(env)

        namespace["defaults"] = classmethod(defaults)
        inline = new_class(
            f"{''.join(path)}Inline",
            (ModuleInline,),
            exec_body=lambda ns: ns.update(namespace),
        )
        if not issubclass(inline, ModuleInline):
            raise TypeError("Generated inline model must implement ModuleInline")
        self.inline = inline
        self._inline_fields: dict[type[BaseModel], type[BaseModel]] = {}
        self.annotation = (
            inline
            if self.locks
            else annotated_type(union_type((ref, inline)), Field(discriminator="kind"))
        )
        # A single-field form evaluates this module without unrelated draft errors.
        value_model = new_class(
            f"{''.join(path)}ValueForm",
            (ModuleValueForm,),
            exec_body=lambda ns: ns.update(
                {
                    "__module__": __name__,
                    "__annotations__": {"value": self.annotation},
                }
            ),
        )
        if not issubclass(value_model, ModuleValueForm):
            raise TypeError("Module value model must implement FormModel")
        self.value_model = value_model

    def options(self, env: FormEnv) -> tuple[Option, ...]:
        """Declare nullable/key/kind selections in the existing module input shape.

        Keys are limited to this field's persistent families in the fixed env.
        Inline values select only kind tags; core supplies fresh branch defaults.
        Fields reuse a class per inline wrapper/kind for this field's lifetime,
        describe the selected inline wrapper and omit its structural tags.
        Empty kinds have no editable content and do not get an Inline selection.
        """
        key_commands: tuple[tuple[str, OptionCommand], ...] = ()
        if not self.locks and env.library is not None:
            key_commands = tuple(
                (
                    key,
                    OptionCommand(
                        "to_inline",
                        "Convert to inline",
                        partial(_copy_spec, key, self.editable, waveform=False),
                    ),
                )
                for key in env.library.module_keys()
                if isinstance(env.library.module_spec(key), self.ref_specs)
            )
        kind_fields: list[tuple[type[BaseModel], type[BaseModel]]] = []
        for kind in _inline_kinds(union_type(self.editable)):
            fields = self._inline_fields.get(kind)
            if fields is None:
                fields = create_model(
                    f"{kind.__name__}InlineFields",
                    __base__=self.inline,
                    module=(kind, ...),
                )
                self._inline_fields[kind] = fields
            kind_fields.append((kind, fields))
        return _module_options(key_commands, kind_fields, nullable=self.nullable)

    def validate_templates(self, catalog: Mapping[str, TemplateDef]) -> None:
        template = (
            self.node
            if isinstance(self.node, Template)
            else self.node.fallback
            if self.node is not None
            else None
        )
        seeds: dict[str, JsonValue] = {}
        if template is not None:
            template_input(template, catalog, self.editable, None)
            seeds = template_seed_tree(template, catalog)
        if isinstance(self.node, Slot):
            for path, value in self.node.set.items():
                set_path(seeds, path, scalar_seed(value))
            for path, value in self.locks.items():
                set_path(seeds, path, scalar_seed(value))
        if seeds:
            self.editable = tuple(
                editable_spec(spec, self.locks, seeds) for spec in self.specs
            )
            module = union_type(self.editable)
            if len(self.specs) > 1:
                module = annotated_type(module, Field(discriminator="type"))
            # Parse the annotation without mutating FieldInfo.annotation; keep the
            # inline class identity borrowed by the section and value form.
            parsed = create_model(
                self.inline.__name__,
                __base__=self.inline,
                __module__=__name__,
                module=(module, deepcopy(self.inline.model_fields["module"])),
            )
            self.inline.model_fields["module"] = parsed.model_fields["module"]
            self.inline.model_rebuild(force=True)
        if template is not None:
            bind_template_defaults(self.editable, template, catalog)

    def initial_input(self, env: FormEnv) -> dict[str, JsonValue] | None:
        if isinstance(self.node, Template):
            return {
                "kind": "inline",
                "module": self._project(
                    template_input(self.node, self.catalog, self.editable, env)
                ),
            }
        if self.node is None:
            return None
        key = slot_key(self.node, env)
        if key is not None and env.library is not None:
            spec = env.library.module_spec(key)
            if not isinstance(spec, self.specs):
                raise ValueError(
                    f"Slot {self.node.path!r} has incompatible module family"
                )
            if not self.node.copy and not self.locks:
                return {"kind": "ref", "name": key}
            model = next(
                model for model in self.editable if type_tag(model) == spec.type
            )
            module = persistent_input(
                model,
                supported_input(model, spec.model_dump(mode="json", exclude_none=True)),
            )
        elif self.node.fallback is not None:
            module = template_input(
                self.node.fallback, self.catalog, self.editable, env
            )
        else:
            return None
        for path, value in self.node.set.items():
            set_path(module, path, scalar_seed(value))
        for path, value in self.locks.items():
            set_path(module, path, scalar_seed(value))
        selected = next(
            model for model in self.editable if type_tag(model) == module.get("type")
        )
        return {
            "kind": "inline",
            "module": supported_input(selected, complete_input(selected, module)),
        }

    def _project(self, module: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        return supported_input(
            selected_model(union_type(self.editable), module), module
        )

    def read(self, raw: JsonValue, env: FormEnv) -> ModuleValues | FormErrors:
        """Evaluate only this module, then materialize against its fixed capture.

        Failures return directed errors at this field, not unrelated cfg errors.
        No full-cfg or extras lower hook runs; source and family checks are the
        same ones used for ordinary Form module evaluation and materialization.
        """
        result = FormState(FormInput(self.value_model, {"value": raw})).result(env)
        if isinstance(result, FormErrors):
            return FormErrors(
                {
                    (*self.path, *path[1:]): message
                    for path, message in result.messages.items()
                }
            )
        item = result.value
        raw_module = (
            item.name
            if isinstance(item, ModuleRef)
            else persistent_input(
                type(item.module), item.module.model_dump(mode="json")
            )
        )
        context = env.library.validation_context() if env.library is not None else None
        try:
            cfg = ModuleCfgFactory.from_raw(raw_module, context=context)
        except ValueError as exc:
            return FormErrors({self.path: str(exc)})
        mode = "key" if isinstance(item, ModuleRef) else "inline"
        source = (
            f"library:{item.name}"
            if isinstance(item, ModuleRef)
            else ".".join(self.path)
        )
        return ModuleValues(cfg.model_dump(mode="json"), mode, source)

    def fresh_inline_input(self, env: FormEnv) -> dict[str, JsonValue]:
        """Build fresh Template/blank input and locks without copying any prior ref."""
        template = (
            self.node
            if isinstance(self.node, Template)
            else self.node.fallback
            if self.node is not None
            else None
        )
        if template is not None:
            module = template_input(template, self.catalog, self.editable, env)
        elif len(self.editable) == 1:
            module = complete_input(self.editable[0], {})
        else:
            return static_defaults(self.inline)
        for path, value in self.locks.items():
            set_path(module, path, scalar_seed(value))
        return {"kind": "inline", "module": self._project(module)}

    def materialize(self, item: BaseModel, env: FormEnv) -> ConfigBase:
        if isinstance(item, ModuleRef):
            raw = item.name
        elif isinstance(item, ModuleInline):
            raw = persistent_input(
                type(item.module), item.module.model_dump(mode="json")
            )
        else:
            raise TypeError("Expected a module reference or inline input")
        context = env.library.validation_context() if env.library is not None else None
        cfg = ModuleCfgFactory.from_raw(raw, context=context)
        return (
            NativeModuleFactory.from_core(cfg, context=context) if self.native else cfg
        )
