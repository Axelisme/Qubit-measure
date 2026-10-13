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

from pydantic import BaseModel, Field, JsonValue, create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from qick.asm_v2 import QickParam

from zcu_tools.cfg_model import (
    AbsModuleCfg,
    ConfigBase,
    ModuleCfgFactory,
    ModuleSpec,
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
    FormWritePath,
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
    scalar_seed,
    selected_model,
    set_path,
    slot_key,
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
    if str not in members(annotation):
        return ()
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


def _spec_options(
    model: type[BaseModel], env: FormEnv
) -> Mapping[tuple[str | int, ...], tuple[Option, ...]]:
    declared: dict[tuple[str | int, ...], tuple[Option, ...]] = {}
    for name, info in model.model_fields.items():
        specs = _key_specs(info.annotation)
        if not specs:
            continue
        options = (
            [Option("None", None)] if type(None) in members(info.annotation) else []
        )
        if env.library is not None:
            waveform = all(spec in members(WaveformSpec) for spec in specs)
            keys = (
                env.library.waveform_keys() if waveform else env.library.module_keys()
            )
            read = env.library.waveform_spec if waveform else env.library.module_spec
            options.extend(
                Option(
                    key,
                    key,
                    commands=(
                        OptionCommand(
                            "to_inline",
                            "Convert to inline",
                            partial(
                                _copy_spec,
                                key,
                                _inline_kinds(info.annotation),
                                waveform=waveform,
                            ),
                        ),
                    ),
                )
                for key in keys
                if isinstance(read(key), specs)
            )
        kinds = _inline_kinds(info.annotation)
        for kind in kinds:
            tag, value = _spec_tag(kind)
            options.append(
                Option(
                    "Inline" if len(kinds) == 1 else f"Inline: {value}",
                    {tag: value},
                    fields=kind,
                )
            )
        declared[(name,)] = tuple(options)
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
    return supported_input(model, spec.model_dump(mode="json", exclude_none=True))


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


def editable_spec(
    model: type[BaseModel],
    locks: Mapping[str, TemplateLiteral],
    seeds: Mapping[str, JsonValue],
) -> type[BaseModel]:
    """Derive the input subclass of a persistent Spec and its nested models.

    locks maps relative leaf paths to single-value Literal fields with that
    default. seeds marks Expr initial values, whose fields lose static defaults.
    QickParam alternatives are removed. FieldMeta becomes Display metadata.
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
        if isinstance(seed, dict) and "expr" in seed:
            derived.default = PydanticUndefined
            derived.default_factory = None
        if name in locks:
            derived.default = locks[name]
        if (display := field_display(info, locked=name in locks)) is not None:
            derived.metadata.append(display)
        namespace[name] = derived

    def options(
        cls: type[FormModel], env: FormEnv
    ) -> Mapping[tuple[str | int, ...], tuple[Option, ...]]:
        return _spec_options(cls, env)

    namespace["options"] = classmethod(options)
    bases = (model,) if issubclass(model, FormModel) else (FormModel, model)
    result = new_class(
        f"{model.__name__}Input", bases, exec_body=lambda ns: ns.update(namespace)
    )
    if not issubclass(result, BaseModel):
        raise TypeError("Editable Spec must be a Pydantic model")
    return result


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
        self, annotation: object, path: tuple[str, ...], node: Slot | Template | None
    ) -> None:
        self.specs = module_specs(annotation)
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
                    if isinstance(env.library.module_spec(key), field.specs)
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
        module = union_type(self.editable)
        if len(self.specs) > 1:
            module = annotated_type(module, Field(discriminator="type"))
        namespace: dict[str, object] = {
            "__module__": __name__,
            "__annotations__": {"module": module},
        }

        def defaults(
            cls: type[ModuleInline], env: FormEnv, previous: Mapping[str, JsonValue]
        ) -> dict[str, JsonValue]:
            del cls, previous
            return field.inline_input({}, env)

        namespace["defaults"] = classmethod(defaults)
        inline = new_class(
            f"{''.join(path)}Inline",
            (ModuleInline,),
            exec_body=lambda ns: ns.update(namespace),
        )
        if not issubclass(inline, ModuleInline):
            raise TypeError("Generated inline model must implement ModuleInline")
        self.inline = inline
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
        Fields describe the selected inline wrapper and omit its structural tags.
        Empty kinds have no editable content and do not get an Inline selection.
        """
        options = [Option("None", None)] if self.nullable else []
        if not self.locks and env.library is not None:
            options.extend(
                Option(
                    key,
                    {"kind": "ref", "name": key},
                    commands=(
                        OptionCommand(
                            "to_inline",
                            "Convert to inline",
                            partial(self.copy_input, key),
                        ),
                    ),
                )
                for key in env.library.module_keys()
                if isinstance(env.library.module_spec(key), self.specs)
            )
        kinds = _inline_kinds(union_type(self.editable))
        for kind in kinds:
            tag, value = _spec_tag(kind)
            fields = create_model(
                f"{kind.__name__}InlineFields", __base__=self.inline, module=(kind, ...)
            )
            options.append(
                Option(
                    "Inline" if len(kinds) == 1 else f"Inline: {value}",
                    {"kind": "inline", "module": {tag: value}},
                    fields=fields,
                )
            )
        return tuple(options)

    def copy_input(self, key: str, env: FormEnv) -> dict[str, JsonValue]:
        """Copy an allowed captured key into inline input, without prior draft data.

        Child keys remain references. Unsupported fields are discarded and locks
        restored. Missing or incompatible sources fail before any write.
        """
        module = _copy_spec(key, self.editable, env, waveform=False)
        for path, value in self.locks.items():
            set_path(module, path, scalar_seed(value))
        return {"kind": "inline", "module": module}

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

    def initial_input(self, env: FormEnv) -> dict[str, JsonValue] | None:
        if isinstance(self.node, Template):
            return {
                "kind": "inline",
                "module": template_input(self.node, self.catalog, self.editable, env),
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
            module = supported_input(
                self._editable_for(spec.type),
                spec.model_dump(mode="json", exclude_none=True),
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
        return {"kind": "inline", "module": complete_input(selected, module)}

    def _editable_for(self, tag: str) -> type[BaseModel]:
        return next(model for model in self.editable if type_tag(model) == tag)

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
            else item.module.model_dump(mode="python")
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

    def inline_input(
        self, previous: Mapping[str, JsonValue], env: FormEnv
    ) -> dict[str, JsonValue]:
        """Build detached inline input, copying captured refs or authored fallback.

        Already-inline input is retained. Missing captured library/key uses the
        Slot Template fallback, or static blank when absent. Wrong families and
        source read failures propagate; copied and fresh input restores locks.
        """
        if previous.get("kind") == "inline":
            return deepcopy(dict(previous))
        name = previous.get("name")
        if (
            previous.get("kind") == "ref"
            and isinstance(name, str)
            and env.library is not None
            and name in env.library.module_keys()
        ):
            return self.copy_input(name, env)
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
        return {"kind": "inline", "module": module}

    def inline_writes(
        self, previous: Mapping[str, JsonValue], env: FormEnv
    ) -> dict[FormWritePath, JsonValue]:
        """Prepare one model-aware leaf batch for nullable or required modules."""
        candidate = self.inline_input(previous, env)
        if self.nullable:
            return {self.path: candidate}
        return _input_writes(self.inline, candidate, self.path)

    def materialize(self, item: BaseModel, env: FormEnv) -> ConfigBase:
        if isinstance(item, ModuleRef):
            raw = item.name
        elif isinstance(item, ModuleInline):
            raw = item.module.model_dump(mode="python")
        else:
            raise TypeError("Expected a module reference or inline input")
        context = env.library.validation_context() if env.library is not None else None
        cfg = ModuleCfgFactory.from_raw(raw, context=context)
        return (
            NativeModuleFactory.from_core(cfg, context=context) if self.native else cfg
        )


def _input_writes(
    model: type[BaseModel], data: Mapping[str, JsonValue], path: tuple[str, ...]
) -> dict[FormWritePath, JsonValue]:
    """Project prepared model input to public leaf writes, preserving containers."""
    batch: dict[FormWritePath, JsonValue] = {}
    for name, value in data.items():
        annotation = model.model_fields[name].annotation
        if isinstance(value, dict) and any(
            isinstance(member, type) and issubclass(member, BaseModel)
            for member in members(annotation)
        ):
            batch.update(
                _input_writes(selected_model(annotation, value), value, (*path, name))
            )
        else:
            batch[(*path, name)] = value
    return batch
