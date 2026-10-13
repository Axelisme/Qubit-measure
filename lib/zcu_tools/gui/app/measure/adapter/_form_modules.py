"""Family-limited persistent module inputs and native cfg materialization."""

from collections.abc import Mapping
from copy import deepcopy
from types import UnionType, new_class
from typing import Annotated, ClassVar, TypeAliasType, Union, get_args, get_origin

from pydantic import BaseModel, Field, JsonValue
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
    get_field_meta,
)
from zcu_tools.gui.form import Display, FormEnv, ModuleInline, ModuleRef
from zcu_tools.program.v2.modules import AbsModuleCfg as NativeModuleBase
from zcu_tools.program.v2.modules import ModuleCfgFactory as NativeModuleFactory

from ._form_seeds import (
    complete_input,
    scalar_seed,
    set_path,
    slot_key,
    template_input,
    template_seed_tree,
)
from ._form_types import annotated_type, literal_type, members, type_tag, union_type

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
        return _editable_spec(annotation, locks, seeds)
    return annotation


def _editable_spec(
    model: type[BaseModel],
    locks: Mapping[str, TemplateLiteral],
    seeds: Mapping[str, JsonValue],
) -> type[BaseModel]:
    annotations: dict[str, object] = {}
    namespace: dict[str, object] = {
        "__module__": __name__,
        "__annotations__": annotations,
    }
    for name, info in model.model_fields.items():
        nested = {
            path.partition(".")[2]: value
            for path, value in locks.items()
            if path.partition(".")[0] == name and "." in path
        }
        seed = seeds.get(name)
        annotations[name] = (
            literal_type(locks[name])
            if name in locks
            else _editable(info.annotation, nested, seed if isinstance(seed, dict) else {})
        )
        derived = deepcopy(info)
        if isinstance(seed, dict) and "expr" in seed:
            derived.default = PydanticUndefined
            derived.default_factory = None
        if name in locks:
            derived.default = locks[name]
        if (meta := get_field_meta(info)) is not None:
            derived.metadata.append(
                Display(label=meta.label, unit=meta.unit, description=meta.description)
            )
        namespace[name] = derived
    result = new_class(
        f"{model.__name__}Input", (model,), exec_body=lambda ns: ns.update(namespace)
    )
    if not issubclass(result, BaseModel):
        raise TypeError("Editable Spec must be a Pydantic model")
    return result


class LockedInline(ModuleInline):
    """Inline branch defaults copy library content, then restore literal locks.

    locks maps relative module leaf paths to immutable authored literals. The
    generated subclass narrows its Spec fields to those same single-value
    Literals. Defaults never reapplies initialization-only set overrides.
    """

    locks: ClassVar[Mapping[str, TemplateLiteral]] = {}

    @classmethod
    def defaults(
        cls, env: FormEnv, previous: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        """Copy a previous reference through ModuleInline, then restore locks."""
        data = super().defaults(env, previous)
        module = data.get("module")
        if isinstance(module, dict):
            for path, value in cls.locks.items():
                set_path(module, path, scalar_seed(value))
        return data


class ModuleField:
    """A core module field's family-limited ref/inline input and lowering policy."""

    def __init__(
        self, annotation: object, path: tuple[str, ...], node: Slot | Template | None
    ) -> None:
        self.specs = module_specs(annotation)
        self.path = path
        self.node = node
        self.catalog: Mapping[str, TemplateDef] = {}
        self.locks = node.lock if node is not None else {}
        self.editable = tuple(_editable_spec(spec, self.locks, {}) for spec in self.specs)
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
        namespace["locks"] = self.locks
        inline = new_class(
            f"{''.join(path)}Inline",
            (LockedInline,),
            exec_body=lambda ns: ns.update(namespace),
        )
        self.inline = inline
        self.annotation = (
            inline
            if self.locks
            else annotated_type(union_type((ref, inline)), Field(discriminator="kind"))
        )

    def validate_templates(self, catalog: Mapping[str, TemplateDef]) -> None:
        template = (
            self.node
            if isinstance(self.node, Template)
            else self.node.fallback
            if self.node is not None
            else None
        )
        if template is not None:
            template_input(template, catalog, self.editable, None)
            seeds = template_seed_tree(template, catalog)
            self.editable = tuple(
                _editable_spec(spec, self.locks, seeds) for spec in self.specs
            )
            module = union_type(self.editable)
            if len(self.specs) > 1:
                module = annotated_type(module, Field(discriminator="type"))
            # Rebuild only this field's generated classes, never the shared Specs.
            self.inline.model_fields["module"].annotation = module
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
            module = spec.model_dump(mode="json", exclude_none=True)
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
