"""Family-limited persistent module inputs and native cfg materialization."""

from collections.abc import Mapping
from copy import deepcopy
from types import new_class
from typing import Annotated, TypeAliasType, get_args, get_origin

from pydantic import BaseModel, Field, JsonValue
from qick.asm_v2 import QickParam

from zcu_tools.cfg_model import (
    AbsModuleCfg,
    ConfigBase,
    ModuleCfgFactory,
    ModuleSpec,
    get_field_meta,
)
from zcu_tools.gui.form import Display, FormEnv, ModuleInline, ModuleRef
from zcu_tools.program.v2.modules import AbsModuleCfg as NativeModuleBase
from zcu_tools.program.v2.modules import ModuleCfgFactory as NativeModuleFactory

from ._form_types import annotated_type, members, type_tag, union_type

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


def _editable(annotation: object) -> object:
    if isinstance(annotation, TypeAliasType):
        return _editable(annotation.__value__)
    real = tuple(member for member in members(annotation) if member is not QickParam)
    items: list[object] = []
    for member in real:
        if isinstance(member, type) and issubclass(member, BaseModel):
            items.append(_editable_spec(member))
        else:
            items.append(member)
    result = union_type(tuple(items))
    original_origin = get_origin(annotation)
    original_args = get_args(annotation)
    if original_origin is Annotated:
        for metadata in original_args[1:]:
            result = annotated_type(result, metadata)
    return result


def _editable_spec(model: type[BaseModel]) -> type[BaseModel]:
    annotations: dict[str, object] = {}
    namespace: dict[str, object] = {
        "__module__": __name__,
        "__annotations__": annotations,
    }
    for name, info in model.model_fields.items():
        annotations[name] = _editable(info.annotation)
        derived = deepcopy(info)
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


class ModuleField:
    """A core module field's family-limited ref/inline input and lowering policy."""

    def __init__(self, annotation: object, path: tuple[str, ...]) -> None:
        self.specs = module_specs(annotation)
        self.path = path
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
        module = union_type(tuple(_editable_spec(spec) for spec in self.specs))
        if len(self.specs) > 1:
            module = annotated_type(module, Field(discriminator="type"))
        namespace: dict[str, object] = {
            "__module__": __name__,
            "__annotations__": {"module": module},
        }
        inline = new_class(
            f"{''.join(path)}Inline",
            (ModuleInline,),
            exec_body=lambda ns: ns.update(namespace),
        )
        annotation = annotated_type(
            union_type((ref, inline)), Field(discriminator="kind")
        )
        self.annotation = annotation

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
