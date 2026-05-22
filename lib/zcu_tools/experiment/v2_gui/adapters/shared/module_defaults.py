from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Type

from zcu_tools.gui.adapter import CfgSectionValue, ModuleRefValue, make_default_value
from zcu_tools.gui.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.specs.readout import make_direct_readout_spec
from zcu_tools.meta_tool import ModuleLibrary
from zcu_tools.program.v2 import AbsReadoutCfg


@dataclass(frozen=True)
class NamedModuleValue:
    name: str
    value: CfgSectionValue


def select_named_module_value(
    *,
    ml: ModuleLibrary,
    module_type: Type[Any],
    preferred_names: list[str],
) -> Optional[NamedModuleValue]:
    candidates = {
        name: module
        for name, module in ml.modules.items()
        if isinstance(module, module_type)
    }
    if not candidates:
        return None

    chosen_name: Optional[str] = None
    for preferred_name in preferred_names:
        if preferred_name in candidates:
            chosen_name = preferred_name
            break
    if chosen_name is None:
        chosen_name = next(iter(reversed(list(candidates.keys()))))

    _, value = module_cfg_to_value(candidates[chosen_name])
    return NamedModuleValue(name=chosen_name, value=value)


def infer_module_ref_fallback(
    module_type: Type[Any],
) -> tuple[str, Callable[[], Any]]:
    if issubclass(module_type, AbsReadoutCfg):
        return "<Custom:Direct Readout>", make_direct_readout_spec
    raise RuntimeError(f"Unsupported module_type for fallback inference: {module_type}")


def make_module_ref_default(
    *,
    ml: ModuleLibrary,
    module_type: Type[Any],
    preferred_names: list[str],
    fallback_key: Optional[str] = None,
    fallback_spec_factory: Optional[Callable[[], Any]] = None,
) -> ModuleRefValue:
    selected = select_named_module_value(
        ml=ml,
        module_type=module_type,
        preferred_names=preferred_names,
    )
    if selected is not None:
        return ModuleRefValue(chosen_key=selected.name, value=selected.value)

    if fallback_key is None or fallback_spec_factory is None:
        inferred_key, inferred_factory = infer_module_ref_fallback(module_type)
        fallback_key = fallback_key or inferred_key
        fallback_spec_factory = fallback_spec_factory or inferred_factory

    return ModuleRefValue(
        chosen_key=fallback_key,
        value=make_default_value(fallback_spec_factory()),
    )
