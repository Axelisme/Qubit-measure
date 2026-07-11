from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zcu_tools.meta_tool import ModuleLibrary

if TYPE_CHECKING:
    from zcu_tools.gui.cfg import (
        CfgSectionValue,
        ReferenceValue,
    )


@dataclass(frozen=True)
class NamedModuleValue:
    name: str
    value: CfgSectionValue


def select_named_module_value(
    *,
    ml: ModuleLibrary,
    module_type: type[Any],
    preferred_names: list[str],
) -> NamedModuleValue | None:
    from zcu_tools.gui.app.main.cfg_schemas import (
        module_cfg_to_value,  # lazy: avoids circular import
    )

    candidates = {
        name: module
        for name, module in ml.modules.items()
        if isinstance(module, module_type)
    }
    if not candidates:
        return None

    chosen_name: str | None = None
    for preferred_name in preferred_names:
        if preferred_name in candidates:
            chosen_name = preferred_name
            break
    if chosen_name is None:
        return None

    _, value = module_cfg_to_value(candidates[chosen_name])
    return NamedModuleValue(name=chosen_name, value=value)


def select_named_waveform_value(
    ml: ModuleLibrary, preferred_names: list[str]
) -> ReferenceValue | None:
    """Waveform twin of ``select_named_module_value``: first preferred-named
    library waveform → a LINKED ``ReferenceValue``, else None."""
    from zcu_tools.gui.app.main.cfg_schemas import waveform_cfg_to_value
    from zcu_tools.gui.cfg import ReferenceValue

    for name in preferred_names:
        if name in ml.waveforms:
            _, wav_val = waveform_cfg_to_value(ml.waveforms[name])
            return ReferenceValue(chosen_key=name, value=wav_val)
    return None
