from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Type

from zcu_tools.meta_tool import ModuleLibrary

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import CfgSectionValue, WaveformRefValue


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
    from zcu_tools.gui.cfg_schemas import (
        module_cfg_to_value,  # lazy: avoids circular import
    )

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
        return None

    _, value = module_cfg_to_value(candidates[chosen_name])
    return NamedModuleValue(name=chosen_name, value=value)


def select_named_waveform_value(
    ml: ModuleLibrary, preferred_names: list[str]
) -> Optional[WaveformRefValue]:
    """Waveform twin of ``select_named_module_value``: first preferred-named
    library waveform → a LINKED ``WaveformRefValue``, else None."""
    from zcu_tools.gui.adapter import WaveformRefValue
    from zcu_tools.gui.cfg_schemas import waveform_cfg_to_value

    for name in preferred_names:
        if name in ml.waveforms:
            _, wav_val = waveform_cfg_to_value(ml.waveforms[name])
            return WaveformRefValue(chosen_key=name, value=wav_val)
    return None
