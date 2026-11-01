from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

from pydantic import ValidationInfo
from qick.asm_v2 import QickParam

from zcu_tools.cfg_model import ConfigBase

if TYPE_CHECKING:
    from zcu_tools.program.v2.timeline import TimelineProgramV2
    from zcu_tools.resources.context import LibrarySnapshot


def resolve_module_ref(value: Any, info: ValidationInfo) -> Any:
    if isinstance(value, str):
        if info.context is None:
            raise ValueError("ModuleLibrary context not found")
        library: LibrarySnapshot = info.context["ml"]
        return library.get_module(value)
    return value


class AbsModuleCfg(ConfigBase, ABC):
    type: str
    desc: str | None = None

    @abstractmethod
    def build(self, name: str) -> Module:
        """Build the runtime module represented by this cfg."""

    @abstractmethod
    def set_param(self, name: str, value: float | QickParam) -> None:
        """Update a sweep-tunable parameter by name."""


def apply_cfg_overrides(
    cfg: AbsModuleCfg | None,
    overrides: Mapping[str, object],
    allowed: Collection[str],
) -> None:
    """Apply allowed constructor knobs in place to an already detached cfg.

    overrides maps knob names to scalar numbers, QICK sweeps or None, which
    keeps cfg. allowed lists the selected constructor TypedDict keys. Pulse,
    readout and reset constructors share this runtime guard because Python
    callers can bypass their static keyword contract. Unknown keys, including
    keys with None values, and nonnumeric values raise TypeError. Non-None
    values with cfg=None raise ValueError. cfg.set_param errors propagate.
    """
    unknown = overrides.keys() - allowed
    if unknown:
        raise TypeError(f"Unknown override keywords: {sorted(unknown)}")
    for knob, value in overrides.items():
        if value is None:
            continue
        if cfg is None:
            raise ValueError("overrides require a module cfg")
        if not isinstance(value, (float, int, QickParam)):
            raise TypeError(f"{knob}: override must be numeric, a QICK sweep or None")
        cfg.set_param(knob, value)


class Module(ABC):
    name: str

    def declare(self, prog: TimelineProgramV2) -> None:
        """Declare resources on prog before emission.

        Only modules with a timeline lowering implement this interface.
        Unsupported module types raise NotImplementedError without emission.
        """
        raise NotImplementedError(f"{type(self).__name__} has no timeline declaration")

    def lower(
        self, prog: TimelineProgramV2, *, timeline: str | None, tag: str | None
    ) -> None:
        """Emit native macros and advance by this owner's blocking duration.

        Called by prog.emit after declaration. None selects the default timeline;
        other names require fork in _body. Events share that timeline's current
        end and the owner adds rule-a duration without moving reference.
        DelayAuto instead advances native reference and rejects explicit names.
        tag belongs to this emission, not the declaration. Unsupported modules
        raise NotImplementedError.
        """
        raise NotImplementedError(f"{type(self).__name__} has no timeline lowering")


ModuleGroup: TypeAlias = Module | Sequence[Module | None] | None


def as_modules(value: ModuleGroup) -> list[Module]:
    """Return an ordered module list; None means no module and is skipped.

    value is one module, one sequence of optional modules, or None. The returned
    list is new; the module objects are not copied.
    """
    if value is None:
        return []
    if isinstance(value, Module):
        return [value]
    return [module for module in value if module is not None]
