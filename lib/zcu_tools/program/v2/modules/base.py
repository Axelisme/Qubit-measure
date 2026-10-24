from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

from pydantic import ValidationInfo
from qick.asm_v2 import QickParam

from zcu_tools.cfg_model import ConfigBase

if TYPE_CHECKING:
    from zcu_tools.program.v2.modular import ModularProgramV2
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


class Module(ABC):
    name: str

    @abstractmethod
    def init(self, prog: ModularProgramV2) -> None: ...

    @abstractmethod
    def run(
        self, prog: ModularProgramV2, t: float | QickParam = 0.0
    ) -> float | QickParam: ...

    def allow_rerun(self) -> bool:
        return False


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
