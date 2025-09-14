from __future__ import annotations
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Optional, TYPE_CHECKING


from ..base import MyProgramV2

if TYPE_CHECKING:
    from ..modules import ModuleLibrary


def str2module(
    ml: ModuleLibrary, module_cfg: Union[str, Dict[str, Any]]
) -> Dict[str, Any]:
    module_cfg = deepcopy(module_cfg)

    if isinstance(module_cfg, str):
        name = module_cfg
        module_cfg = deepcopy(ml.get_module(name))
        module_cfg["name"] = name

    return module_cfg


class Module(ABC):
    @abstractmethod
    def init(self, prog: MyProgramV2) -> None:
        pass

    @abstractmethod
    def run(self, prog: MyProgramV2) -> None:
        pass

    @classmethod
    def derive_cfg(
        self, ml: ModuleLibrary, module_cfg: Optional[Union[str, Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("derive_cfg not implemented for this module")
