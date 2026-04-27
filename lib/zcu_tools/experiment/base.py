from __future__ import annotations

from typing_extensions import Generic, Optional, TypeVar

from zcu_tools.experiment.cfg_model import ExpCfgModel

T_Result = TypeVar("T_Result")
T_Config = TypeVar("T_Config", bound=ExpCfgModel)


class AbsExperiment(Generic[T_Result, T_Config]):
    """
    Defines the interface for an experiment.
    """

    def __init__(self) -> None:
        self.last_cfg: Optional[T_Config] = None
        self.last_result: Optional[T_Result] = None
