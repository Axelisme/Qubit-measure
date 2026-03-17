from __future__ import annotations

from abc import ABC, abstractmethod

from typing_extensions import Any, Generic, Mapping, Optional, TypeVar

T_Result = TypeVar("T_Result")
T_Config = TypeVar("T_Config", bound=Mapping[str, Any])


class AbsExperiment(Generic[T_Result, T_Config], ABC):
    """
    Defines the interface for an experiment.
    """

    def __init__(self) -> None:
        self.last_cfg: Optional[T_Config] = None
        self.last_result: Optional[T_Result] = None

    @abstractmethod
    def run(self, *args, **kwargs) -> T_Result: ...

    @abstractmethod
    def save(
        self, filepath: str, result: Optional[T_Result] = None, **kwargs
    ) -> None: ...

    @abstractmethod
    def load(self, filepath: str, **kwargs) -> T_Result: ...
