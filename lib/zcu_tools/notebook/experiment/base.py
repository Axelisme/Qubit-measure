from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

from zcu_tools.liveplot import AbsLivePlotter

T_Result = TypeVar("T_Result")


class AbsExperiment(Generic[T_Result], ABC):
    """
    Defines the interface for an experiment.
    """

    @abstractmethod
    def run(
        self,
        progress: bool = True,
        liveplotter: Optional[AbsLivePlotter] = None,
        **kwargs,
    ) -> T_Result:
        pass

    @abstractmethod
    def analyze(self, result: Optional[T_Result] = None, **kwargs) -> Any:
        pass

    @abstractmethod
    def save(
        self,
        filepath: str,
        result: Optional[T_Result] = None,
        comment: Optional[str] = None,
        **kwargs,
    ) -> None:
        pass
