from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar

T_Result = TypeVar("T_Result")


class AbsExperiment(Generic[T_Result], ABC):
    """
    Defines the interface for an experiment.
    """

    def __init__(self) -> None:
        self.last_cfg: Optional[Dict[str, Any]] = None
        self.last_result: Optional[T_Result] = None

    @abstractmethod
    def run(self, progress: bool = True, **kwargs) -> T_Result:
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
        tag: str = "",
        **kwargs,
    ) -> None:
        pass
