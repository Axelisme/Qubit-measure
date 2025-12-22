from abc import ABC, abstractmethod
from typing import Any, Mapping, Generic, Optional, TypeVar, Union, List


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
    def run(self) -> T_Result: ...

    @abstractmethod
    def analyze(self, result: Optional[T_Result] = None) -> None: ...

    @abstractmethod
    def save(
        self,
        filepath: str,
        result: Optional[T_Result] = None,
        comment: Optional[str] = None,
        tag: str = "",
    ) -> None: ...

    @abstractmethod
    def load(self, filepath: Union[str, List[str]], **kwargs) -> T_Result: ...
