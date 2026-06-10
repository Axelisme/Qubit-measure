from __future__ import annotations

from abc import ABC, abstractmethod

ProgressValue = int | float
ProgressTotal = ProgressValue | None


class BaseProgressBar(ABC):
    @abstractmethod
    def set_description(self, description: str) -> None:
        pass

    @abstractmethod
    def update(self, value: ProgressValue = 1) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def refresh(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @property
    @abstractmethod
    def total(self) -> ProgressTotal:
        pass

    @total.setter
    @abstractmethod
    def total(self, value: ProgressTotal) -> None:
        pass

    @property
    @abstractmethod
    def n(self) -> ProgressValue:
        pass

    @property
    @abstractmethod
    def desc(self) -> str:
        pass
