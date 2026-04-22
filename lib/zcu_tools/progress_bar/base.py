from __future__ import annotations

from abc import ABC, abstractmethod
from typing_extensions import Optional


class ProgressSink(ABC):
    @abstractmethod
    def start(
        self, total: Optional[int], desc: str = "progress", leave: bool = True
    ) -> None: ...

    @abstractmethod
    def update_to(self, n: int) -> None: ...

    @abstractmethod
    def close(self) -> None: ...
