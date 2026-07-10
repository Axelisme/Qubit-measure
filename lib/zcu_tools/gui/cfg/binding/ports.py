from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..model import CfgSectionValue


class ExpressionEvaluator(Protocol):
    def __call__(self, expression: str) -> int | float: ...


class OptionProvider(Protocol):
    def __call__(self, source_id: str) -> Sequence[object]: ...


@dataclass(frozen=True)
class ResolvedReference:
    label: str
    value: CfgSectionValue | None


class ReferenceCatalog(Protocol):
    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]: ...

    def resolve(self, kind: str, key: str) -> ResolvedReference | None: ...
