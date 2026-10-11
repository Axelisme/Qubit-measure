from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..model import CfgSectionValue


class ExpressionEvaluator(Protocol):
    """Evaluate authored text against fixed sources; failures propagate.

    Return finite numeric scalars or name strings. Field owners enforce the
    target type; a string must never be coerced into a numeric scalar.
    """

    def __call__(self, expression: str) -> int | float | complex | str: ...


class OptionProvider(Protocol):
    def __call__(self, source_id: str) -> Sequence[object]: ...


@dataclass(frozen=True)
class ResolvedReference:
    label: str
    value: CfgSectionValue | None


class ReferenceCatalog(Protocol):
    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]: ...

    def resolve(self, kind: str, key: str) -> ResolvedReference | None: ...
