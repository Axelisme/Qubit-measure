from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from typing_extensions import Literal, Union

if TYPE_CHECKING:
    from .pipeline import ChunkList

PSEUDO_LABELS: frozenset[str] = frozenset({"PREV", "HERE", "NEXT", "SKIP"})

PseudoLabel = Literal["PREV", "HERE", "NEXT", "SKIP"]


@dataclass(frozen=True)
class Label:
    """A first-class logical label identity.

    Value object: equality and hashing are based solely on ``name``.
    Two ``Label`` instances with the same name are considered identical.
    """

    name: str

    def __str__(self) -> str:
        return f"&{self.name}"

    def __repr__(self) -> str:
        return f"Label({self.name!r})"


@dataclass(frozen=True)
class LabelRef:
    """A label reference (jump target / address-load operand).

    ``target`` is either a ``Label`` (normal label) or a ``PseudoLabel`` str
    (hardware-reserved: HERE / NEXT / SKIP / PREV).
    """

    target: Union[Label, PseudoLabel]

    def is_pseudo(self) -> bool:
        return isinstance(self.target, str)

    def as_label(self) -> Label:
        """Return the underlying Label; raises if this is a pseudo-label."""
        if not isinstance(self.target, Label):
            raise TypeError(f"LabelRef {self!r} is a pseudo-label, not a Label")
        return self.target

    def __str__(self) -> str:
        if isinstance(self.target, Label):
            return str(self.target)
        return self.target

    def __repr__(self) -> str:
        return f"LabelRef({self.target!r})"


def make_label(base: str, allocated: set[str]) -> Label:
    """Create a ``Label`` whose name is unique within ``allocated``.

    If ``base`` is already in ``allocated``, a numeric suffix is appended
    (``base_0``, ``base_1``, …) until a free name is found.  The chosen name
    is added to ``allocated`` before returning.
    """
    name = base
    counter = 0
    while name in allocated:
        name = f"{base}_{counter}"
        counter += 1
    allocated.add(name)
    return Label(name)


def collect_referenced_labels(chunks: ChunkList) -> set[Label]:
    """Collect all Labels referenced across a chunk list.

    Uses ``BaseInst.need_labels`` so multi-label references (e.g. a dmem
    dispatch table addressed via ``DmemAddr``) keep every referenced label
    alive, not just the single ``need_label``.
    """
    from .instructions import BaseInst
    from .node import BasicBlockNode

    refs: set[Label] = set()
    for chunk in chunks:
        if not isinstance(chunk, BasicBlockNode):
            continue
        for inst in (
            *chunk.labels,
            *chunk.insts,
            *([chunk.branch] if chunk.branch else []),
        ):
            if isinstance(inst, BaseInst):
                refs |= inst.need_labels
    return refs
