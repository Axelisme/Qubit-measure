from __future__ import annotations

from collections.abc import Iterable

from .instructions import GenericInst, Instruction

PSEUDO_LABELS = frozenset({"PREV", "HERE", "NEXT", "SKIP"})


def iter_label_references(inst: Instruction) -> Iterable[str]:
    label = inst.need_label
    if label:
        return (label,)
    return ()
