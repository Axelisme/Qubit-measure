from __future__ import annotations

from collections.abc import Iterable

from .instructions import GenericInst, Instruction

PSEUDO_LABELS = frozenset({"PREV", "HERE", "NEXT", "SKIP"})


def iter_label_references(inst: Instruction) -> Iterable[str]:
    if not isinstance(inst, GenericInst):
        return ()

    label = inst.args.get("LABEL")
    if not isinstance(label, str):
        return ()
    if label in PSEUDO_LABELS:
        return ()

    return (label,)
