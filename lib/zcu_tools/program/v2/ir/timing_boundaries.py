"""Carry timing metadata through optimization using non-removable label anchors."""

from .instructions import Instruction, LabelInst, MetaInst
from .labels import make_label


def anchor_boundaries(
    insts: list[Instruction],
) -> tuple[list[Instruction], dict[str, MetaInst]]:
    """Replace TIMING_BOUNDARY metadata with unique optimizer label anchors."""
    allocated = {str(inst.name) for inst in insts if isinstance(inst, LabelInst)}
    anchors: dict[str, MetaInst] = {}
    result: list[Instruction] = []
    for inst in insts:
        if isinstance(inst, MetaInst) and inst.type == "TIMING_BOUNDARY":
            label = make_label("timing_boundary", allocated)
            anchors[label.name] = inst
            result.append(LabelInst(name=label, can_remove=False))
        else:
            result.append(inst)
    return result, anchors


def restore_boundaries(
    insts: list[Instruction], anchors: dict[str, MetaInst]
) -> list[Instruction]:
    """Restore anchored metadata; reject lost anchors instead of guessing addresses."""
    result: list[Instruction] = []
    restored: set[str] = set()
    for inst in insts:
        if isinstance(inst, LabelInst) and inst.name.name in anchors:
            name = inst.name.name
            if name in restored:
                raise ValueError(f"Timing boundary anchor duplicated: {name}")
            restored.add(name)
            result.append(anchors[name])
        else:
            result.append(inst)
    if restored != anchors.keys():
        raise ValueError("Timing boundary anchors lost or cloned during optimization")
    return result
