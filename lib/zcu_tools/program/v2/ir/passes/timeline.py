from __future__ import annotations

import dataclasses
from typing import cast

from ..instructions import Instruction, NopInst, TimeInst
from ..node import BasicBlockNode
from ..pipeline import AbsLinearPass
from ..utils import regs_from_value


def _is_zero_ref_increment(inst: Instruction) -> bool:
    if not isinstance(inst, TimeInst):
        return False
    if inst.c_op != "inc_ref":
        return False
    if inst.r1 is not None:
        return False
    if inst.lit is None:
        return False
    lit = inst.lit
    if not lit.startswith("#"):
        return False
    try:
        return int(lit[1:]) == 0
    except ValueError:
        return False


def _is_lit_time(inst: Instruction) -> bool:
    """True for TIME inc_ref #N with N > 0 (no register operand)."""
    if not isinstance(inst, TimeInst):
        return False
    if inst.c_op != "inc_ref" or inst.r1 is not None or inst.lit is None:
        return False
    if not inst.lit.startswith("#"):
        return False
    try:
        return int(inst.lit[1:]) > 0
    except ValueError:
        return False


def _get_lit_time_value(inst: TimeInst) -> int:
    return int(cast(str, inst.lit)[1:])


def _is_reg_time(inst: Instruction) -> bool:
    """True for TIME inc_ref rX (register-driven increment)."""
    return (
        isinstance(inst, TimeInst)
        and inst.c_op == "inc_ref"
        and inst.r1 is not None
    )


def _is_anchored_timed(inst: Instruction) -> bool:
    """True when inst has a time field of the form '@N' with N a plain integer."""
    t = getattr(inst, "time", None)
    if not isinstance(t, str) or not t.startswith("@"):
        return False
    return not regs_from_value(t)  # empty set → no register token → pure integer


def _adjust_time_field(inst: Instruction, delta: int) -> Instruction:
    """Return a copy of inst with time adjusted by +delta (precondition: _is_anchored_timed)."""
    old = int(cast(str, getattr(inst, "time"))[1:])
    return dataclasses.replace(inst, time=f"@{old + delta}")  # type: ignore[call-overload]


class ZeroDelayDCELinear(AbsLinearPass):
    """Remove TIME inc_ref #0 instructions from a BasicBlockNode.

    fix_addr_size=False: removes zero-delay TIME instructions.
    fix_addr_size=True:  replaces them with NopInst to preserve stride.
    """

    def process_block(self, block: BasicBlockNode) -> None:
        if block.fix_addr_size:
            block.insts = [
                NopInst() if _is_zero_ref_increment(inst) else inst
                for inst in block.insts
            ]
        else:
            block.insts = [
                inst for inst in block.insts if not _is_zero_ref_increment(inst)
            ]


class TimedMergeLinear(AbsLinearPass):
    """Aggressive TIME inc_ref optimisation pass.

    For free blocks (fix_addr_size=False):
      - Sinks lit-TIME instructions (#N, N>0) forward past non-timed instructions.
      - When a lit-TIME meets a timed instruction (@M), absorbs the accumulated
        delta into its timestamp (@M → @(M+delta)) and places the TIME after it.
      - Multiple lit-TIMEs are accumulated (equivalent to merging them).
      - reg-TIME (register-driven) acts as a conservative flush-and-barrier:
        flushes pending_lit before it, then is emitted in-place.

    For fixed blocks (fix_addr_size=True):
      - Falls back to adjacent-only merge with NopInst padding to preserve stride.
    """

    def process_block(self, block: BasicBlockNode) -> None:
        if block.fix_addr_size:
            self._merge_fixed(block)
        else:
            self._merge_free(block)

    def _merge_free(self, block: BasicBlockNode) -> None:
        pending_lit: int = 0
        result: list[Instruction] = []

        for inst in block.insts:
            if _is_lit_time(inst):
                # Accumulate without emitting; the TIME will be placed later.
                pending_lit += _get_lit_time_value(cast(TimeInst, inst))

            elif _is_anchored_timed(inst):
                # Absorb pending_lit into the timestamp; do NOT reset pending_lit.
                # All subsequent timed instructions in the same baseline segment
                # must be adjusted by the same amount.
                result.append(
                    _adjust_time_field(inst, pending_lit) if pending_lit > 0 else inst
                )

            elif _is_reg_time(inst):
                # Conservative barrier: flush accumulated lit-TIME before reg-TIME.
                if pending_lit > 0:
                    result.append(TimeInst(c_op="inc_ref", lit=f"#{pending_lit}"))
                    pending_lit = 0
                result.append(inst)

            else:
                # Non-time, non-anchored instruction: emit as-is.
                # pending_lit silently sinks past it.
                result.append(inst)

        if pending_lit > 0:
            result.append(TimeInst(c_op="inc_ref", lit=f"#{pending_lit}"))

        block.insts = result

    def _merge_fixed(self, block: BasicBlockNode) -> None:
        # Merge run values into the first slot; fill the rest with NOP.
        result: list[Instruction] = list(block.insts)
        i = 0
        while i < len(result):
            if not _is_lit_time(result[i]):
                i += 1
                continue
            # Start of a run — find its extent.
            j = i + 1
            while j < len(result) and _is_lit_time(result[j]):
                j += 1
            if j == i + 1:
                i += 1
                continue
            # Run from i to j-1: sum values into slot i, NOP out i+1..j-1.
            total = sum(
                _get_lit_time_value(cast(TimeInst, result[k])) for k in range(i, j)
            )
            result[i] = TimeInst(c_op="inc_ref", lit=f"#{total}")
            for k in range(i + 1, j):
                result[k] = NopInst()
            i = j
        block.insts = result
