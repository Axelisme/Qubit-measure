from __future__ import annotations

from typing import cast

from ..instructions import Instruction, NopInst, TimeInst
from ..node import BasicBlockNode
from ..pipeline import AbsLinearPass


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


def _is_mergeable_time_increment(inst: Instruction) -> bool:
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
        value = int(lit[1:])
    except ValueError:
        return False
    return value > 0


class ZeroDelayDCELinear(AbsLinearPass):
    """Remove TIME inc_ref #0 instructions from a BasicBlockNode.

    fix_inst_num=False: removes zero-delay TIME instructions.
    fix_inst_num=True:  replaces them with NopInst to preserve stride.
    """

    def process_block(self, block: BasicBlockNode) -> None:
        if block.fix_inst_num:
            block.insts = [
                NopInst() if _is_zero_ref_increment(inst) else inst
                for inst in block.insts
            ]
        else:
            block.insts = [
                inst for inst in block.insts if not _is_zero_ref_increment(inst)
            ]


class TimedMergeLinear(AbsLinearPass):
    """Merge adjacent TIME inc_ref #N instructions in a BasicBlockNode.

    fix_inst_num=False: merges adjacent runs into a single instruction.
    fix_inst_num=True:  merges the value into the first instruction of each
                        run, then replaces the remaining instructions with
                        NopInst to preserve stride.
    """

    def process_block(self, block: BasicBlockNode) -> None:
        if block.fix_inst_num:
            self._merge_fixed(block)
        else:
            self._merge_free(block)

    def _merge_free(self, block: BasicBlockNode) -> None:
        result: list[Instruction] = []
        pending_run: list[TimeInst] = []

        def flush() -> None:
            if not pending_run:
                return
            if len(pending_run) == 1:
                result.append(pending_run[0])
            else:
                total = sum(int(t.lit[1:]) for t in pending_run if t.lit is not None)
                result.append(TimeInst(c_op="inc_ref", lit=f"#{total}"))
            pending_run.clear()

        for inst in block.insts:
            if _is_mergeable_time_increment(inst):
                pending_run.append(cast(TimeInst, inst))
            else:
                flush()
                result.append(inst)

        flush()
        block.insts = result

    def _merge_fixed(self, block: BasicBlockNode) -> None:
        # Merge run values into the first slot; fill the rest with NOP.
        result: list[Instruction] = list(block.insts)
        i = 0
        while i < len(result):
            if not _is_mergeable_time_increment(result[i]):
                i += 1
                continue
            # Start of a run — find its extent.
            j = i + 1
            while j < len(result) and _is_mergeable_time_increment(result[j]):
                j += 1
            if j == i + 1:
                i += 1
                continue
            # Run from i to j-1: sum values into slot i, NOP out i+1..j-1.
            total = sum(
                int(cast(TimeInst, result[k]).lit[1:])  # type: ignore[union-attr]
                for k in range(i, j)
            )
            result[i] = TimeInst(c_op="inc_ref", lit=f"#{total}")
            for k in range(i + 1, j):
                result[k] = NopInst()
            i = j
        block.insts = result
