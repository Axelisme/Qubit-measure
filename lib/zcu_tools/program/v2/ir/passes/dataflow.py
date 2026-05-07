from __future__ import annotations

from ..analysis import instruction_reads, instruction_writes
from ..instructions import Instruction, NopInst
from ..node import BasicBlockNode
from ..pipeline import AbsLinearPass
from .base import is_safe_linear_inst


class DeadWriteEliminationLinear(AbsLinearPass):
    """Remove overwritten register writes in a BasicBlockNode.

    fix_addr_size=False: removes dead-write instructions from the list.
    fix_addr_size=True:  replaces dead-write instructions with NopInst to
                        preserve jump-table stride.
    """

    def process_block(self, block: BasicBlockNode) -> None:
        insts = block.insts
        dead: set[int] = self._find_dead_indices(insts)
        if not dead:
            return
        if block.fix_addr_size:
            block.insts = [
                NopInst() if i in dead else inst for i, inst in enumerate(insts)
            ]
        else:
            block.insts = [inst for i, inst in enumerate(insts) if i not in dead]

    def _find_dead_indices(self, insts: list[Instruction]) -> set[int]:
        pending: dict[str, int] = {}  # reg -> index of last pending write
        dead: set[int] = set()

        for idx, inst in enumerate(insts):
            if not is_safe_linear_inst(inst):
                pending.clear()
                continue

            reads = instruction_reads(inst)
            writes = list(instruction_writes(inst))

            if len(writes) > 1:
                pending.clear()
                continue

            for reg in reads:
                pending.pop(reg, None)

            if len(writes) == 1:
                dst = writes[0]
                prev_idx = pending.get(dst)
                if prev_idx is not None:
                    dead.add(prev_idx)
                pending[dst] = idx

        return dead
