from __future__ import annotations

from ..analysis import instruction_reads, instruction_writes
from ..instructions import Instruction, JumpInst, NopInst, TestInst
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


class DeadTestEliminationLinear(AbsLinearPass):
    """Remove TestInst whose result (flag) is never consumed by a conditional jump.

    A TestInst is dead when, before the next conditional JumpInst reads the flag,
    either another TestInst overwrites it, or the block ends without a conditional jump.

    fix_addr_size=False: removes dead TestInst from the list.
    fix_addr_size=True:  replaces dead TestInst with NopInst.
    """

    def process_block(self, block: BasicBlockNode) -> None:
        dead = self._find_dead_indices(block.insts, block.branch)
        if not dead:
            return
        if block.fix_addr_size:
            block.insts = [
                NopInst() if i in dead else inst for i, inst in enumerate(block.insts)
            ]
        else:
            block.insts = [inst for i, inst in enumerate(block.insts) if i not in dead]

    def _find_dead_indices(
        self, insts: list[Instruction], branch: JumpInst | None
    ) -> set[int]:
        pending: int | None = None  # index of the last TestInst not yet consumed
        dead: set[int] = set()

        for idx, inst in enumerate(insts):
            if isinstance(inst, TestInst):
                if pending is not None:
                    dead.add(pending)  # previous TEST was never consumed
                pending = idx
            elif isinstance(inst, JumpInst) and inst.if_cond is not None:
                pending = None  # flag consumed by conditional jump

        # After all insts, check the block's branch.
        if pending is not None:
            if branch is None or branch.if_cond is None:
                dead.add(pending)  # flag never consumed before block exit

        return dead
