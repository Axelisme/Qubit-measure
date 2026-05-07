from __future__ import annotations

import re

from ..analysis import instruction_reads, instruction_writes
from ..instructions import Instruction, JumpInst, NopInst, RegWriteInst, TestInst
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


def _is_const_increment(inst: Instruction) -> tuple[str, int] | None:
    if not isinstance(inst, RegWriteInst):
        return None
    if inst.src != "op" or inst.op is None:
        return None
    if inst.uf is not None or inst.if_cond is not None or inst.wr is not None or inst.label is not None or inst.addr is not None:
        return None
        
    m = re.match(r"^([a-zA-Z0-9_]+)\s*([\+\-])\s*#(\d+)$", inst.op.strip())
    if not m:
        return None
    lhs, op_char, val_str = m.groups()
    if lhs != inst.dst:
        return None
    
    val = int(val_str)
    if op_char == "-":
        val = -val
    return inst.dst, val

def _make_increment_inst(reg: str, val: int) -> RegWriteInst:
    if val >= 0:
        op_str = f"{reg} + #{val}"
    else:
        op_str = f"{reg} - #{-val}"
    return RegWriteInst(dst=reg, src="op", op=op_str)

class IncRegMergeLinear(AbsLinearPass):
    """Merge adjacent constant register increments.
    
    For free blocks (fix_addr_size=False):
      - Sinks constant increments (REG_WR rX op (rX + #C)) forward past instructions 
        that do not read or write rX.
      - Accumulates multiple increments into a single increment.
      - Drops the increment if the accumulated value is 0.

    For fixed blocks (fix_addr_size=True):
      - Only merges strictly adjacent increments for the same register to preserve stride.
    """
    def process_block(self, block: BasicBlockNode) -> None:
        if block.fix_addr_size:
            self._merge_fixed(block)
        else:
            self._merge_free(block)

    def _merge_free(self, block: BasicBlockNode) -> None:
        pending: dict[str, int] = {}
        result: list[Instruction] = []

        for inst in block.insts:
            if not is_safe_linear_inst(inst):
                # Flush ALL pending before unsafe instruction
                for reg, val in pending.items():
                    if val != 0:
                        result.append(_make_increment_inst(reg, val))
                pending.clear()
                result.append(inst)
                continue

            inc_info = _is_const_increment(inst)
            if inc_info is not None:
                reg, val = inc_info
                pending[reg] = pending.get(reg, 0) + val
            else:
                reads = instruction_reads(inst)
                writes = instruction_writes(inst)
                
                # Flush pending increments if their register is read or written
                for reg in list(pending.keys()):
                    if reg in reads or reg in writes:
                        val = pending.pop(reg)
                        if val != 0:
                            result.append(_make_increment_inst(reg, val))
                            
                result.append(inst)

        # Flush remaining at the end of the block
        for reg, val in pending.items():
            if val != 0:
                result.append(_make_increment_inst(reg, val))

        block.insts = result

    def _merge_fixed(self, block: BasicBlockNode) -> None:
        result: list[Instruction] = list(block.insts)
        i = 0
        while i < len(result):
            inst = result[i]
            inc_info = _is_const_increment(inst)
            if inc_info is None:
                i += 1
                continue
                
            reg, val = inc_info
            # Find run of increments for the SAME register
            j = i + 1
            while j < len(result):
                next_info = _is_const_increment(result[j])
                if next_info is not None and next_info[0] == reg:
                    val += next_info[1]
                    j += 1
                else:
                    break
                    
            if j == i + 1:
                i += 1
                continue
                
            # Merge run
            if val != 0:
                result[i] = _make_increment_inst(reg, val)
            else:
                result[i] = NopInst()
                
            for k in range(i + 1, j):
                result[k] = NopInst()
            i = j
            
        block.insts = result
