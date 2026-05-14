"""DeadWriteEliminationPass: remove overwritten register writes with no intervening reads.

Purpose
-------
If a register is written twice without being read in between, the first write
is dead — its value is never observed.  Removing it shrinks the instruction
stream and may expose further optimisation opportunities.

Example
-------
Before::

    REG_WR r0 imm #1    ; dead: r0 is immediately overwritten
    REG_WR r0 imm #5
    TIME inc_ref #100

After::

    REG_WR r0 imm #5
    TIME inc_ref #100

QICK Hardware Notes
-------------------
- Volatile registers (s0–s14) have hardware side effects on every write
  (e.g., s14 is the time reference, s0–s13 drive hardware signals directly).
  Their writes are NEVER eliminated regardless of liveness.
- The ``-uf`` flag on a RegWriteInst updates the condition-flag register as a
  side effect of the ALU operation.  Such instructions are treated as barriers:
  all pending shadow tracking is cleared because the flag state affects
  subsequent conditional jumps.
- A ``REG_WR dst wmem`` instruction reads from wave memory (a hardware FIFO)
  and simultaneously writes the entire ``w0–w5`` / ``r_wave`` alias group.
  It is treated as an opaque read+write barrier: pending tracking is cleared
  and the wmem read itself is never eliminated.
- The ``r_wave`` bundle reference: because ``WmemWriteInst.reg_read`` includes
  the full wave-register bundle, a shadow entry for ``r_wave`` acts as a
  DCE-level sentinel that flushes when *any* ``wN`` alias appears in a read or write.
- JumpInst and other non-data instructions with control-flow semantics are
  treated as barriers (``_is_write_tracking_barrier``): all pending entries
  are conservatively flushed.

Decision Notes
--------------
Tracking uses canonical register names. Instructions with multiple destinations
(like ``r_wave`` expanding to ``{w0, …, w5}``) are tracked precisely: the
instruction is only marked dead if *all* of its destinations are shadowed by
subsequent writes before any intervening reads. If any destination is read,
the entire instruction is marked live. Hardware side-effects (volatile
registers, status flag updates, or wave memory reads) always protect an
instruction from elimination.
"""

from __future__ import annotations

from ...instructions import (
    ArithInst,
    BaseInst,
    CallInst,
    ClearInst,
    ComInst,
    CustomPeripheralInst,
    DivInst,
    DmemReadInst,
    DmemWriteInst,
    DportReadInst,
    FlagInst,
    NetInst,
    NopInst,
    PortWriteInst,
    RegWriteInst,
    RetInst,
    TimeInst,
    TrigInst,
    WaitInst,
    WmemWriteInst,
)
from ...node import BasicBlockNode
from ...operands import Register, SrcKeyword
from ...pipeline import AbsChunkPass, ChunkList, PipeLineContext


class DeadWriteEliminationPass(AbsChunkPass):
    """Remove overwritten register writes in free BasicBlockNode chunks."""

    def process(
        self, chunks: ChunkList, ctx: PipeLineContext
    ) -> tuple[ChunkList, bool]:
        _ = ctx
        changed = False
        for chunk in chunks:
            if not isinstance(chunk, BasicBlockNode):
                continue
            changed |= self._process_block(chunk)
        return chunks, changed

    def _process_block(self, block: BasicBlockNode) -> bool:
        if block.disable_opt:
            return False
        insts = block.insts
        dead: set[int] = self._find_dead_indices(insts)
        if not dead:
            return False
        block.insts = [inst for i, inst in enumerate(insts) if i not in dead]
        return True

    def _find_dead_indices(self, insts: list[BaseInst]) -> set[int]:
        # inst_pending_regs: index -> set of registers written by this instruction
        # that are still pending (not read and not yet shadowed).
        inst_pending_regs: dict[int, set[str]] = {}
        # reg_to_inst: canonical reg name -> index of the latest instruction that wrote to it.
        reg_to_inst: dict[str, int] = {}
        dead: set[int] = set()

        for idx, inst in enumerate(insts):
            # Treat conditionally executed writes as barriers: they do not
            # reliably shadow previous writes because they may be skipped.
            if (
                self._is_write_tracking_barrier(inst)
                or getattr(inst, "if_cond", None) is not None
            ):
                inst_pending_regs.clear()
                reg_to_inst.clear()
                continue

            # Special case: instructions that read from external state (wmem)
            # are never dead, but they do shadow previous writes to wave registers.
            is_wmem_read = (
                isinstance(inst, RegWriteInst) and inst.src == SrcKeyword.WMEM
            )

            reads = inst.reg_read
            writes = inst.reg_write

            # 1. Process Reads: any read makes the source instruction "not dead".
            for reg in reads:
                if reg in reg_to_inst:
                    prev_idx = reg_to_inst[reg]
                    if prev_idx in inst_pending_regs:
                        # This instruction is now "live" because at least one of
                        # its outputs is read. Remove it from tracking.
                        for r in inst_pending_regs.pop(prev_idx):
                            reg_to_inst.pop(r, None)

            # 2. Process Side-effects: instructions with hardware side effects
            # (volatile regs, -uf, or wmem read) are never candidates for removal.
            can_be_dead = not (
                getattr(inst, "uf", False)
                or is_wmem_read
                or any(Register(w).is_volatile_reg() for w in writes)
            )

            # 3. Process Writes: shadowing previous writes.
            for reg in writes:
                if reg in reg_to_inst:
                    prev_idx = reg_to_inst[reg]
                    if prev_idx in inst_pending_regs:
                        inst_pending_regs[prev_idx].remove(reg)
                        if not inst_pending_regs[prev_idx]:
                            # ALL outputs of prev_idx are now shadowed.
                            dead.add(prev_idx)
                            del inst_pending_regs[prev_idx]

            # 4. Track this instruction if it's a candidate for DCE.
            if can_be_dead and writes:
                inst_pending_regs[idx] = set(writes)
                for reg in writes:
                    reg_to_inst[reg] = idx

        return dead

    def _is_write_tracking_barrier(self, inst: BaseInst) -> bool:
        return not isinstance(
            inst,
            (
                TimeInst,
                WaitInst,
                RegWriteInst,
                DmemReadInst,
                DmemWriteInst,
                PortWriteInst,
                WmemWriteInst,
                NopInst,
            ),
        ) and not isinstance(
            inst,
            (
                ArithInst,
                CallInst,
                ClearInst,
                ComInst,
                CustomPeripheralInst,
                DivInst,
                DportReadInst,
                FlagInst,
                NetInst,
                RetInst,
                TrigInst,
            ),
        )
