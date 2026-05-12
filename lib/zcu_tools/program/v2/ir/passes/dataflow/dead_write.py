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
- The ``r_wave`` bundle reference: because ``wN.get_write_regs()`` returns
  ``{wN, r_wave}``, a shadow entry for ``r_wave`` acts as a DCE-level sentinel
  that flushes when *any* ``wN`` alias appears in a read or write.
- JumpInst and other non-data instructions with control-flow semantics are
  treated as barriers (``_is_write_tracking_barrier``): all pending entries
  are conservatively flushed.

Decision Notes
--------------
Tracking uses canonical register names (``inst.reg_write`` already returns
canonicals after the operands refactor).  Aliased writes (``len(writes) > 1``,
e.g., ``r_wave`` expanding to ``{r_wave, w0, …, w5}``) are treated as a group
flush rather than shadow tracking to avoid incorrectly marking one alias as
dead when another alias is later read.
"""

from __future__ import annotations

from ...instructions import (
    BaseInst,
    DmemReadInst,
    DmemWriteInst,
    NopInst,
    PortWriteInst,
    RegWriteInst,
    TimeInst,
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
        if block.fix_addr_size:
            return False
        insts = block.insts
        dead: set[int] = self._find_dead_indices(insts)
        if not dead:
            return False
        block.insts = [inst for i, inst in enumerate(insts) if i not in dead]
        return True

    def _find_dead_indices(self, insts: list[BaseInst]) -> set[int]:
        # pending: canonical reg name -> index of last pending write.  Using
        # canonical names ensures that an alias (e.g. w_freq) and the
        # underlying register (w0) collide correctly when checking shadows.
        pending: dict[str, int] = {}
        dead: set[int] = set()

        for idx, inst in enumerate(insts):
            if self._is_write_tracking_barrier(inst):
                pending.clear()
                continue

            # REG_WR <dst> wmem reads from wave memory (a non-register side
            # effect) and writes to r_wave / w0..w5 as an aliased group.  Treat
            # it as an opaque read barrier: clear pending and skip shadow
            # tracking so the wmem read is never eliminated and its writes
            # cannot be shadowed by later wN writes.
            if isinstance(inst, RegWriteInst) and inst.src == SrcKeyword.WMEM:
                pending.clear()
                continue

            reads = set(inst.reg_read)
            writes = list(inst.reg_write)

            for reg in reads:
                pending.pop(reg, None)

            # Do not eliminate instructions with hardware side effects:
            # 1. Flag updates (-uf)
            # 2. Volatile registers (s0-s14)
            # 3. Multiple writes (usually aliasing like r_wave, treat as barrier for simplicity)
            if getattr(inst, "uf", False):
                pending.clear()
                continue

            if len(writes) == 1:
                dst = writes[0]
                if Register(dst).is_volatile():
                    continue
                prev_idx = pending.get(dst)
                if prev_idx is not None:
                    dead.add(prev_idx)
                pending[dst] = idx
            elif len(writes) > 1:
                # Aliasing write (e.g. r_wave). Clear all shadowed registers from pending.
                for dst in writes:
                    prev_idx = pending.get(dst)
                    if prev_idx is not None:
                        dead.add(prev_idx)
                    pending.pop(dst, None)

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
        )
