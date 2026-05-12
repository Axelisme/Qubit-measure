from __future__ import annotations

from ..analysis import instruction_reads, instruction_writes, is_wmem_load
from ..instructions import (
    BaseInst,
    DmemReadInst,
    DmemWriteInst,
    Instruction,
    JumpInst,
    NopInst,
    PortWriteInst,
    RegWriteInst,
    TestInst,
    TimeInst,
    WaitInst,
    WmemWriteInst,
)
from ..node import BasicBlockNode
from ..pipeline import AbsChunkPass, ChunkList, PipeLineContext


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
            if is_wmem_load(inst):
                pending.clear()
                continue

            reads = instruction_reads(inst)
            writes = list(instruction_writes(inst))

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


class DeadTestEliminationPass(AbsChunkPass):
    """Remove dead TestInst from free BasicBlockNode chunks."""

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
        dead = self._find_dead_indices(block.insts, block.branch)
        if not dead:
            return False
        block.insts = [inst for i, inst in enumerate(block.insts) if i not in dead]
        return True

    def _find_dead_indices(
        self, insts: list[BaseInst], branch: JumpInst | None
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


from ..operands import AluExpr, AluOp, Immediate, Register, SrcKeyword


def _is_const_increment(inst: Instruction) -> tuple[str, int] | None:
    if not isinstance(inst, RegWriteInst):
        return None
    if inst.src != "op" or not isinstance(inst.op, AluExpr):
        return None
    if (
        inst.uf
        or inst.if_cond is not None
        or inst.wr is not None
        or inst.label is not None
        or inst.addr is not None
    ):
        return None

    op = inst.op
    if op.op not in (AluOp.ADD, AluOp.SUB):
        return None
    if not isinstance(op.rhs, Immediate):
        return None

    lhs = op.lhs
    if lhs != inst.dst:
        return None

    # Only optimize general-purpose user registers (r0, r1, ...).
    # System registers (sN) and wave registers (wN/r_wave) are excluded for safety,
    # as they often represent hardware state or pulse parameters that should
    # remain at their original program positions.
    if not lhs.is_general_reg():
        return None

    canon = lhs.canonical()

    val = op.rhs.value

    if op.op == AluOp.SUB:
        val = -val
    # Pending entries are keyed by canonical name so that aliased writes
    # (e.g. w_freq -> w0) are flushed when *any* alias appears in another
    # instruction's reads or writes.
    return canon, val


def _make_increment_inst(reg: str, val: int) -> RegWriteInst:
    if val >= 0:
        op = AluExpr(Register(reg), AluOp.ADD, Immediate(val))
    else:
        op = AluExpr(Register(reg), AluOp.SUB, Immediate(-val))
    return RegWriteInst(dst=Register(reg), src=SrcKeyword.OP, op=op)


class IncRegMergePass(AbsChunkPass):
    """Merge adjacent constant register increments.

    For free blocks (fix_addr_size=False):
      - Sinks constant increments (REG_WR rX op (rX + #C)) forward past instructions
        that do not read or write rX.
      - Accumulates multiple increments into a single increment.
      - Drops the increment if the accumulated value is 0.
    """

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
        before = list(block.insts)
        self._merge_free(block)
        return before != block.insts

    def _merge_free(self, block: BasicBlockNode) -> None:
        # Pending entries are keyed by canonical register name (see
        # _is_const_increment), so any alias appearing in another
        # instruction's reads/writes can be flushed correctly.
        pending: dict[str, int] = {}
        result: list[BaseInst] = []

        for inst in block.insts:
            if self._is_increment_motion_barrier(inst):
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

                # Flush pending increments if their register (or any alias of
                # it) is read or written by this instruction.
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

    def _is_increment_motion_barrier(self, inst: BaseInst) -> bool:
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
