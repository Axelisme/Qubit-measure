from __future__ import annotations

from ..analysis import instruction_reads, instruction_writes
from ..instructions import (
    BaseInst,
    DmemReadInst,
    DmemWriteInst,
    Instruction,
    JumpInst,
    LabelInst,
    MetaInst,
    NopInst,
    PortWriteInst,
    RegWriteInst,
    TestInst,
    TimeInst,
    WaitInst,
    WmemWriteInst,
)
from ..labels import is_volatile_reg_name
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
        pending: dict[str, int] = {}  # reg -> index of last pending write
        dead: set[int] = set()

        for idx, inst in enumerate(insts):
            if self._is_write_tracking_barrier(inst):
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
            if getattr(inst, "uf", None) is not None:
                pending.clear()
                continue

            if len(writes) == 1:
                dst = writes[0]
                if is_volatile_reg_name(dst):
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
            inst, (TimeInst, WaitInst, RegWriteInst, DmemReadInst, NopInst)
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


from ..operands import AluExpr, Literal, Register


def _is_const_increment(inst: Instruction) -> tuple[str, int] | None:
    if not isinstance(inst, RegWriteInst):
        return None
    if inst.src != "op" or not isinstance(inst.op, AluExpr):
        return None
    if (
        inst.uf is not None
        or inst.if_cond is not None
        or inst.wr is not None
        or inst.label is not None
        or inst.addr is not None
    ):
        return None

    op = inst.op
    if op.op not in ("+", "-"):
        return None
    if not isinstance(op.rhs, Literal) or not op.rhs.value.startswith("#"):
        return None

    lhs_name = op.lhs.name
    if lhs_name != inst.dst.name:
        return None

    # Do not merge or move system registers (s0-s15) for safety, as they often
    # represent hardware state or IO that should remain at its original position.
    if lhs_name.startswith("s"):
        return None

    try:
        val = int(op.rhs.value[1:])
    except ValueError:
        return None

    if op.op == "-":
        val = -val
    return inst.dst.name, val


def _make_increment_inst(reg: str, val: int) -> RegWriteInst:
    if val >= 0:
        op = AluExpr(Register(reg), "+", Literal(f"#{val}"))
    else:
        op = AluExpr(Register(reg), "-", Literal(f"#{-val}"))
    return RegWriteInst(dst=Register(reg), src="op", op=op)


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
