"""Prove and apply closed-form native loop shells from their actual IR effects."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..ir.instructions import Instruction, LabelInst, RegWriteInst, TestInst
from ..ir.io_instructions import WmemWriteInst
from ..ir.operands import AluOp, Immediate, MemAddr, Register, SrcKeyword
from ._compiled_state import IntArray, Machine, signed32


@dataclass
class AffineUpdates:
    """Proven constant deltas for one native iteration, excluding its counter.

    registers maps canonical register names to signed increments. waves maps
    (WMEM word address, field 0..5) to increments. scratch holds overwritten
    shell temporaries, which cannot be live inputs to the vectorized body.
    """

    registers: dict[str, int] = field(default_factory=dict)
    waves: dict[tuple[int, int], int] = field(default_factory=dict)
    scratch: set[str] = field(default_factory=set)

    def apply(self, machine: Machine, index: IntArray) -> None:
        """Advance each independent lane by its zero-based native iteration."""
        for reg, delta in self.registers.items():
            machine.write(
                Register(reg),
                machine.value(Register(reg)) + delta * index,
                np.ones(machine.size, dtype=np.bool_),
            )
        for (address, field_index), delta in self.waves.items():
            value = machine.wmem[address][:, field_index] + delta * index
            width = (32, 32, 24, 32, 32, 16)[field_index]
            machine.wmem[address][:, field_index] = (
                signed32(value) if width == 32 else value % 2**width
            )


class _AffineParser:
    def __init__(self) -> None:
        self.updates = AffineUpdates()
        self.constants: dict[str, int] = {}
        self.address: int | None = None
        self.changes: dict[int, int] = {}

    def consume(self, inst: Instruction) -> bool:
        if isinstance(inst, (LabelInst, TestInst)):
            return True
        if isinstance(inst, WmemWriteInst):
            return self._store(inst)
        if (
            not isinstance(inst, RegWriteInst)
            or inst.wr
            or inst.uf
            or inst.if_cond
            or inst.ww
            or inst.wp
        ):
            return False
        return self._register(inst)

    def _register(self, inst: RegWriteInst) -> bool:
        dst = inst.dst.canonical_name
        if inst.src == SrcKeyword.LABEL and dst == "s15":
            return True
        if inst.src == SrcKeyword.IMM and inst.lit is not None:
            self.constants[dst] = inst.lit.value
            self.updates.scratch.add(dst)
            return True
        if (
            inst.src == SrcKeyword.WMEM
            and dst == "r_wave"
            and isinstance(inst.addr, MemAddr)
        ):
            if self.address is not None:
                return False
            self.address, self.changes = inst.addr.value, {}
            self.updates.scratch.update(f"w{i}" for i in range(6))
            return True
        return self._increment(inst)

    def _increment(self, inst: RegWriteInst) -> bool:
        expr = inst.op
        dst = inst.dst.canonical_name
        if (
            inst.src != SrcKeyword.OP
            or expr is None
            or expr.op != AluOp.ADD
            or expr.lhs.canonical_name != dst
        ):
            return False
        delta = (
            expr.rhs.value
            if isinstance(expr.rhs, Immediate)
            else self.constants.get(expr.rhs.canonical_name)
            if isinstance(expr.rhs, Register)
            else None
        )
        if delta is None:
            return False
        if dst.startswith("w"):
            if self.address is None:
                return False
            index = int(dst[1:])
            self.changes[index] = self.changes.get(index, 0) + delta
        else:
            self.updates.registers[dst] = self.updates.registers.get(dst, 0) + delta
        return True

    def _store(self, inst: WmemWriteInst) -> bool:
        if (
            not isinstance(inst.addr, MemAddr)
            or inst.addr.value != self.address
            or inst.wr
            or inst.op
            or inst.wp
            or inst.uf
            or inst.if_cond
        ):
            return False
        for index, delta in self.changes.items():
            key = (inst.addr.value, index)
            self.updates.waves[key] = self.updates.waves.get(key, 0) + delta
        self.address = None
        return True


def _deltas(instructions: Sequence[Instruction]) -> AffineUpdates | None:
    parser = _AffineParser()
    for inst in instructions:
        if not parser.consume(inst):
            return None
    return parser.updates if parser.address is None else None


def prove_native_shell(
    opening: Sequence[Instruction],
    increments: Sequence[Instruction],
    resets: Sequence[Instruction],
    *,
    counter: str,
    count: int,
) -> AffineUpdates | None:
    """Return affine effects only when initialization and resets prove independence.

    Unrecognized or nonlinear shells return None and require ordered execution.
    No macro type, waveform identity or arbitrary jump shape selects this path.
    """
    initializers = [inst for inst in opening if not isinstance(inst, LabelInst)]
    if len(initializers) != 1:
        return None
    init = initializers[0]
    if (
        not isinstance(init, RegWriteInst)
        or init.dst.canonical_name != counter
        or init.src != SrcKeyword.IMM
        or init.lit != Immediate(0)
        or init.wr
        or init.op
        or init.uf
        or init.if_cond
    ):
        return None
    step, reset = _deltas(increments), _deltas(resets)
    if step is None or reset is None:
        return None
    for key in step.registers.keys() | reset.registers.keys():
        if step.registers.get(key, 0) * count + reset.registers.get(key, 0) != 0:
            return None
    for key in step.waves.keys() | reset.waves.keys():
        if step.waves.get(key, 0) * count + reset.waves.get(key, 0) != 0:
            return None
    step.scratch |= reset.scratch
    return step
