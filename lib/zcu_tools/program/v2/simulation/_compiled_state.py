"""Lane-valued register and memory effects for the compiled frontend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..base import MyProgramV2
from ..ir.instructions import ArithInst, DivInst, RegWriteInst, TestInst, TimeInst
from ..ir.io_instructions import (
    DmemReadInst,
    DmemWriteInst,
    DportWriteInst,
    PortWriteInst,
    WmemWriteInst,
)
from ..ir.operands import (
    AluExpr,
    AluOp,
    Immediate,
    ImmValue,
    MemAddr,
    Register,
    SrcKeyword,
    TimeOffset,
)

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def signed32(value: IntArray) -> IntArray:
    """Return two's-complement signed 32-bit words in an int64 container."""
    return (value + 2**31) % 2**32 - 2**31


def _binary_word(op: AluOp, a: IntArray, b: IntArray) -> IntArray:
    binary = {
        AluOp.ADD: np.add,
        AluOp.SUB: np.subtract,
        AluOp.AND: np.bitwise_and,
        AluOp.AND_BIT: np.bitwise_and,
        AluOp.OR: np.bitwise_or,
        AluOp.OR_BIT: np.bitwise_or,
        AluOp.XOR: np.bitwise_xor,
        AluOp.XOR_BIT: np.bitwise_xor,
    }
    if op in binary:
        return signed32(binary[op](a, b))
    if op in (AluOp.CAT, AluOp.COLON2):
        return signed32(((a & 0xFFFF) << 16) | (b & 0xFFFF))
    if op == AluOp.ASR:
        return signed32(a >> (b & 15))
    if op in (AluOp.SL, AluOp.LSHIFT):
        return signed32(a << (b & 15))
    if op in (AluOp.SR, AluOp.RSHIFT):
        return signed32((a & 0xFFFFFFFF) >> (b & 15))
    raise ValueError(f"unsupported ALU operation {op.value}")


def _parity(value: IntArray) -> IntArray:
    parity = value & 0xFFFFFFFF
    for shift in (16, 8, 4, 2, 1):
        parity = parity ^ (parity >> shift)
    return parity & 1


type ModifiedInstruction = (
    RegWriteInst
    | DmemReadInst
    | DmemWriteInst
    | WmemWriteInst
    | PortWriteInst
    | DportWriteInst
)


@dataclass(frozen=True)
class _AluEffects:
    active: BoolArray
    result: IntArray
    side: IntArray | None


class Machine:
    """Mutable lane state, initialized from compiled memory, never shared across builds.

    size is the number of simultaneously evaluated lanes. Registers start undefined
    except s0. Missing reads and addresses outside initialized memory raise ValueError.
    labels maps pre-optimization label names to logical word positions.
    """

    def __init__(self, prog: MyProgramV2, size: int, labels: dict[str, int]) -> None:
        self.size = size
        self.labels = labels
        self.registers: dict[str, IntArray] = {"s0": np.zeros(size, dtype=np.int64)}
        self.arithmetic_result: IntArray | None = None
        self.flags: IntArray | None = None
        self.flags_valid = np.zeros(size, dtype=np.bool_)
        self.register_valid = {"s0": np.ones(size, dtype=np.bool_)}
        self.active = np.ones(size, dtype=np.bool_)
        self.read_mask = self.active.copy()
        self.ref = np.zeros(size, dtype=np.int64)
        self.wmem_capacity = int(prog.tproccfg["wmem_size"])
        self.wmem = {
            i: np.repeat(
                np.asarray(
                    [
                        int(w[f])
                        for f in ("freq", "phase", "env", "gain", "length", "conf")
                    ],
                    dtype=np.int64,
                )[None, :],
                size,
                axis=0,
            )
            for i, w in enumerate(prog.waves)
        }
        self.wmem_valid = {i: np.ones(size, dtype=np.bool_) for i in self.wmem}
        data = prog.compile_datamem()
        self.dmem_capacity = int(prog.tproccfg["dmem_size"])
        self.dmem = {
            i: np.full(size, int(word), dtype=np.int64)
            for i, word in enumerate([] if data is None else data)
        }
        self.dmem_valid = {i: np.ones(size, dtype=np.bool_) for i in self.dmem}
        self.writes: set[str] = set()

    def value(
        self, operand: Register | Immediate | ImmValue | MemAddr | TimeOffset | None
    ) -> IntArray:
        """Read an operand, raising for undefined, clock or peripheral registers."""
        if isinstance(operand, Register):
            name = operand.canonical_name
            if name == "s11":
                raise ValueError("tProc clock read s11 (s_usr_time)")
            if name in ("s6", "s7"):
                config = self.value(Register("s2"))
                if (
                    np.any(self.read_mask & ((config & 15) != 1))
                    or self.arithmetic_result is None
                ):
                    raise ValueError(
                        f"undefined core source for {name}; requires s_cfg source=arith"
                    )
                return signed32(self.arithmetic_result >> (32 if name == "s7" else 0))
            if name not in self.registers:
                if not np.any(self.read_mask):
                    return np.zeros(self.size, dtype=np.int64)
                raise ValueError(f"undefined register {name}")
            if np.any(self.read_mask & ~self.register_valid[name]):
                raise ValueError(f"undefined register {name}")
            return self.registers[name]
        if operand is None:
            raise ValueError("missing operand")
        return np.full(self.size, operand.value, dtype=np.int64)

    def expression(self, expr: AluExpr | None) -> IntArray:
        """Evaluate ALU input operands before any destination or side write."""
        if expr is None:
            raise ValueError("missing ALU expression")
        a = self.value(expr.lhs)
        b = (
            self.value(expr.rhs)
            if isinstance(expr.rhs, (Register, Immediate))
            else None
        )
        if expr.op == AluOp.NONE:
            return a.copy()
        unary = {
            AluOp.ABS: np.abs,
            AluOp.NOT: np.invert,
            AluOp.BANG: np.invert,
            AluOp.MSH: lambda word: (word & 0xFFFFFFFF) >> 16,
            AluOp.LSH: lambda word: word & 0xFFFF,
            AluOp.SWP: lambda word: (
                ((word & 0xFFFF) << 16) | ((word & 0xFFFFFFFF) >> 16)
            ),
            AluOp.PAR: _parity,
        }
        if expr.op in unary:
            return signed32(unary[expr.op](a))
        if b is None:
            raise ValueError(f"unsupported ALU operation {expr.op.value}")
        return _binary_word(expr.op, a, b)

    def condition(self, condition: str | None) -> BoolArray:
        """Return the active lanes selected by pre-instruction Z/S flags."""
        if condition is None:
            return self.active.copy()
        if self.flags is None or np.any(self.read_mask & ~self.flags_valid):
            raise ValueError("undefined ALU flags")
        return {
            "Z": self.flags == 0,
            "NZ": self.flags != 0,
            "S": self.flags < 0,
            "NS": self.flags >= 0,
        }[condition] & self.active

    def write(self, dst: Register, value: IntArray, active: BoolArray) -> None:
        """Apply a CPU write, preserving read-only SFRs and cfg/address widths."""
        name = dst.canonical_name
        if name in {"s0", "s1", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11"}:
            return
        if name == "s2":
            value = value & 0xFF
        elif name == "s15":
            value = value & 0xFFFF
        self._store_register(name, value, active)

    def _store_register(self, name: str, value: IntArray, active: BoolArray) -> None:
        old = self.registers.get(name, np.zeros(self.size, dtype=np.int64))
        self.registers[name] = np.where(active, signed32(value), old)
        self.register_valid[name] = (
            self.register_valid.get(name, np.zeros(self.size, dtype=np.bool_)) | active
        )
        self.writes.add(name)

    def address(self, operand: Register | MemAddr | None, length: int) -> IntArray:
        """Resolve initialized memory word addresses, rejecting missing/out-of-bounds."""
        address = self.value(operand)
        if np.any(self.read_mask & ((address < 0) | (address >= length))):
            raise ValueError(f"memory address out of bounds: {address.tolist()}")
        return address

    def wave(self, operand: Register | MemAddr | None) -> IntArray:
        """Copy current WMEM words, rejecting uninitialized per-lane addresses."""
        index = self.address(operand, self.wmem_capacity)
        result = np.empty((self.size, 6), dtype=np.int64)
        result.fill(0)
        for address in np.unique(index[self.read_mask]):
            selected = (index == address) & self.read_mask
            if int(address) not in self.wmem or not np.all(
                self.wmem_valid[int(address)][selected]
            ):
                raise ValueError(f"uninitialized WMEM[{address}]")
            result[selected] = self.wmem[int(address)][selected]
        return result

    def write_wave(self, operand: Register | MemAddr | None, active: BoolArray) -> None:
        """Store all wave registers, initializing newly written memory lanes."""
        index = self.address(operand, self.wmem_capacity)
        words = np.stack([self.value(Register(f"w{i}")) for i in range(6)], axis=1)
        for i, width in enumerate((32, 32, 24, 32, 32, 16)):
            words[:, i] &= 2**width - 1
        for address in np.unique(index[active]):
            selected = active & (index == address)
            if int(address) not in self.wmem:
                self.wmem[int(address)] = np.zeros((self.size, 6), dtype=np.int64)
                self.wmem_valid[int(address)] = np.zeros(self.size, dtype=np.bool_)
            self.wmem[int(address)][selected] = words[selected]
            self.wmem_valid[int(address)] |= selected
            self.writes.add(f"WMEM[{address}]")

    def _alu_effects(self, inst: ModifiedInstruction) -> _AluEffects:
        active = self.condition(inst.if_cond)
        self.read_mask = active
        literal = (
            inst.lit
            if isinstance(inst, (RegWriteInst, DmemReadInst, DmemWriteInst))
            else None
        )
        result = (
            self.expression(inst.op)
            if inst.op is not None
            else self.value(literal)
            if literal is not None
            else np.zeros(self.size, dtype=np.int64)
        )
        side = None
        if inst.wr is not None:
            if inst.wr.src_type == "op":
                side = result
            elif inst.wr.src_type == "imm" and literal is not None:
                side = self.value(literal)
            else:
                raise ValueError("missing/unknown side-write source")
        return _AluEffects(active, result, side)

    def _commit_alu(self, inst: ModifiedInstruction, effects: _AluEffects) -> None:
        if effects.side is not None and inst.wr is not None:
            self.write(inst.wr.dst, effects.side, effects.active)
        if inst.uf:
            old = (
                np.zeros(self.size, dtype=np.int64)
                if self.flags is None
                else self.flags
            )
            self.flags = np.where(effects.active, effects.result, old)
            self.flags_valid |= effects.active
        self.read_mask = self.active.copy()

    def port_effects(self, inst: PortWriteInst | DportWriteInst) -> None:
        """Apply ALU/side-register/WW effects after sampling the port output.

        Outputs must be sampled first; this evaluates the same prewrite register
        state before committing the simultaneous side write and flag result.
        """
        effects = self._alu_effects(inst)
        if isinstance(inst, PortWriteInst) and inst.ww is not None:
            self.write_wave(inst.addr, effects.active)
        self._commit_alu(inst, effects)

    def execute(
        self,
        inst: RegWriteInst
        | DmemReadInst
        | DmemWriteInst
        | WmemWriteInst
        | TestInst
        | TimeInst
        | ArithInst
        | DivInst,
        position: int,
    ) -> None:
        """Apply compute/memory/time effects, sampling operands before side writes.

        Output-coupled variants require the caller to sample the port first.
        Undefined reads, unsupported effects and memory bounds raise ValueError.
        """
        if isinstance(inst, ArithInst):
            self._arithmetic(inst)
        elif isinstance(inst, DivInst):
            numerator = self.value(inst.num) & 0xFFFFFFFF
            denominator = self.value(inst.den) & 0xFFFFFFFF
            if np.any(self.active & (denominator == 0)):
                raise ValueError("DIV denominator is zero")
            denominator = np.where(self.active, denominator, 1)
            self._store_register("s4", numerator // denominator, self.active)
            self._store_register("s5", numerator % denominator, self.active)
        elif isinstance(inst, TimeInst):
            self._advance(inst)
        elif isinstance(inst, TestInst):
            result = self.expression(inst.op)
            self.flags = np.where(
                self.active, result, 0 if self.flags is None else self.flags
            )
            self.flags_valid |= self.active
        else:
            effects = self._alu_effects(inst)
            if isinstance(inst, RegWriteInst):
                self._reg_write(inst, effects.result, effects.active, position)
            elif isinstance(inst, DmemReadInst):
                self._read_dmem(inst, effects.active)
            elif isinstance(inst, DmemWriteInst):
                self._write_dmem(inst, effects)
            else:
                self.write_wave(inst.addr, effects.active)
            self._commit_alu(inst, effects)

    def _arithmetic(self, inst: ArithInst) -> None:
        # This bounded signed model does not claim DSP internal bit-exactness.
        operands = [
            self.value(r) for r in (inst.r1, inst.r2, inst.r3, inst.r4) if r is not None
        ]
        pre_adder = inst.c_op.startswith(("P", "M"))
        offset = int(pre_adder)
        a = (operands[offset] + 2**26) % 2**27 - 2**26
        b = (operands[offset + 1] + 2**17) % 2**18 - 2**17
        if pre_adder:
            d = (operands[0] + 2**26) % 2**27 - 2**26
            a = d + a if inst.c_op.startswith("P") else d - a
            if np.any(self.active & ((a < -(2**26)) | (a >= 2**26))):
                raise ValueError("ARITH pre-adder exceeds signed 27-bit range")
        result = a * b
        if len(operands) > offset + 2:
            c = signed32(operands[offset + 2])
            result = result + c if inst.c_op.endswith("P") else result - c
        # |P| <= 2**26 * 2**17 + 2**31 < 2**45 after the pre-adder check.
        self.arithmetic_result = np.where(
            self.active,
            result,
            0 if self.arithmetic_result is None else self.arithmetic_result,
        )
        self._store_register("s3", result, self.active)

    def _advance(self, inst: TimeInst) -> None:
        if inst.c_op == "rst":
            self.ref[self.active] = 0
            return
        value = self.value(inst.lit or inst.r1)
        if inst.c_op == "inc_ref":
            self.ref += np.where(self.active, value, 0)
        elif inst.c_op == "set_ref":
            self.ref = np.where(self.active, value, self.ref)
        else:
            raise ValueError(f"clock-dependent TIME {inst.c_op}")

    def _read_dmem(self, inst: DmemReadInst, active: BoolArray) -> None:
        index = self.address(inst.addr, self.dmem_capacity)
        value = np.zeros(self.size, dtype=np.int64)
        for address in np.unique(index[active]):
            selected = active & (index == address)
            if int(address) not in self.dmem or not np.all(
                self.dmem_valid[int(address)][selected]
            ):
                raise ValueError(f"uninitialized DMEM[{address}]")
            value[selected] = self.dmem[int(address)][selected]
        self.write(inst.dst, value, active)

    def _write_dmem(self, inst: DmemWriteInst, effects: _AluEffects) -> None:
        index = self.address(inst.dst, self.dmem_capacity)
        value = self.value(inst.lit) if inst.src == "imm" else effects.result
        for address in np.unique(index[effects.active]):
            selected = effects.active & (index == address)
            if int(address) not in self.dmem:
                self.dmem[int(address)] = np.zeros(self.size, dtype=np.int64)
                self.dmem_valid[int(address)] = np.zeros(self.size, dtype=np.bool_)
            self.dmem[int(address)][selected] = signed32(value[selected])
            self.dmem_valid[int(address)] |= selected
            self.writes.add(f"DMEM[{address}]")

    def _reg_write(
        self, inst: RegWriteInst, alu: IntArray | None, active: BoolArray, position: int
    ) -> None:
        if inst.src == SrcKeyword.WMEM and inst.dst.canonical_name == "r_wave":
            words = self.wave(inst.addr)
            if inst.ww is not None:
                self.write_wave(inst.addr, active)
            for i in range(6):
                self.write(Register(f"w{i}"), words[:, i], active)
            return
        if inst.src == SrcKeyword.IMM:
            value = self.value(inst.lit)
        elif inst.src == SrcKeyword.OP:
            value = alu
        elif isinstance(inst.src, Register):
            value = self.value(inst.src)
        elif inst.src == SrcKeyword.LABEL and inst.label is not None:
            name = str(inst.label)
            address = (
                position + {"HERE": 0, "NEXT": 1, "SKIP": 2, "PREV": -1}[name]
                if inst.label.is_pseudo()
                else self.labels[name]
            )
            value = np.full(self.size, address, dtype=np.int64)
        else:
            raise ValueError("missing register source")
        if value is None:
            raise ValueError("missing register source")
        self.write(inst.dst, value, active)
