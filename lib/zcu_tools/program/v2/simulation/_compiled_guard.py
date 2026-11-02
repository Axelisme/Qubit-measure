"""Static shot live-in analysis, including structured control and memory aliases."""

from __future__ import annotations

from ..ir.instructions import ArithInst, BaseInst, MetaInst, RegWriteInst, TimeInst
from ..ir.io_instructions import (
    DmemReadInst,
    DmemWriteInst,
    DportWriteInst,
    PortWriteInst,
    TrigInst,
    WmemWriteInst,
)
from ..ir.operands import AluExpr, AluOp, Immediate, MemAddr, Register, SrcKeyword
from ._compiled_state import Machine
from ._compiled_stream import DecodedInstruction, NativeLoop, StructuredBranch


def _register_effects(inst: BaseInst) -> tuple[set[str], set[str]]:
    reads, writes = set(inst.reg_read), set(inst.reg_write)
    # IR s14 barriers describe reference changes, not reads of the output word.
    if isinstance(inst, TimeInst):
        return set(inst.r1.regs()) if inst.r1 is not None else set(), set()
    if (
        isinstance(inst, (PortWriteInst, DportWriteInst, TrigInst))
        and inst.time is not None
    ) or (
        isinstance(inst, RegWriteInst)
        and inst.src == SrcKeyword.WMEM
        and inst.wp is None
    ):
        reads.discard("s14")
    if isinstance(inst, WmemWriteInst) and inst.wp is None:
        reads.discard("s14")
    if reads & {"s6", "s7"}:
        reads.add("s2")
    if isinstance(inst, ArithInst):
        writes |= {"s6", "s7"}
    return reads, writes


type _AddressValue = int | str | tuple[int, int, str]


def _address_value(
    operand: Register | Immediate, values: dict[str, _AddressValue]
) -> _AddressValue:
    return (
        values.get(operand.canonical_name, f"?{operand.canonical_name}")
        if isinstance(operand, Register)
        else operand.value
    )


def _address_expression(
    expr: AluExpr, values: dict[str, _AddressValue]
) -> _AddressValue:
    left = _address_value(expr.lhs, values)
    if expr.op == AluOp.NONE:
        return left
    if not isinstance(expr.rhs, (Register, Immediate)) or expr.op not in (
        AluOp.ADD,
        AluOp.SUB,
    ):
        return "?"
    right = _address_value(expr.rhs, values)
    if isinstance(left, str) or isinstance(right, str):
        return "?"
    a, b = (left, left) if isinstance(left, int) else left[:2]
    c, d = (right, right) if isinstance(right, int) else right[:2]
    low, high = (a + c, b + d) if expr.op == AluOp.ADD else (a - d, b - c)
    if low < -(2**31) or high >= 2**31:
        return "?"
    if low == high:
        return low
    return low, high, f"({left}{expr.op.value}{right})"


def _memory_effects(
    inst: BaseInst,
    constants: dict[str, _AddressValue],
    ranges: dict[str, tuple[int, int] | None],
) -> tuple[set[str], set[str]]:
    def address(operand: Register | MemAddr | None) -> str:
        if isinstance(operand, MemAddr):
            return str(operand.value)
        if isinstance(operand, Register):
            name = operand.canonical_name
            return str(constants.get(name, f"?{name}"))
        return "?"

    reads: set[str] = set()
    writes: set[str] = set()
    if isinstance(inst, DmemReadInst):
        reads.add(f"DMEM[{address(inst.addr)}]")
    elif isinstance(inst, DmemWriteInst):
        writes.add(f"DMEM[{address(inst.dst)}]")
    if (
        isinstance(inst, RegWriteInst)
        and inst.src == SrcKeyword.WMEM
        or isinstance(inst, PortWriteInst)
        and inst.src == SrcKeyword.WMEM
    ):
        reads.add(f"WMEM[{address(inst.addr)}]")
    if (
        isinstance(inst, WmemWriteInst)
        or isinstance(inst, (RegWriteInst, PortWriteInst))
        and inst.ww is not None
    ):
        writes.add(f"WMEM[{address(inst.addr)}]")
    for name in reads | writes:
        operand = (
            inst.dst if isinstance(inst, DmemWriteInst) else getattr(inst, "addr", None)
        )
        value = (
            constants.get(operand.canonical_name, "?")
            if isinstance(operand, Register)
            else operand.value
            if isinstance(operand, MemAddr)
            else "?"
        )
        ranges[name] = (
            (value, value)
            if isinstance(value, int)
            else value[:2]
            if isinstance(value, tuple)
            else None
        )
    return reads, writes


def _address_writes(
    inst: BaseInst, index: int, values: dict[str, _AddressValue]
) -> None:
    value = (
        _address_expression(inst.op, values)
        if isinstance(inst, RegWriteInst) and inst.op is not None
        else "?"
    )
    for name in inst.reg_write:
        values[name] = f"?{name}@{index}"
    if isinstance(inst, RegWriteInst) and inst.if_cond is None:
        values[inst.dst.canonical_name] = (
            inst.lit.value
            if inst.src == SrcKeyword.IMM and inst.lit is not None
            else value
        )


def _memory_overlap(
    left: str, right: str, ranges: dict[str, tuple[int, int] | None]
) -> bool:
    if left not in ranges or right not in ranges or left[:4] != right[:4]:
        return False
    a, b = ranges[left], ranges[right]
    return a is None or b is None or max(a[0], b[0]) <= min(a[1], b[1])


class _LiveInAnalysis:
    """Track definite shot writes and possible entry reads across marked control."""

    def __init__(
        self,
        records: tuple[DecodedInstruction, ...],
        loops: dict[str, NativeLoop],
        branches: tuple[StructuredBranch, ...],
        constants: dict[str, _AddressValue],
    ) -> None:
        self.records = records
        self.branches = {branch.start: branch for branch in branches}
        self.loops = {
            loop.start: loop for loop in loops.values() if records[loop.start].body
        }
        self.constants = constants
        self.inputs: dict[str, DecodedInstruction] = {}
        self.written: set[str] = set()
        self.ranges: dict[str, tuple[int, int] | None] = {}

    def _branch(
        self,
        branch: StructuredBranch,
        defined: set[str],
        values: dict[str, _AddressValue],
    ) -> set[str]:
        if branch.compare not in defined:
            self.inputs.setdefault(branch.compare, self.records[branch.start])
        paths = []
        candidates = []
        for a, b in branch.cases:
            selected = values.copy()
            paths.append(self._walk(a + 1, b, defined.copy(), selected))
            candidates.append(selected)
        for name in {name for candidate in candidates for name in candidate}:
            first = candidates[0].get(name, "?")
            values[name] = (
                first
                if all(path.get(name, "?") == first for path in candidates)
                else f"?{name}@branch-{branch.start}"
            )
        return defined | set.intersection(*paths)

    def _loop(
        self, loop: NativeLoop, defined: set[str], values: dict[str, _AddressValue]
    ) -> set[str]:
        marker = self.records[loop.start].instruction
        if not isinstance(marker, MetaInst):
            raise ValueError("missing LOOP_START")
        info = marker.info
        if isinstance(info["n"], str) and info["n"] not in defined:
            self.inputs.setdefault(info["n"], self.records[loop.start])
        selected = values.copy()
        after = self._walk(
            loop.body_start + 1, loop.body_end, defined | {loop.counter}, selected
        )
        for name, value in selected.items():
            if values.get(name) != value:
                values[name] = f"?{name}@loop-{loop.start}"
        minimum = (
            info["n"]
            if isinstance(info["n"], int)
            else (info.get("range_hint") or (0, 0))[0]
        )
        return (defined | after if minimum > 0 else defined) | {loop.counter}

    def _instruction(
        self, index: int, defined: set[str], values: dict[str, _AddressValue]
    ) -> None:
        record = self.records[index]
        inst = record.instruction
        if not isinstance(inst, BaseInst):
            return
        reads, writes = _register_effects(inst)
        memory_reads, memory_writes = _memory_effects(inst, values, self.ranges)
        reads |= memory_reads
        writes |= memory_writes
        for name in reads - defined:
            self.inputs.setdefault(name, record)
        self.written.update(writes)
        _address_writes(inst, index, values)
        if getattr(inst, "if_cond", None) is None:
            defined |= writes

    def _walk(
        self, first: int, last: int, defined: set[str], values: dict[str, _AddressValue]
    ) -> set[str]:
        index = first
        while index < last:
            if index in self.branches:
                branch = self.branches[index]
                defined = self._branch(branch, defined, values)
                index = branch.end + 1
            elif index in self.loops:
                loop = self.loops[index]
                defined = self._loop(loop, defined, values)
                index = loop.end + 1
            else:
                self._instruction(index, defined, values)
                index += 1
        return defined

    def check(self, bounds: tuple[int, int], *, reject_carry: bool) -> set[str]:
        self._walk(*bounds, set(), self.constants.copy())
        carried = sorted(
            name
            for name in self.inputs
            if name in self.written
            or any(_memory_overlap(name, other, self.ranges) for other in self.written)
        )
        if carried and reject_carry:
            record = self.inputs[carried[0]]
            raise ValueError(
                f"{type(record.instruction).__name__} at preopt {record.position}, "
                f"loops={record.loops}, cases={record.cases}, source={record.source!r}: "
                f"cross-shot state live-in (register/memory): {', '.join(carried)}"
            )
        return set(self.inputs)


def guard_live_in(
    records: tuple[DecodedInstruction, ...],
    loops: dict[str, NativeLoop],
    branches: tuple[StructuredBranch, ...],
    bounds: tuple[int, int],
    seed: Machine,
    *,
    reject_carry: bool,
) -> set[str]:
    """Return shot entry dependencies; reject mutable ones across points/reps.

    records and boundaries are validated predecoded snapshot indexes. seed holds
    prologue-only resolved register words, used solely to disambiguate addresses,
    never as per-shot definitions. Sweep counters span 0..n-1. When reject_carry
    is true, any entry register/memory read that the shot can write raises
    ValueError with its instruction position and scopes, even for equal values.
    A single-shot request has no cross-shot boundary, but still returns entry
    dependencies for shell validation.
    """
    swept = {loop.counter for loop in loops.values()}
    for loop in loops.values():
        marker = records[loop.start].instruction
        if isinstance(marker, MetaInst):
            swept.update(marker.info["sweeps"]["registers"])
            swept.update(marker.info.get("register_tables", {}))
    constants: dict[str, _AddressValue] = {
        name: int(value[0])
        for name, value in seed.registers.items()
        if name not in swept
    }
    for loop in loops.values():
        marker = records[loop.start].instruction
        if not records[loop.start].body and isinstance(marker, MetaInst):
            constants[loop.counter] = (0, marker.info["n"] - 1, loop.counter)
    return _LiveInAnalysis(records, loops, branches, constants).check(
        bounds, reject_carry=reject_carry
    )
