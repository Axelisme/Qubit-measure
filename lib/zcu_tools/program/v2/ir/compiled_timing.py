"""Read-only scheduled-cycle analysis of linked tProc v2 assembly."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import prod
from operator import add, and_, or_, sub, xor
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import QickProgramV2

from .hw_semantics import WAVE_REGS
from .instructions import (
    BaseInst,
    JumpInst,
    NopInst,
    RegWriteInst,
    TestInst,
    TimeInst,
    WaitInst,
)
from .io_instructions import (
    DmemReadInst,
    DportWriteInst,
    PortWriteInst,
    TrigInst,
    WmemWriteInst,
)
from .labels import LabelRef
from .linker import IRLinker
from .operands import (
    AluExpr,
    AluOp,
    Immediate,
    ImmValue,
    MemAddr,
    Register,
    SrcKeyword,
    TimeOffset,
)

# Unsigned hardware fields in freq, phase, env, gain, length, conf order.
WaveWords = tuple[int, int, int, int, int, int]


def compiled_wmem(program: QickProgramV2) -> dict[int, WaveWords]:
    """Copy initial WMEM from a successfully compiled QICK v2 program.

    Keys are zero-based waveform addresses. Each value contains unsigned raw
    freq, phase, env, gain, length and conf words (32/32/24/32/32/16 bits).
    Swept fields use their compiled starting values. No physical units are
    decoded. Raises ValueError if the program has no compiled binary.
    """
    if not program.binprog:
        raise ValueError("Program must be compiled before reading WMEM")
    return {
        index: (
            int(wave["freq"]) % 2**32,
            int(wave["phase"]) % 2**32,
            int(wave["env"]) % 2**24,
            int(wave["gain"]) % 2**32,
            int(wave["length"]) % 2**32,
            int(wave["conf"]) % 2**16,
        )
        for index, wave in enumerate(program.waves)
    }


@dataclass(frozen=True)
class TimingLoop:
    """One outer-to-inner coordinate axis: unique name and positive shot count."""

    name: str
    count: int


@dataclass(frozen=True)
class TimingLayout:
    """Linked body boundaries and outer-to-inner shot axes (including reps).

    Addresses denote the point BEFORE executing an instruction. body_end may
    equal body_start for an empty body or the address just past the program.
    Loops label shot ordinals only; assembly determines actual control flow.
    """

    body_start: int
    body_end: int
    loops: tuple[TimingLoop, ...] = ()

    @classmethod
    def from_metadata(
        cls,
        meta_infos: Sequence[Mapping[str, object]],
        loops: tuple[TimingLoop, ...] = (),
    ) -> "TimingLayout":
        """Read final linked TIMING_BOUNDARY body_start/body_end meta records.

        The producer emits each marker once through the existing meta channel.
        Missing, duplicate or malformed boundaries raise ValueError.
        """
        boundaries: dict[str, int] = {}
        for meta in meta_infos:
            if meta.get("kind") != "meta" or meta.get("type") != "TIMING_BOUNDARY":
                continue
            name, address = meta.get("name"), meta.get("p_addr")
            if name not in ("body_start", "body_end"):
                raise ValueError(f"Unknown timing boundary: {name!r}")
            if not isinstance(address, int):
                raise ValueError(
                    "Timing boundaries require names and integer addresses"
                )
            if name in boundaries:
                raise ValueError(f"Duplicate timing boundary: {name}")
            boundaries[name] = address
        if boundaries.keys() != {"body_start", "body_end"}:
            raise ValueError("Missing body_start/body_end timing boundaries")
        return cls(boundaries["body_start"], boundaries["body_end"], loops)


@dataclass(frozen=True)
class CompiledTimingInput:
    """Public linked assembly, shot layout and optional compiled data memory.

    prog_list and labels are the final linked compiler outputs; layout identifies
    body boundaries and shot axes. dmem is the one-dimensional signed int32 word
    array returned by ProgramResourcesMixin.compile_datamem(), or its transported
    sequence of Python ints. Addresses are zero-based WORD offsets, not bytes.
    None and an empty sequence mean no data is available, not zero-filled memory.
    wmem optionally maps zero-based waveform addresses to six unsigned hardware
    words in freq, phase, env, gain, length, conf order, as from compiled_wmem().
    None leaves wave contents unknown; an empty mapping provides no waveforms.
    Analysis never mutates these inputs. Executed reads require a known in-range
    address and a signed int32 word; missing/malformed data, unknown/out-of-range
    addresses and unsupported memory effects (including writes) raise ValueError.
    """

    prog_list: Sequence[Mapping[str, object]]
    labels: Mapping[str, object]
    layout: TimingLayout
    dmem: Sequence[int] | NDArray[np.int32] | None = None
    wmem: Mapping[int, WaveWords] | None = None


@dataclass(frozen=True)
class TimingEvent:
    """One scheduled output, in execution order, not sorted by time.

    kind is wave (WPORT_WR, including config/segments), data (DPORT_WR) or
    trigger (TRIG). port is the
    physical tProc output port, NOT a logical generator/ADC channel. cycle is
    absolute from the initial reference origin 0. address identifies the linked
    instruction; value holds the data-port mask or trigger level, else None.
    wave_words holds six unsigned hardware fields in freq, phase, env, gain,
    length, conf order for wave outputs when initial WMEM was supplied; otherwise
    it is None. These words are not decoded into physical units.
    """

    kind: Literal["wave", "data", "trigger"]
    port: int
    cycle: int
    address: int
    value: int | None = None
    wave_words: WaveWords | None = None


@dataclass(frozen=True)
class ShotTiming:
    """One body execution and its shell, all reference times in tProc cycles.

    index is the execution ordinal. loop_indices follow layout.loops order.
    start_cycle is reference at body_start; body_end_cycle is reference before
    body_end. end_cycle is reference at next body_start or program termination,
    including final_wait/final_delay (WAIT does not advance reference).
    events includes body and following shell outputs, in execution order.
    """

    index: int
    loop_indices: tuple[int, ...]
    start_cycle: int
    body_end_cycle: int
    end_cycle: int
    events: tuple[TimingEvent, ...]


@dataclass(frozen=True)
class TimingAnalysis:
    """shots in execution order; initialization outputs are separate from shots.

    initial_events precede first body_start. end_cycle is final program reference.
    These are scheduled times, not CPU latency, hardware completion or pulse
    duration. No waveform lengths or module duration estimates are consulted.
    """

    shots: tuple[ShotTiming, ...]
    initial_events: tuple[TimingEvent, ...]
    end_cycle: int


@dataclass
class _Machine:
    labels: Mapping[str, object]
    dmem: Sequence[int] | NDArray[np.int32] | None
    wmem: dict[int, WaveWords] | None = None
    registers: dict[str, int | None] = field(default_factory=lambda: {"s0": 0})
    reference: int = 0
    flags: int | None = None

    def known_value(
        self, operand: Register | Immediate | ImmValue | TimeOffset
    ) -> int | None:
        if isinstance(operand, Register):
            if operand.canonical_name == "s11":
                raise ValueError("Unknown hardware time source: s11 (s_usr_time)")
            return self.registers.get(operand.canonical_name)
        return operand.value

    def value(self, operand: Register | Immediate | ImmValue | TimeOffset) -> int:
        value = self.known_value(operand)
        if value is None:
            raise ValueError(f"Unknown register value: {operand}")
        return value

    def expression(self, expr: AluExpr) -> int | None:
        lhs = self.known_value(expr.lhs)
        rhs = (
            self.known_value(expr.rhs)
            if isinstance(expr.rhs, (Register, Immediate))
            else None
        )
        if lhs is None or (expr.rhs is not None and rhs is None):
            return None
        if expr.rhs is None and expr.op in (AluOp.NONE, AluOp.ABS):
            if expr.op == AluOp.ABS:
                # tprocv2_assembler v3 rev23 encodes unary ABS as 1000.
                # Registers are signed 32-bit; the assembler does not specify the
                # minimum word's overflow result, so do not guess wrap or saturation.
                lhs = _signed32(lhs)
                if lhs == -(2**31):
                    raise ValueError("ABS input -2147483648 has unspecified overflow")
                lhs = -lhs if lhs < 0 else lhs
            return lhs
        binary = {
            AluOp.ADD: add,
            AluOp.SUB: sub,
            AluOp.AND: and_,
            AluOp.AND_BIT: and_,
            AluOp.OR: or_,
            AluOp.OR_BIT: or_,
            AluOp.XOR: xor,
            AluOp.XOR_BIT: xor,
        }
        if rhs is None:
            raise ValueError(f"Unsupported timing ALU expression: {expr}")
        if expr.op in binary:
            return _signed32(binary[expr.op](lhs, rhs))
        if expr.op == AluOp.SR:
            return _signed32((lhs & 0xFFFFFFFF) >> (rhs & 15))
        if expr.op == AluOp.ASR:
            return _signed32(lhs >> (rhs & 15))
        if expr.op == AluOp.SL:
            return _signed32(lhs << (rhs & 15))
        raise ValueError(f"Unsupported timing ALU expression: {expr}")

    def condition(self, condition: str | None) -> bool:
        if condition in (None, "1"):
            return True
        if condition == "0":
            return False
        if self.flags is None:
            raise ValueError("Unknown condition flags")
        conditions = {
            "Z": self.flags == 0,
            "NZ": self.flags != 0,
            "S": self.flags < 0,
            "NS": self.flags >= 0,
        }
        if condition not in conditions:
            raise ValueError(f"Unsupported control condition: {condition}")
        return conditions[condition]

    def label_address(self, label: LabelRef, address: int) -> int:
        if label.is_pseudo():
            offsets = {"HERE": 0, "NEXT": 1, "SKIP": 2, "PREV": -1}
            return address + offsets[str(label)]
        name = label.as_label().name
        if name not in self.labels:
            raise ValueError(f"Unknown label: {name}")
        return IRLinker.parse_label_addr(name, self.labels[name])

    def register_write(self, inst: RegWriteInst | DmemReadInst, address: int) -> None:
        if not self.condition(inst.if_cond):
            return
        if isinstance(inst, DmemReadInst):
            self.read_dmem(inst)
            return
        if inst.wr or inst.ww or inst.wp or inst.port:
            raise ValueError("Unsupported register write side effects")
        if inst.src == SrcKeyword.WMEM and inst.dst.name == "r_wave":
            self.load_wave(inst)
            return
        if inst.src == SrcKeyword.IMM and inst.lit is not None:
            value = inst.lit.value
        elif inst.src == SrcKeyword.OP and inst.op is not None:
            value = self.expression(inst.op)
        elif inst.src == SrcKeyword.LABEL and inst.label is not None:
            value = self.label_address(inst.label, address)
        else:
            raise ValueError(f"Unsupported register write: {inst}")
        # The flag ALU reads prewrite operands, which may include the destination.
        if inst.uf:
            self.flags = self.register_flags(inst, value)
        self.registers[inst.dst.canonical_name] = (
            None if value is None else _signed32(value)
        )

    def register_flags(self, inst: RegWriteInst, value: int | None) -> int | None:
        """Evaluate the flag ALU independently of immediate destination writeback."""
        if inst.src == SrcKeyword.IMM and inst.op is not None:
            return self.expression(inst.op)
        return value

    def read_dmem(self, inst: DmemReadInst) -> None:
        if inst.wr or inst.op or inst.lit or inst.uf:
            raise ValueError("Unsupported dmem read side effects")
        if not isinstance(inst.addr, (Register, MemAddr)):
            raise ValueError("Missing dmem read address")
        address = (
            self.value(inst.addr)
            if isinstance(inst.addr, Register)
            else inst.addr.value
        )
        if self.dmem is None:
            raise ValueError("Missing compiled dmem for read")
        if isinstance(self.dmem, np.ndarray) and (
            self.dmem.ndim != 1 or self.dmem.dtype != np.int32
        ):
            raise ValueError("Compiled dmem must be a one-dimensional int32 array")
        if not 0 <= address < len(self.dmem):
            raise ValueError(f"Dmem word address out of range: {address}")
        word = self.dmem[address]
        if (
            isinstance(word, bool)
            or not isinstance(word, (int, np.integer))
            or not -(2**31) <= int(word) < 2**31
        ):
            raise ValueError(f"Dmem word is not signed int32 at address {address}")
        self.registers[inst.dst.canonical_name] = int(word)

    def jump(self, inst: JumpInst, address: int) -> int:
        if inst.uf:
            raise ValueError("Unsupported jump flag update")
        taken = self.condition(inst.if_cond)
        if inst.wr is not None:
            if inst.wr.src_type != "op" or inst.op is None:
                raise ValueError("Unsupported jump side write")
            self.registers[inst.wr.dst.canonical_name] = self.expression(inst.op)
        if not taken:
            return address + inst.addr_inc
        if inst.label is not None:
            return self.label_address(inst.label, address)
        if isinstance(inst.addr, Register):
            return self.value(inst.addr)
        raise ValueError("Missing jump target")

    def advance(self, inst: TimeInst) -> None:
        if inst.c_op != "inc_ref":
            source = " (hardware time s11)" if inst.c_op == "updt" else ""
            raise ValueError(f"Unsupported reference operation: {inst.c_op}{source}")
        operand = inst.lit if inst.lit is not None else inst.r1
        if operand is None:
            raise ValueError("Missing reference increment")
        self.reference += self.value(operand)

    def wave_address(self, operand: Register | MemAddr | None) -> int:
        if not isinstance(operand, (Register, MemAddr)):
            raise ValueError("Missing wave memory address")
        return self.value(operand) if isinstance(operand, Register) else operand.value

    def read_wave(self, operand: Register | MemAddr | None) -> WaveWords:
        address = self.wave_address(operand)
        if self.wmem is None or address not in self.wmem:
            raise ValueError(f"Missing compiled wmem at address {address}")
        words = self.wmem[address]
        if len(words) != 6 or any(isinstance(word, bool) for word in words):
            raise ValueError(
                f"Compiled wmem must contain six integer words at {address}"
            )
        return words

    def load_wave(self, inst: RegWriteInst) -> None:
        if self.wmem is None:
            # Without initial WMEM, retain the timing-only unknown-data behavior.
            for name in WAVE_REGS:
                self.registers.pop(name, None)
        else:
            for index, word in enumerate(self.read_wave(inst.addr)):
                self.registers[f"w{index}"] = _signed32(word)
        if inst.uf:
            self.flags = None

    def store_wave(self, inst: WmemWriteInst) -> None:
        taken = self.condition(inst.if_cond)
        if inst.time or inst.wr or inst.op or inst.uf or inst.wp:
            raise ValueError("Unsupported wave memory side effects")
        if taken and self.wmem is not None:
            self.read_wave(inst.addr)
            self.wmem[self.wave_address(inst.addr)] = self.wave_registers()

    def wave_registers(self) -> WaveWords:
        return (
            self.value(Register("w0")) % 2**32,
            self.value(Register("w1")) % 2**32,
            self.value(Register("w2")) % 2**24,
            self.value(Register("w3")) % 2**32,
            self.value(Register("w4")) % 2**32,
            self.value(Register("w5")) % 2**16,
        )

    def output(
        self, inst: PortWriteInst | DportWriteInst | TrigInst, address: int
    ) -> TimingEvent | None:
        if not isinstance(inst, TrigInst):
            if not self.condition(inst.if_cond):
                return None
            if inst.wr or inst.op or inst.uf:
                raise ValueError("Unsupported output side effects")
        kind: Literal["wave", "data", "trigger"] = "wave"
        value = None
        if isinstance(inst, DportWriteInst):
            kind, value = "data", self.value(inst.data)
        elif isinstance(inst, TrigInst):
            kind, value = "trigger", int(inst.src == "set")
        wave_words = None
        if isinstance(inst, PortWriteInst) and self.wmem is not None:
            wave_words = self.read_wave(inst.addr)
        offset = inst.time if inst.time is not None else Register("s14")
        return TimingEvent(
            kind,
            self.value(inst.dst),
            self.reference + self.value(offset),
            address,
            value,
            wave_words,
        )

    def execute(self, inst: BaseInst, address: int) -> int:
        following = address + inst.addr_inc
        if isinstance(inst, NopInst):
            return following
        if isinstance(inst, (RegWriteInst, DmemReadInst)):
            self.register_write(inst, address)
        elif isinstance(inst, TimeInst):
            self.advance(inst)
        elif isinstance(inst, TestInst):
            self.flags = self.expression(inst.op)
        elif isinstance(inst, JumpInst):
            return self.jump(inst, address)
        elif isinstance(inst, WaitInst):
            if inst.c_op != "time" or inst.time is None:
                raise ValueError("Unsupported wait control")
            self.value(inst.time)
            if isinstance(inst.addr, Register) and self.value(inst.addr) != address + 1:
                raise ValueError("WAIT target must be its internal polling jump")
        elif isinstance(inst, WmemWriteInst):
            self.store_wave(inst)
        elif not isinstance(inst, (PortWriteInst, DportWriteInst, TrigInst)):
            raise ValueError(f"Unsupported timing instruction: {inst}")
        return following


def _signed32(value: int) -> int:
    return (value + 2**31) % 2**32 - 2**31


def _coordinates(index: int, loops: tuple[TimingLoop, ...]) -> tuple[int, ...]:
    result: list[int] = []
    for loop in reversed(loops):
        result.append(index % loop.count)
        index //= loop.count
    return tuple(reversed(result))


@dataclass
class _Shots:
    layout: TimingLayout
    shots: list[ShotTiming] = field(default_factory=list)
    initial_events: list[TimingEvent] = field(default_factory=list)
    events: list[TimingEvent] = field(default_factory=list)
    start: int | None = None
    body_end: int | None = None

    def finish(self, reference: int) -> None:
        if self.start is None:
            return
        if self.body_end is None:
            raise ValueError("Shot did not execute body_end")
        index = len(self.shots)
        self.shots.append(
            ShotTiming(
                index,
                _coordinates(index, self.layout.loops),
                self.start,
                self.body_end,
                reference,
                tuple(self.events),
            )
        )

    def boundary(self, address: int, reference: int) -> None:
        if address == self.layout.body_start:
            self.finish(reference)
            self.start = reference
            self.body_end = None
            self.events = []
        if address == self.layout.body_end:
            if self.start is None or self.body_end is not None:
                raise ValueError("body_end outside a body execution")
            self.body_end = reference

    def event(self, event: TimingEvent) -> None:
        (self.initial_events if self.start is None else self.events).append(event)


def _decode(compiled: CompiledTimingInput) -> tuple[dict[int, BaseInst], int]:
    layout = compiled.layout
    if any(loop.count <= 0 for loop in layout.loops):
        raise ValueError("Loop counts must be positive")
    if len({loop.name for loop in layout.loops}) != len(layout.loops):
        raise ValueError("Timing loop names must be unique")
    decoded: dict[int, BaseInst] = {}
    for raw in compiled.prog_list:
        address = raw.get("P_ADDR")
        if not isinstance(address, int) or address in decoded:
            raise ValueError("Instructions require unique integer P_ADDR")
        decoded[address] = BaseInst.from_dict(dict(raw))
    if not decoded:
        raise ValueError("Empty compiled program")
    end_address = max(a + inst.addr_inc for a, inst in decoded.items())
    if layout.body_start not in decoded or layout.body_end not in {
        *decoded,
        end_address,
    }:
        raise ValueError("Body boundaries must be linked instruction addresses")
    return decoded, end_address


def analyze_timing(
    compiled: CompiledTimingInput, *, max_steps: int = 1_000_000
) -> TimingAnalysis:
    """Execute known assembly timing/control in-process without changing inputs.

    Reference starts at cycle 0, independently of the s14 output-offset register.
    Literal TIME fields override s14; absent TIME reads the known s14 offset.
    TIME inc_ref advances reference. WAIT time validates its operand but changes
    no reference. The terminal unconditional self-jump is program termination.
    Layout boundaries are observed before execution; loops only label shots.
    Supports inc_ref, known immediate/ALU/label register writes, read-only dmem
    loads from known word addresses, TEST, known jumps, timed port outputs and
    plain wave-memory updates. Dmem loads require compiled signed int32 words;
    missing/malformed data, unknown/out-of-range addresses and memory writes or
    read side effects raise ValueError. ALU supports copy,
    add/subtract, bitwise AND/OR/XOR, signed 32-bit unary ABS and hardware
    low-four-bit SR/ASR/SL shifts. ABS rejects -2147483648 because its overflow
    result is not specified by the assembler.
    With initial WMEM, wave loads, wave-register updates and WMEM_WR modify a
    private memory copy; wave output events carry the current six raw words.
    Without initial WMEM, wave data stays unknown until overwritten and cannot
    drive time/control.
    Hardware s11 reads (including Resync) raise an error naming the source.
    Unknown timing/control registers, unsupported instructions/side effects,
    malformed boundaries, shot-count mismatches and exceeding positive max_steps
    raise ValueError. Assembly decoding can also raise KeyError for missing fields.
    Output times describe scheduling, not elapsed CPU or hardware execution.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    decoded, end_address = _decode(compiled)
    machine = _Machine(
        compiled.labels,
        compiled.dmem,
        None if compiled.wmem is None else dict(compiled.wmem),
    )
    recorder = _Shots(compiled.layout)
    address = min(decoded)
    for _ in range(max_steps):
        recorder.boundary(address, machine.reference)
        if address == end_address:
            break
        if address not in decoded:
            raise ValueError(f"Control flow to non-instruction address: {address}")
        inst = decoded[address]
        if isinstance(inst, (PortWriteInst, DportWriteInst, TrigInst)):
            event = machine.output(inst, address)
            if event is not None:
                recorder.event(event)
        following = machine.execute(inst, address)
        if (
            following == address
            and isinstance(inst, JumpInst)
            and inst.if_cond is None
            and address + inst.addr_inc == end_address
        ):
            break
        address = following
    else:
        raise ValueError("Timing analysis exceeded max_steps")
    recorder.finish(machine.reference)
    if len(recorder.shots) != prod(loop.count for loop in compiled.layout.loops):
        raise ValueError("Executed shot count does not match layout loops")
    return TimingAnalysis(
        tuple(recorder.shots), tuple(recorder.initial_events), machine.reference
    )
