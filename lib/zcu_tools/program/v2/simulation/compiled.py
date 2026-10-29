"""Build numeric requests from pre-optimization compiled instructions."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import numpy as np

from zcu_tools.core.simulation import AcquireRequest, AcquisitionMode, ReadoutWindow

from ..base import MyProgramV2
from ..ir.instructions import (
    ArithInst,
    BaseInst,
    DivInst,
    JumpInst,
    LabelInst,
    MetaInst,
    NopInst,
    RegWriteInst,
    TestInst,
    TimeInst,
    WaitInst,
)
from ..ir.io_instructions import (
    DmemReadInst,
    DmemWriteInst,
    DportWriteInst,
    PortWriteInst,
    WmemWriteInst,
)
from ..ir.operands import ImmValue, Register, SrcKeyword
from ._compiled_affine import AffineUpdates, prove_native_shell
from ._compiled_output import Outputs
from ._compiled_state import IntArray, Machine
from ._compiled_stream import (
    DecodedInstruction,
    NativeLoop,
    StructuredBranch,
    predecode,
)
from .binding import SimulationBinding


def _execute(record: DecodedInstruction, machine: Machine, outputs: Outputs) -> None:
    inst = record.instruction
    if isinstance(inst, (MetaInst, LabelInst, NopInst)):
        return
    if isinstance(
        inst,
        (
            RegWriteInst,
            DmemReadInst,
            DmemWriteInst,
            WmemWriteInst,
            TestInst,
            TimeInst,
            ArithInst,
            DivInst,
        ),
    ):
        if isinstance(inst, (RegWriteInst, WmemWriteInst)) and inst.wp is not None:
            if inst.port is None:
                raise ValueError("missing coupled output port")
            source = SrcKeyword.WMEM if inst.wp == "wmem" else Register("r_wave")
            outputs.wave(
                machine,
                PortWriteInst(
                    dst=ImmValue(int(inst.port)),
                    src=source,
                    addr=inst.addr,
                    time=inst.time if isinstance(inst, WmemWriteInst) else None,
                ),
                record.position,
                record.source,
            )
        machine.execute(inst, record.position)
    elif isinstance(inst, PortWriteInst):
        outputs.wave(machine, inst, record.position, record.source)
        machine.port_effects(inst)
    elif isinstance(inst, DportWriteInst):
        outputs.trigger(machine, inst)
        machine.port_effects(inst)
    elif not isinstance(inst, WaitInst) or record.body:
        raise ValueError("unsupported instruction effect")


def _failure(record: DecodedInstruction, cause: Exception) -> ValueError:
    return ValueError(
        f"{type(record.instruction).__name__} at preopt {record.position}, loops={record.loops}, cases={record.cases}, source={record.source!r}: {cause}"
    )


class _OrderedExecution:
    def __init__(
        self,
        prog: MyProgramV2,
        records: tuple[DecodedInstruction, ...],
        loops: dict[str, NativeLoop],
        outputs: Outputs,
        branches: tuple[StructuredBranch, ...],
    ) -> None:
        self.records, self.loops, self.outputs = records, loops, outputs
        outputs.loop_positions = {
            name: records[loop.start].position for name, loop in loops.items()
        }
        labels = {
            str(r.instruction.name): r.position
            for r in records
            if isinstance(r.instruction, LabelInst)
        }
        self.machine = Machine(prog, 1, labels)
        self.names = _point_names(records, loops)
        self.dims = tuple(prog.loop_dict[n] for n in self.names)
        self.coordinates: dict[str, int] = {}
        self.branch_jumps: dict[int, tuple[StructuredBranch, bool]] = {}
        self.selected: dict[str, int] = {}
        self.branch_starts = {branch.start: branch for branch in branches}
        for branch in branches:
            self._branch_shell(branch)
        self.jumps: dict[int, NativeLoop] = {}
        self.guards: dict[int, NativeLoop] = {}
        for loop in loops.values():
            closing = [
                i
                for i in range(loop.body_end + 1, loop.end)
                if isinstance(records[i].instruction, JumpInst)
            ]
            if len(closing) != 1:
                raise ValueError(
                    f"JumpInst at preopt native loop {loop.name}: missing/ambiguous backedge"
                )
            self.jumps[closing[0]] = loop
            if not loop.native:
                for index in range(loop.start + 1, loop.body_start):
                    if isinstance(records[index].instruction, JumpInst):
                        self.guards[index] = loop

    def run(self) -> None:
        pc = 0
        while pc < len(self.records):
            record = self.records[pc]
            try:
                inst = record.instruction
                if pc in self.branch_starts:
                    self._select(self.branch_starts[pc])
                if isinstance(inst, MetaInst):
                    self._marker(inst)
                elif isinstance(inst, JumpInst):
                    pc = self._jump(inst, pc)
                    continue
                else:
                    _execute(record, self.machine, self.outputs)
            except (ValueError, KeyError, IndexError) as exc:
                raise _failure(record, exc) from exc
            pc += 1

    def _marker(self, inst: MetaInst) -> None:
        if inst.type == "LOOP_START":
            self.coordinates[inst.name] = 0
        elif inst.type == "LOOP_BODY_START":
            if self.records[self.loops[inst.name].start].body:
                self._iterations()
        elif inst.type == "LOOP_END":
            self.coordinates.pop(inst.name, None)
            self._iterations()
        elif inst.type == "TIMING_BOUNDARY":
            if inst.name == "body_start":
                coordinates = tuple(self.coordinates[n] for n in self.names)
                point = (
                    int(np.ravel_multi_index(coordinates, self.dims))
                    if self.dims
                    else 0
                )
                self.outputs.begin(self.machine, np.asarray([point], dtype=np.int64))
            else:
                self.outputs.finish(self.machine)

    def _iterations(self) -> None:
        self.outputs.iterations = tuple(
            (name, index)
            for name, index in self.coordinates.items()
            if self.records[self.loops[name].start].body
        )

    def _target(self, inst: JumpInst, position: int) -> int:
        if inst.label is None:
            return int(self.machine.value(inst.addr)[0])
        name = str(inst.label)
        if inst.label.is_pseudo():
            return position + {"HERE": 0, "NEXT": 1, "SKIP": 2, "PREV": -1}[name]
        return self.machine.labels[name]

    def _jump(self, inst: JumpInst, pc: int) -> int:
        record = self.records[pc]
        if pc in self.branch_jumps:
            return self._branch_jump(inst, pc)
        if pc in self.guards:
            return self._inner_jump(inst, pc, self.guards[pc], guard=True)
        if pc in self.jumps and not self.jumps[pc].native:
            return self._inner_jump(inst, pc, self.jumps[pc], guard=False)
        if pc not in self.jumps:
            if record.body or record.loops or pc != len(self.records) - 2:
                raise ValueError("unmarked arbitrary jump")
            if (
                self._target(inst, record.position) != record.position
                or inst.if_cond
                or inst.wr
                or inst.op
            ):
                raise ValueError("unsupported program terminal jump")
            return pc + 1
        loop = self.jumps[pc]
        taken = self.machine.condition(inst.if_cond)
        result = (
            self.machine.expression(inst.op) if inst.wr is not None or inst.uf else None
        )
        target = self._target(inst, record.position)
        if target != self.machine.labels[loop.name]:
            raise ValueError("native loop backedge does not target its marked loop")
        if result is not None and inst.wr is not None:
            self.machine.write(inst.wr.dst, result, np.ones(1, dtype=np.bool_))
        if inst.uf:
            self.machine.flags = result
        if taken[0]:
            self.coordinates[loop.name] += 1
            self._iterations()
            return loop.body_start + 1
        return pc + 1

    def _branch_shell(self, branch: StructuredBranch) -> None:
        header = range(branch.start + 1, branch.cases[0][0])
        for index in header:
            if isinstance(self.records[index].instruction, JumpInst):
                self.branch_jumps[index] = (branch, True)
        for case_index, (_, end) in enumerate(branch.cases):
            next_start = (
                branch.cases[case_index + 1][0]
                if case_index + 1 < len(branch.cases)
                else branch.end
            )
            for index in range(end + 1, next_start):
                if isinstance(self.records[index].instruction, JumpInst):
                    self.branch_jumps[index] = (branch, False)

    def _select(self, branch: StructuredBranch) -> None:
        selected = int(self.machine.value(Register(branch.compare))[0])
        if not 0 <= selected < len(branch.cases):
            raise ValueError(
                f"branch {branch.name} selector {selected} has no marked case"
            )
        self.selected[branch.name] = selected

    def _branch_jump(self, inst: JumpInst, pc: int) -> int:
        branch, header = self.branch_jumps[pc]
        if inst.wr is not None or inst.uf or inst.op is not None:
            raise ValueError("unsupported branch jump side effects")
        if not self.machine.condition(inst.if_cond)[0]:
            return pc + 1
        target = self._target(inst, self.records[pc].position)
        if header:
            case_start = branch.cases[self.selected[branch.name]][0]
            if target == self.records[case_start].position:
                return case_start
            # A marked dispatch table may contain forward-only jump stubs.
            for index in range(pc + 1, branch.cases[0][0]):
                if self.records[index].position == target:
                    return index
        elif target == self.records[branch.end].position:
            return branch.end
        raise ValueError(f"branch {branch.name} jump escapes its selected marked case")

    def _inner_jump(
        self, inst: JumpInst, pc: int, loop: NativeLoop, *, guard: bool
    ) -> int:
        boundary = loop.end if guard else loop.body_start
        if (
            self._target(inst, self.records[pc].position)
            != self.records[boundary].position
        ):
            raise ValueError("structured loop jump escapes its marked boundary")
        if inst.wr is not None or inst.uf or inst.op is not None:
            raise ValueError("unsupported structured loop jump side effects")
        if not self.machine.condition(inst.if_cond)[0]:
            return pc + 1
        if not guard:
            self.coordinates[loop.name] = int(
                self.machine.value(Register(loop.counter))[0]
            )
            self._iterations()
        return boundary


def _register_effects(inst: BaseInst) -> tuple[set[str], set[str]]:
    reads, writes = set(inst.reg_read), set(inst.reg_write)
    # IR s14 barriers describe reference changes, not reads of the output word.
    if isinstance(inst, TimeInst):
        return set(inst.r1.regs()) if inst.r1 is not None else set(), set()
    if (
        isinstance(inst, (PortWriteInst, DportWriteInst)) and inst.time is not None
    ) or (isinstance(inst, RegWriteInst) and inst.src == SrcKeyword.WMEM):
        reads.discard("s14")
    if reads & {"s6", "s7"}:
        reads.add("s2")
    if isinstance(inst, ArithInst):
        writes |= {"s6", "s7"}
    return reads, writes


def _live_inputs(
    records: Sequence[DecodedInstruction],
) -> tuple[set[str], set[str]] | None:
    defined: set[str] = set()
    inputs: set[str] = set()
    written: set[str] = set()
    flags = False
    for record in records:
        inst = record.instruction
        if isinstance(inst, JumpInst):
            return None
        if not isinstance(inst, BaseInst):
            continue
        if (
            isinstance(inst, (DmemWriteInst, WmemWriteInst))
            or (isinstance(inst, RegWriteInst) and (inst.ww or inst.wp))
            or (isinstance(inst, PortWriteInst) and inst.ww)
        ):
            return None
        if getattr(inst, "if_cond", None) is not None and not flags:
            return None
        reads, writes = _register_effects(inst)
        inputs |= reads - defined
        written |= writes
        if getattr(inst, "if_cond", None) is None:
            defined |= writes
            flags |= isinstance(inst, TestInst) or bool(getattr(inst, "uf", False))
    return inputs, written


def _counter_jump(jump: JumpInst, counter: str) -> bool:
    return (
        jump.if_cond == "NZ"
        and jump.wr is not None
        and jump.wr.dst.canonical_name == counter
        and jump.wr.src_type == "op"
        and not jump.uf
        and jump.op is not None
        and str(jump.op) == f"{counter} + #1"
    )


def _shell(
    prog: MyProgramV2, records: tuple[DecodedInstruction, ...], loop: NativeLoop
) -> AffineUpdates | None:
    closing = [
        i
        for i in range(loop.body_end + 1, loop.end)
        if isinstance(records[i].instruction, JumpInst)
    ]
    if len(closing) != 1:
        return None
    backedge = closing[0]
    jump = records[backedge].instruction
    if not isinstance(jump, JumpInst) or not _counter_jump(jump, loop.counter):
        return None
    test = records[backedge - 1].instruction
    count = prog.loop_dict[loop.name]
    if (
        not isinstance(test, TestInst)
        or str(test.op) != f"{loop.counter} - #{count - 1}"
    ):
        return None
    labels = [
        r.instruction
        for r in records[loop.body_end + 1 : backedge]
        if isinstance(r.instruction, RegWriteInst)
        and r.instruction.dst.canonical_name == "s15"
    ]
    target = (
        str(jump.label)
        if jump.label is not None
        else str(labels[-1].label)
        if labels
        else ""
    )
    if target != loop.name:
        return None
    return prove_native_shell(
        [r.instruction for r in records[loop.start + 1 : loop.body_start]],
        [r.instruction for r in records[loop.body_end + 1 : backedge]],
        [r.instruction for r in records[backedge + 1 : loop.end]],
        counter=loop.counter,
        count=count,
    )


def _vector_shells(
    prog: MyProgramV2,
    records: tuple[DecodedInstruction, ...],
    loops: dict[str, NativeLoop],
    inputs: tuple[set[str], set[str]],
) -> dict[str, AffineUpdates] | None:
    if any(records[loop.start].body for loop in loops.values()):
        return None
    if inputs[0] & inputs[1]:
        return None
    shells: dict[str, AffineUpdates] = {}
    for name, loop in loops.items():
        shell = _shell(prog, records, loop)
        if shell is None or shell.scratch & inputs[0] or loop.counter in inputs[1]:
            return None
        shells[name] = shell
    reps = shells["reps"]
    if reps.registers or reps.waves or loops["reps"].counter in inputs[0]:
        return None
    return shells


def _safe_tail(records: Sequence[DecodedInstruction], live: set[str]) -> bool:
    for record in records:
        inst = record.instruction
        if isinstance(inst, (MetaInst, WaitInst)):
            continue
        if not isinstance(inst, (RegWriteInst, TimeInst)):
            return False
        if isinstance(inst, RegWriteInst) and (
            set(inst.reg_write) & live or inst.ww or inst.wp
        ):
            return False
    return True


def _canonical_terminal(records: Sequence[DecodedInstruction]) -> bool:
    instructions = [
        r.instruction for r in records if not isinstance(r.instruction, MetaInst)
    ]
    if len(instructions) != 2:
        return False
    load, jump = instructions
    return (
        isinstance(load, RegWriteInst)
        and load.dst.canonical_name == "s15"
        and load.src == SrcKeyword.LABEL
        and str(load.label) == "NEXT"
        and not any((load.if_cond, load.wr, load.op, load.uf, load.ww, load.wp))
        and isinstance(jump, JumpInst)
        and isinstance(jump.addr, Register)
        and jump.addr.canonical_name == "s15"
        and jump.label is None
        and not any((jump.if_cond, jump.wr, jump.op, jump.uf))
    )


def _point_names(
    records: tuple[DecodedInstruction, ...], loops: dict[str, NativeLoop]
) -> tuple[str, ...]:
    return tuple(
        name
        for name, loop in loops.items()
        if name != "reps" and not records[loop.start].body
    )


def _native_grid(prog: MyProgramV2) -> tuple[tuple[str, ...], IntArray]:
    names = tuple(n for n in prog.loop_dict if n != "reps")
    dims = tuple(prog.loop_dict[n] for n in names)
    grid = (
        np.indices(dims, dtype=np.int64).reshape(len(dims), prod(dims))
        if dims
        else np.empty((0, 1), dtype=np.int64)
    )
    return names, grid


def _try_vector(
    prog: MyProgramV2,
    records: tuple[DecodedInstruction, ...],
    loops: dict[str, NativeLoop],
    outputs: Outputs,
) -> bool:
    boundaries = [
        i
        for i, r in enumerate(records)
        if isinstance(r.instruction, MetaInst)
        and r.instruction.type == "TIMING_BOUNDARY"
    ]
    start, end = boundaries
    body = records[start + 1 : end]
    inputs = _live_inputs(body)
    shells = (
        _vector_shells(prog, records, loops, inputs) if inputs is not None else None
    )
    if shells is None or inputs is None:
        return False
    tail = records[end + 1 : min(loop.body_end for loop in loops.values())]
    if not _safe_tail(tail, inputs[0]) or not _canonical_terminal(
        records[max(loop.end for loop in loops.values()) + 1 :]
    ):
        return False
    labels = {
        str(r.instruction.name): r.position
        for r in records
        if isinstance(r.instruction, LabelInst)
    }
    names, grid = _native_grid(prog)
    size = outputs.size
    machine = Machine(prog, size, labels)
    for record in records[:start]:
        if not isinstance(
            record.instruction, (RegWriteInst, TimeInst, MetaInst, LabelInst, NopInst)
        ):
            return False
        _located_execute(record, machine, outputs)
    for name, indexes in zip(names, grid, strict=True):
        shells[name].apply(machine, indexes)
        machine.registers[loops[name].counter] = indexes.copy()
    outputs.begin(machine, np.arange(size, dtype=np.int64))
    for record in body:
        _located_execute(record, machine, outputs)
    # Speculative lanes lack cross-point config state. Prove independence before
    # resolving reads; otherwise only ordered execution can validate their windows.
    if not outputs.configs_independent(machine):
        return False
    try:
        outputs.finish(machine)
    except (ValueError, KeyError, IndexError) as exc:
        raise _failure(records[end], exc) from exc
    for record in tail:
        _located_execute(record, machine, outputs)
    return outputs.configs_independent(machine)


def _located_execute(
    record: DecodedInstruction, machine: Machine, outputs: Outputs
) -> None:
    try:
        _execute(record, machine, outputs)
    except (ValueError, KeyError, IndexError) as exc:
        raise _failure(record, exc) from exc


def build_compiled_request(
    prog: MyProgramV2,
    mode: AcquisitionMode,
    binding: SimulationBinding,
    sequence: int,
    *,
    routes: Sequence[tuple[int, int]],
) -> AcquireRequest:
    """Detach compiled instructions into numeric slots without calling a backend.

    prog is a compiled native or Modular v2 program. routes is the SimulationSoc's
    explicit (generator, ADC) wiring, not program declarations. mode chooses sums
    or traces; binding and sequence supply sampling identity. Unsupported effects,
    malformed markers, ambiguous routing and rep-varying outputs raise ValueError
    with instruction type, pre-optimization position and available source/loops.
    Acquisition still uses the separate Module frontend until its migration.
    """
    prog.compile()
    records, loops, branches = predecode(prog)
    for channel, config in prog.ro_chs.items():
        if config["trigs"] == 0:
            raise ValueError(
                f"ADC {channel}: zero-read declaration has no compiled triggers"
            )
    names = _point_names(records, loops)
    dims = tuple(prog.loop_dict[n] for n in names)
    outputs = Outputs(prog, prod(dims), routes, binding.envelope_pool)
    if not _try_vector(prog, records, loops, outputs):
        outputs = Outputs(prog, prod(dims), routes, binding.envelope_pool)
        _OrderedExecution(prog, records, loops, outputs, branches).run()
    template = outputs.template()
    windows = tuple(
        ReadoutWindow(
            s.channel,
            int(s.generator_channels[0, -1]),
            s.samples,
            s.sample_rate_mhz,
            tuple(float(x) for x in s.trigger_us[0]),
            tuple(float(x) for x in s.frequency_mhz[0]),
            tuple(float(x) for x in s.phase_deg[0]),
            tuple(int(x) for x in s.generator_channels[0]),
        )
        for s in template.readouts
    )
    if prog.loop_dims is None or prog.avg_level is None:
        raise ValueError("compiled acquisition layout is missing")
    return AcquireRequest(
        mode,
        tuple(prog.loop_dims),
        prog.avg_level,
        windows,
        binding.root_seed,
        sequence,
        binding.rng_version,
        dims,
        names,
        prog.loop_dict["reps"],
        (template,),
        prog.relax_delay,
    )
