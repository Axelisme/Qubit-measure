"""Build numeric requests from pre-optimization compiled instructions."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import numpy as np

from zcu_tools.core.simulation import AcquireRequest, AcquisitionMode, ReadoutWindow

from ..base import MyProgramV2
from ..ir.instructions import (
    BaseInst,
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
from ._compiled_state import Machine
from ._compiled_stream import DecodedInstruction, NativeLoop, predecode
from .binding import SimulationBinding


def _execute(record: DecodedInstruction, machine: Machine, outputs: Outputs) -> None:
    inst = record.instruction
    if isinstance(inst, (MetaInst, LabelInst, NopInst)):
        return
    if isinstance(
        inst,
        (RegWriteInst, DmemReadInst, DmemWriteInst, WmemWriteInst, TestInst, TimeInst),
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
        f"{type(record.instruction).__name__} at preopt {record.position}, loops={record.loops}, source={record.source!r}: {cause}"
    )


class _OrderedExecution:
    def __init__(
        self,
        prog: MyProgramV2,
        records: tuple[DecodedInstruction, ...],
        loops: dict[str, NativeLoop],
        outputs: Outputs,
    ) -> None:
        self.records, self.loops, self.outputs = records, loops, outputs
        labels = {
            str(r.instruction.name): r.position
            for r in records
            if isinstance(r.instruction, LabelInst)
        }
        self.machine = Machine(prog, 1, labels)
        self.names = tuple(n for n in prog.loop_dict if n != "reps")
        self.dims = tuple(prog.loop_dict[n] for n in self.names)
        self.coordinates = dict.fromkeys(loops, 0)
        self.jumps: dict[int, NativeLoop] = {}
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

    def run(self) -> None:
        pc = 0
        while pc < len(self.records):
            record = self.records[pc]
            try:
                inst = record.instruction
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

    def _target(self, inst: JumpInst, position: int) -> int:
        if inst.label is None:
            return int(self.machine.value(inst.addr)[0])
        name = str(inst.label)
        if inst.label.is_pseudo():
            return position + {"HERE": 0, "NEXT": 1, "SKIP": 2, "PREV": -1}[name]
        return self.machine.labels[name]

    def _jump(self, inst: JumpInst, pc: int) -> int:
        record = self.records[pc]
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
            return loop.body_start + 1
        return pc + 1


def _live_inputs(
    records: Sequence[DecodedInstruction],
) -> tuple[set[str], set[str]] | None:
    defined: set[str] = set()
    inputs: set[str] = set()
    flags = False
    for record in records:
        inst = record.instruction
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
        reads, writes = set(inst.reg_read), set(inst.reg_write)
        # IR uses s14 as a scheduling barrier for reference changes. The
        # interpreter keeps that reference separately from the output-time word.
        if isinstance(inst, TimeInst):
            reads, writes = set(inst.r1.regs()) if inst.r1 is not None else set(), set()
        elif (
            isinstance(inst, (PortWriteInst, DportWriteInst)) and inst.time is not None
        ) or (isinstance(inst, RegWriteInst) and inst.src == SrcKeyword.WMEM):
            reads.discard("s14")
        inputs |= reads - defined
        defined |= writes
        flags |= isinstance(inst, TestInst) or bool(getattr(inst, "uf", False))
    return inputs, defined


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
        if not isinstance(inst, (RegWriteInst, TimeInst)) or (
            isinstance(inst, RegWriteInst) and inst.dst.canonical_name in live
        ):
            return False
    return True


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
    if not _safe_tail(
        records[end + 1 : min(loop.body_end for loop in loops.values())], inputs[0]
    ):
        return False
    labels = {
        str(r.instruction.name): r.position
        for r in records
        if isinstance(r.instruction, LabelInst)
    }
    names = tuple(n for n in prog.loop_dict if n != "reps")
    dims = tuple(prog.loop_dict[n] for n in names)
    size = prod(dims)
    grid = (
        np.indices(dims, dtype=np.int64).reshape(len(dims), size)
        if dims
        else np.empty((0, 1), dtype=np.int64)
    )
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
    outputs.finish(machine)
    return True


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
    records, loops = predecode(prog)
    for channel, config in prog.ro_chs.items():
        if config["trigs"] == 0:
            raise ValueError(
                f"ADC {channel}: zero-read declaration has no compiled triggers"
            )
    names = tuple(n for n in prog.loop_dict if n != "reps")
    dims = tuple(prog.loop_dict[n] for n in names)
    outputs = Outputs(prog, prod(dims), routes, binding.envelope_pool)
    if not _try_vector(prog, records, loops, outputs):
        _OrderedExecution(prog, records, loops, outputs).run()
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
