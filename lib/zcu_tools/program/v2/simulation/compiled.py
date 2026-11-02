"""Build numeric requests from pre-optimization compiled instructions."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from math import prod
from typing import Any

import numpy as np

from zcu_tools.core.simulation import (
    AcquireRequest,
    AcquisitionMode,
    ChargeDriveRequirement,
    ReadoutWindow,
    TimeBase,
)

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
    TrigInst,
    WmemWriteInst,
)
from ..ir.operands import AluExpr, AluOp, ImmValue, Register, SrcKeyword
from ._compiled_guard import guard_live_in
from ._compiled_output import Outputs
from ._compiled_state import IntArray, Machine, signed32
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
    elif isinstance(inst, TrigInst):
        outputs.trigger(machine, inst)
    elif not isinstance(inst, WaitInst) or record.body:
        raise ValueError("unsupported instruction effect")


def _failure(record: DecodedInstruction, cause: Exception) -> ValueError:
    return ValueError(
        f"{type(record.instruction).__name__} at preopt {record.position}, loops={record.loops}, cases={record.cases}, source={record.source!r}: {cause}"
    )


def _point_names(
    records: tuple[DecodedInstruction, ...], loops: dict[str, NativeLoop]
) -> tuple[str, ...]:
    return tuple(
        name
        for name, loop in loops.items()
        if name != "reps" and not records[loop.start].body
    )


def _native_grid(prog: MyProgramV2, names: tuple[str, ...]) -> IntArray:
    dims = tuple(prog.loop_dict[n] for n in names)
    return (
        np.indices(dims, dtype=np.int64).reshape(len(dims), prod(dims))
        if dims
        else np.empty((0, 1), dtype=np.int64)
    )


def _loop_info(
    records: tuple[DecodedInstruction, ...], loop: NativeLoop
) -> dict[str, Any]:
    marker = records[loop.start].instruction
    if not isinstance(marker, MetaInst):
        raise ValueError("missing LOOP_START")
    return marker.info


def _validate_loop(records: tuple[DecodedInstruction, ...], loop: NativeLoop) -> None:
    """Check the marked counter and backedge, without inferring sweep values."""
    closing = [
        r
        for r in records[loop.body_end + 1 : loop.end]
        if isinstance(r.instruction, JumpInst)
    ]
    if len(closing) != 1:
        raise _failure(
            records[loop.start], ValueError("missing/ambiguous loop backedge")
        )
    record = closing[0]
    jump = record.instruction
    if not isinstance(jump, JumpInst):
        raise _failure(record, ValueError("missing loop backedge"))
    info = _loop_info(records, loop)
    n = info["n"]
    if loop.native:
        tests = [
            r.instruction
            for r in records[loop.body_end + 1 : loop.end]
            if isinstance(r.instruction, TestInst)
        ]
        if (
            len(tests) != 1
            or str(tests[0].op) != f"{loop.counter} - #{n - 1}"
            or jump.if_cond != "NZ"
            or jump.wr is None
            or jump.wr.dst.canonical_name != loop.counter
            or str(jump.op) != f"{loop.counter} + #1"
        ):
            raise _failure(
                record, ValueError("loop counter/count disagrees with marker")
            )
        labels = [
            r.instruction
            for r in records[loop.body_end + 1 : loop.end]
            if isinstance(r.instruction, RegWriteInst)
            and r.instruction.src == SrcKeyword.LABEL
        ]
        target = (
            str(jump.label)
            if jump.label is not None
            else str(labels[-1].label)
            if labels
            else ""
        )
        if target != loop.name:
            raise _failure(record, ValueError("loop backedge disagrees with marker"))
    else:
        _validate_inner_loop(records, loop, jump)


def _validate_inner_loop(
    records: tuple[DecodedInstruction, ...], loop: NativeLoop, jump: JumpInst
) -> None:
    n = _loop_info(records, loop)["n"]
    count = n if isinstance(n, str) else f"#{n}"
    closing = records[loop.body_end + 1 : loop.end]
    tests = [r.instruction for r in closing if isinstance(r.instruction, TestInst)]
    loads = [
        r.instruction
        for r in closing
        if isinstance(r.instruction, RegWriteInst)
        and r.instruction.src == SrcKeyword.LABEL
    ]
    target = (
        str(jump.label)
        if jump.label is not None
        else str(loads[-1].label)
        if loads
        else ""
    )
    header = records[loop.start + 1 : loop.body_start]
    labels = {
        str(r.instruction.name): r.position
        for r in header
        if isinstance(r.instruction, LabelInst)
    }
    initial = [
        r.instruction
        for r in header
        if isinstance(r.instruction, RegWriteInst)
        and r.instruction.dst.canonical_name == loop.counter
        and r.instruction.src == SrcKeyword.IMM
        and r.instruction.lit is not None
        and r.instruction.lit.value == 0
    ]
    if (
        len(tests) != 1
        or str(tests[0].op) != f"{loop.counter} - {count}"
        or jump.if_cond != "S"
        or jump.wr is not None
        or jump.uf
        or jump.op
        or labels.get(target) != records[loop.body_start].position
        or len(initial) != 1
    ):
        raise _failure(
            records[loop.start],
            ValueError("inner loop counter/count/backedge disagrees with marker"),
        )


def _validate_definition(info: dict[str, Any], machine: Machine) -> None:
    registers, waves = info["sweeps"]["registers"], info["sweeps"]["waves"]
    _check_table_wrap(info, machine)
    for name, definition in registers.items():
        if not np.all(
            machine.value(Register(name)) == signed32(np.array([definition["start"]]))
        ):
            raise ValueError(
                f"sweep initial register {name} disagrees with compiled word"
            )
        if not isinstance(definition["step"], int):
            raise ValueError("sweep step must be integer words")
    for address, fields in waves.items():
        for field, definition in fields.items():
            if not 0 <= field < 6 or not np.all(
                machine.wmem[address][:, field] % 2**32 == definition["start"] % 2**32
            ):
                raise ValueError("sweep initial waveform disagrees with compiled word")
            if not isinstance(definition["step"], int):
                raise ValueError("sweep step must be integer words")


def _shell_compute(record: DecodedInstruction, machine: Machine) -> None:
    inst = record.instruction
    if isinstance(inst, (MetaInst, LabelInst, NopInst, WaitInst)):
        return
    if not isinstance(
        inst, (RegWriteInst, DmemReadInst, WmemWriteInst, TestInst, TimeInst)
    ):
        raise _failure(record, ValueError("unsupported acquisition shell effect"))
    if isinstance(inst, (RegWriteInst, WmemWriteInst)) and inst.wp is not None:
        raise _failure(record, ValueError("output inside acquisition shell"))
    machine.execute(inst, record.position)


def _check_update(
    info: dict[str, Any],
    before: Machine,
    after: Machine,
    multiplier: int,
    scratch: set[str],
) -> None:
    registers, waves = info["sweeps"]["registers"], info["sweeps"]["waves"]
    allowed = set(registers) | set(info.get("register_tables", {})) | scratch
    allowed |= {f"WMEM[{address}]" for address in waves.keys() | info["tables"].keys()}
    if after.writes - allowed:
        raise ValueError(
            f"shell writes undeclared state: {sorted(after.writes - allowed)}"
        )
    for name, definition in registers.items():
        expected = signed32(
            before.value(Register(name)) + multiplier * definition["step"]
        )
        if np.any(after.value(Register(name)) != expected):
            raise ValueError(f"sweep update for {name} disagrees with shell")
    for name, definition in info.get("register_tables", {}).items():
        indexes = before.value(Register(info["counter_reg"]))
        expected = before.value(Register(name)).copy()
        for point in range(info["n"]):
            active = indexes == point
            expected[active] = before.dmem[definition["offset"] + point][active]
        if np.any(after.value(Register(name)) != expected):
            raise ValueError("register table update disagrees with shell")
    for address in waves.keys() | info["tables"].keys():
        expected = before.wmem[address].copy()
        for field, definition in waves.get(address, {}).items():
            expected[:, field] += multiplier * definition["step"]
            expected[:, field] %= 2 ** (16 if field == 5 else 32)
        indexes = before.value(Register(info["counter_reg"]))
        for field, offset in info["tables"].get(address, {}).items():
            for point in range(info["n"]):
                active = indexes == point
                expected[active, field] = before.dmem[offset + point][active] % 2**32
        if np.any(after.wmem[address] % 2**32 != expected % 2**32):
            raise ValueError("wave sweep/table update disagrees with shell")


def _validate_native_sweeps(
    prog: MyProgramV2,
    records: tuple[DecodedInstruction, ...],
    loops: dict[str, NativeLoop],
    seed: Machine,
) -> None:
    scratch = {"s15", *(f"w{i}" for i in range(6))}
    if "scratch" in prog.reg_dict:
        scratch.add(prog.reg_dict["scratch"].full_addr())
    for loop in loops.values():
        if not loop.native:
            continue
        info = _loop_info(records, loop)
        try:
            _validate_definition(info, seed)
            if loop.name == "reps" and any(
                (
                    info["sweeps"]["registers"],
                    info["sweeps"]["waves"],
                    info["tables"],
                    info.get("register_tables"),
                )
            ):
                raise ValueError("reps shell may only update its own counter")
            before = deepcopy(seed)
            indexes = (
                np.arange(seed.size, dtype=np.int64) % info["n"]
                if info["tables"] or info.get("register_tables")
                else np.zeros(seed.size, dtype=np.int64)
            )
            before.write(Register(loop.counter), indexes, before.active)
            _apply_tables(info, before, indexes)
            actual = deepcopy(before)
            actual.writes.clear()
            closing = records[loop.body_end + 1 : loop.end]
            jump_index = next(
                i for i, r in enumerate(closing) if isinstance(r.instruction, JumpInst)
            )
            for record in closing[:jump_index]:
                _shell_compute(record, actual)
            _check_update(info, before, actual, 1, scratch)
            actual.writes.clear()
            for record in closing[jump_index + 1 :]:
                _shell_compute(record, actual)
            _check_update(info, before, actual, -(info["n"] - 1), scratch)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise _failure(
                records[loop.start], ValueError(f"sweep shell disagrees: {exc}")
            ) from exc


def _timing_writes(
    records: Sequence[DecodedInstruction],
) -> dict[int, frozenset[str]]:
    """Map reaching tail write positions to registers feeding reference advance."""
    needed: set[str] = set()
    timing: dict[int, frozenset[str]] = {}
    for record in reversed(records):
        inst = record.instruction
        if isinstance(inst, TimeInst) and inst.r1 is not None:
            needed.update(inst.r1.regs())
        elif isinstance(inst, RegWriteInst) and set(inst.reg_write) & needed:
            timing[record.position] = inst.reg_write
            if inst.if_cond is None:
                needed -= set(inst.reg_write)
            needed.update(inst.reg_read)
    return timing


def _check_table_wrap(info: dict[str, Any], seed: Machine) -> None:
    """Require table fields to be independent sweeps that wrap to initial WMEM."""
    for name, definition in info.get("register_tables", {}).items():
        values, offset = definition["values"], definition["offset"]
        if name in info["sweeps"]["registers"]:
            raise ValueError("conflicting table and sweep register")
        if not np.all(seed.value(Register(name)) == values[0]):
            raise ValueError(
                "register table initial word disagrees with compiled register"
            )
        for point in range(info["n"]):
            if not np.all(seed.dmem[offset + point] == values[(point + 1) % info["n"]]):
                raise ValueError("register table values disagree with rotated DMEM")
    for address, fields in info["tables"].items():
        expected = seed.wmem[address].copy()
        for field, offset in fields.items():
            if (
                address in info["sweeps"]["waves"]
                and field in info["sweeps"]["waves"][address]
            ):
                raise ValueError("conflicting table and sweep field")
            if not 0 <= field < 6:
                raise ValueError("invalid table field")
            if (
                int(seed.dmem[offset + info["n"] - 1][0]) % 2**32
                != int(expected[0, field]) % 2**32
            ):
                raise ValueError("table wrap disagrees with initial WMEM")


def _timing_maximum_tests(
    records: Sequence[DecodedInstruction], timing: dict[int, frozenset[str]]
) -> set[int]:
    """Recognize TEST/copy/if(S)-copy maxima consumed by a reference advance.

    Only the emitted register subtraction and its two pure copies qualify.
    The destination must feed TIME; standalone tests and other flag consumers
    remain undeclared shell effects.
    """
    accepted = set()
    for test_record, initial_record, selected_record in zip(
        records[:-2], records[1:-1], records[2:], strict=True
    ):
        test = test_record.instruction
        initial = initial_record.instruction
        selected = selected_record.instruction
        if not (
            isinstance(test, TestInst)
            and test.uf
            and test.op.op == AluOp.SUB
            and isinstance(test.op.rhs, Register)
            and isinstance(initial, RegWriteInst)
            and selected_record.position in timing
        ):
            continue
        if initial == RegWriteInst(
            dst=initial.dst, src=SrcKeyword.OP, op=AluExpr(test.op.lhs, AluOp.NONE)
        ) and selected == RegWriteInst(
            dst=initial.dst,
            src=SrcKeyword.OP,
            op=AluExpr(test.op.rhs, AluOp.NONE),
            if_cond="S",
        ):
            accepted.add(test_record.position)
    return accepted


def _validate_hooks(
    records: tuple[DecodedInstruction, ...],
    loops: dict[str, NativeLoop],
    seed: Machine,
    body_end: int,
    live: set[str],
) -> None:
    axes = [loop for loop in loops.values() if not records[loop.start].body]
    for index, loop in enumerate(axes):
        first = axes[index + 1].end + 1 if index + 1 < len(axes) else body_end + 1
        try:
            if loop.name == "reps" and loop.counter in live:
                raise ValueError(
                    f"reps shell counter {loop.counter} is a cross-shot live-in"
                )
            actual = deepcopy(seed)
            actual.writes.clear()
            tail = records[first : loop.body_end]
            timing = _timing_writes(tail)
            maxima = _timing_maximum_tests(tail, timing)
            for record in tail:
                inst = record.instruction
                if isinstance(inst, (MetaInst, LabelInst, NopInst, TimeInst, WaitInst)):
                    continue
                if isinstance(inst, TestInst) and record.position in maxima:
                    continue
                if isinstance(inst, RegWriteInst) and inst.wp is None:
                    actual.writes.update(inst.reg_write)
                else:
                    raise _failure(
                        record, ValueError("undeclared acquisition shell effect")
                    )
            memory_writes = {
                name for name in actual.writes if name.startswith(("DMEM[", "WMEM["))
            }
            timing_registers = {reg for writes in timing.values() for reg in writes}
            registers = (
                actual.writes - memory_writes - {"s12", "s15"} - timing_registers
            )
            if memory_writes or registers & live or registers:
                raise ValueError(
                    f"{loop.name} shell writes undeclared/cross-shot state: {sorted(actual.writes)}"
                )
            if "s12" in live:
                raise ValueError("cross-shot live-in s12 is updated by the shot shell")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise _failure(
                records[loop.start], ValueError(f"table/shot shell disagrees: {exc}")
            ) from exc


def _apply_tables(
    info: dict[str, Any],
    machine: Machine,
    indexes: IntArray,
) -> None:
    for name, definition in info.get("register_tables", {}).items():
        machine.registers[name] = np.asarray(definition["values"], dtype=np.int64)[
            indexes
        ]
    for address, fields in info["tables"].items():
        for field, offset in fields.items():
            for point in range(1, info["n"]):
                active = indexes == point
                machine.wmem[address][active, field] = (
                    machine.dmem[offset + point - 1][active] % 2**32
                )


def _apply_sweeps(
    records: tuple[DecodedInstruction, ...],
    loops: dict[str, NativeLoop],
    names: tuple[str, ...],
    grid: IntArray,
    machine: Machine,
) -> None:
    for name, loop in loops.items():
        info = _loop_info(records, loop)
        try:
            registers, waves = info["sweeps"]["registers"], info["sweeps"]["waves"]
            if records[loop.start].body:
                continue
            indexes = (
                grid[names.index(name)]
                if name in names
                else np.zeros(machine.size, dtype=np.int64)
            )
            machine.write(Register(loop.counter), indexes, machine.active)
            for register, definition in registers.items():
                machine.registers[register] = signed32(
                    machine.registers[register] + indexes * definition["step"]
                )
            for address, fields in waves.items():
                for field, definition in fields.items():
                    machine.wmem[address][:, field] += indexes * definition["step"]
                    machine.wmem[address][:, field] %= 2 ** (32 if field != 5 else 16)
            _apply_tables(info, machine, indexes)
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            raise _failure(
                records[loop.start],
                ValueError(f"invalid loop sweep/table definition: {exc}"),
            ) from exc


def _run_body(
    records: tuple[DecodedInstruction, ...],
    loops: dict[str, NativeLoop],
    branches: tuple[StructuredBranch, ...],
    machine: Machine,
    outputs: Outputs,
    bounds: tuple[int, int],
) -> None:
    local = {loop.start: loop for loop in loops.values() if records[loop.start].body}
    branch_starts = {b.start: b for b in branches}

    def run(first: int, last: int) -> None:
        index = first
        while index < last:
            if not np.any(machine.active):
                return
            record = records[index]
            if index in local:
                loop = local[index]
                count = _loop_info(records, loop)["n"]
                n = (
                    machine.value(Register(count))
                    if isinstance(count, str)
                    else np.full(machine.size, count, dtype=np.int64)
                )
                if np.any(machine.active & (n < 0)):
                    raise _failure(record, ValueError("negative loop count"))
                outer = machine.active.copy()
                iterations = outputs.iterations
                machine.write(
                    Register(loop.counter),
                    np.zeros(machine.size, dtype=np.int64),
                    outer,
                )
                for iteration in range(int(np.max(n[outer], initial=0))):
                    machine.active = outer & (iteration < n)
                    machine.read_mask = machine.active.copy()
                    machine.write(
                        Register(loop.counter),
                        np.full(machine.size, iteration, dtype=np.int64),
                        machine.active,
                    )
                    outputs.iterations = (*iterations, (loop.name, iteration))
                    run(loop.body_start + 1, loop.body_end)
                outputs.iterations = iterations
                machine.active = outer
                machine.read_mask = outer.copy()
                machine.write(Register(loop.counter), n, outer)
                index = loop.end + 1
                continue
            if index in branch_starts:
                branch = branch_starts[index]
                selector = machine.value(Register(branch.compare))
                outer = machine.active.copy()
                if np.any(outer & ((selector < 0) | (selector >= len(branch.cases)))):
                    raise _failure(
                        record, ValueError("branch selector has no marked case")
                    )
                for case, (a, b) in enumerate(branch.cases):
                    machine.active = outer & (selector == case)
                    machine.read_mask = machine.active.copy()
                    run(a + 1, b)
                machine.active = outer
                machine.read_mask = outer.copy()
                index = branch.end + 1
                continue
            _located_execute(record, machine, outputs)
            index += 1

    run(*bounds)


def _build_lanes(
    prog: MyProgramV2,
    records: tuple[DecodedInstruction, ...],
    loops: dict[str, NativeLoop],
    branches: tuple[StructuredBranch, ...],
    outputs: Outputs,
) -> None:
    boundaries = [
        i
        for i, r in enumerate(records)
        if isinstance(r.instruction, MetaInst)
        and r.instruction.type == "TIMING_BOUNDARY"
    ]
    start, end = boundaries
    for loop in loops.values():
        _validate_loop(records, loop)
    labels = {
        str(r.instruction.name): r.position
        for r in records
        if isinstance(r.instruction, LabelInst)
    }
    machine = Machine(prog, outputs.size, labels)
    outputs.loop_positions = {
        name: records[loop.start].position for name, loop in loops.items()
    }
    for record in records[:start]:
        _located_execute(record, machine, outputs)
    seed = Machine(prog, 1, labels)
    seed_outputs = Outputs(prog, 1, outputs.routes, outputs.pool, outputs.time_base)
    for record in records[:start]:
        _located_execute(record, seed, seed_outputs)
    live = guard_live_in(
        records,
        loops,
        branches,
        (start + 1, end),
        seed,
        reject_carry=outputs.size * prog.loop_dict["reps"] > 1,
    )
    _validate_native_sweeps(prog, records, loops, machine)
    _validate_hooks(records, loops, seed, end, live)
    names = _point_names(records, loops)
    grid = _native_grid(prog, names)
    _apply_sweeps(records, loops, names, grid, machine)
    outputs.begin(machine, np.arange(outputs.size, dtype=np.int64))
    _run_body(records, loops, branches, machine, outputs, (start + 1, end))
    outputs.finish(machine)
    _finish_shell(records, loops, machine, outputs, end)


def _finish_shell(
    records: tuple[DecodedInstruction, ...],
    loops: dict[str, NativeLoop],
    machine: Machine,
    outputs: Outputs,
    end: int,
) -> None:
    # Shot timing and progress counter are outside the user body. Native closing
    # shells are validated, not iterated; reps never enter the execution path.
    tail_end = min(
        loop.body_end for loop in loops.values() if not records[loop.start].body
    )
    table_addresses = {
        address
        for loop in loops.values()
        if loop.native
        for address in _loop_info(records, loop)["tables"]
    }
    timing = _timing_writes(records[end + 1 : tail_end])
    timing_registers = {reg for writes in timing.values() for reg in writes}
    for record in records[end + 1 : tail_end]:
        inst = record.instruction
        if table_addresses and isinstance(
            inst, (RegWriteInst, DmemReadInst, WmemWriteInst)
        ):
            if isinstance(inst, WmemWriteInst):
                continue
            if isinstance(
                inst, DmemReadInst
            ) or inst.dst.canonical_name not in timing_registers | {"s12", "s15"}:
                continue
        _located_execute(record, machine, outputs)
    outputs.validate_fifo(machine)
    terminal = next(r for r in reversed(records) if isinstance(r.instruction, BaseInst))
    for record in records[max(loop.end for loop in loops.values()) + 1 :]:
        inst = record.instruction
        if (
            record is terminal
            and isinstance(inst, JumpInst)
            and not record.body
            and not inst.if_cond
            and not inst.wr
            and not inst.op
        ):
            target = (
                machine.value(inst.addr)
                if inst.label is None
                else np.full(machine.size, machine.labels.get(str(inst.label), -1))
            )
            if np.all(target == record.position):
                continue
        _located_execute(record, machine, outputs)


def _located_execute(
    record: DecodedInstruction, machine: Machine, outputs: Outputs
) -> None:
    outer = machine.active
    try:
        condition = getattr(record.instruction, "if_cond", None)
        if condition is not None:
            machine.active = machine.condition(condition)
            machine.read_mask = machine.active.copy()
        _execute(record, machine, outputs)
    except (ValueError, KeyError, IndexError) as exc:
        raise _failure(record, exc) from exc
    finally:
        machine.active = outer
        machine.read_mask = outer.copy()


def build_compiled_request(
    prog: MyProgramV2,
    mode: AcquisitionMode,
    binding: SimulationBinding,
    sequence: int,
    *,
    routes: Sequence[tuple[int, int]],
    charge_requirements: tuple[ChargeDriveRequirement, ...] = (),
) -> AcquireRequest:
    """Detach compiled instructions into numeric slots without calling a backend.

    prog is a compiled native v2 program. routes is the SimulationSoc's
    explicit (generator, ADC) wiring, not program declarations. mode chooses sums
    or traces; binding and sequence supply sampling identity. charge_requirements
    is forwarded unchanged for environment-owned physical validation; the builder
    does not infer drive roles from names or instructions. Unsupported effects,
    malformed/missing loop definitions, cross-shot state or pending config carry,
    and ambiguous routing raise ValueError
    with instruction type, pre-optimization position and available source/loops.
    SimulationSoc acquisition uses this builder for native programs.
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
    clocks = [float(prog.tproccfg["f_time"])]
    for ch in prog.gen_chs:
        cfg = prog.soccfg["gens"][ch]
        clocks.extend(
            (float(cfg["f_fabric"]), float(cfg["f_fabric"]) * int(cfg["samps_per_clk"]))
        )
    clocks.extend(float(prog.soccfg["readouts"][ch]["f_output"]) for ch in prog.ro_chs)
    time_base = TimeBase.from_clocks(clocks)
    outputs = Outputs(prog, prod(dims), routes, binding.envelope_pool, time_base)
    _build_lanes(prog, records, loops, branches, outputs)
    template = outputs.template()
    windows = tuple(
        ReadoutWindow(
            s.channel,
            int(s.generator_channels[0, -1]),
            s.samples,
            s.sample_period_ticks,
            tuple(int(x) for x in s.trigger_ticks[0]),
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
        int(prog.us2cycles(prog.relax_delay))
        * time_base.period_ticks(prog.tproccfg["f_time"]),
        time_base=time_base,
        charge_requirements=charge_requirements,
    )
