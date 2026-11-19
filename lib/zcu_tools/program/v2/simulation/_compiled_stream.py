"""Predecode detached instructions and validate semantic marker scopes once."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import NoReturn

from ..base import MyProgramV2
from ..ir.compiled_stream import CONTROL_JUMP, CONTROL_LABEL, SOURCE_END, SOURCE_START
from ..ir.hw_semantics import PORT_RESULT_REGS
from ..ir.instruction_fields import CondCode, parse_cond_code
from ..ir.instructions import (
    BaseInst,
    Instruction,
    JumpInst,
    LabelInst,
    MetaInst,
    RegWriteInst,
    TestInst,
)
from ..ir.labels import LabelRef
from ..ir.operands import AluExpr, AluOp, Immediate, Register, SrcKeyword


@dataclass(frozen=True)
class ForwardJump:
    """Transfer at a CONTROL_JUMP marker in the decoded stream.

    end indexes the actual JUMP, after its setup instructions. target indexes
    the destination CONTROL_LABEL. condition is the JUMP's Z/S test or None
    for an unconditional transfer.
    """

    end: int
    target: int
    condition: CondCode | None


@dataclass(frozen=True)
class DecodedInstruction:
    """One retained IR instruction and its pre-optimization diagnostic context.

    instruction is the sole detached snapshot copy. position is its logical word
    offset. source is the innermost source name or empty. body excludes the QICK
    shot shell. loops lists enclosing loop names from outermost inward. cases
    lists enclosing branch names and zero-based case indices, outermost first.
    jump is set only on a validated CONTROL_JUMP marker.
    """

    instruction: Instruction
    position: int
    source: str
    body: bool
    loops: tuple[str, ...]
    cases: tuple[str, ...]
    jump: ForwardJump | None = None

    def failure(self, reason: str) -> ValueError:
        """Report a rejected instruction or marker with its snapshot location."""
        return ValueError(
            f"{type(self.instruction).__name__} at preopt {self.position}, "
            f"loops={self.loops}, cases={self.cases}, source={self.source!r}: {reason}"
        )


@dataclass(frozen=True)
class ForwardSegment:
    """Run records in [first, stop), then transfer or resume, if requested.

    jump describes a transfer after its setup instructions. resume indexes a
    destination marker with waiting paths. At most one is set; neither is set
    on the final segment of the body.
    """

    first: int
    stop: int
    jump: ForwardJump | None = None
    resume: int | None = None


def forward_segments(
    records: tuple[DecodedInstruction, ...],
    bounds: tuple[int, int],
    waiting: Mapping[int, object],
) -> Iterator[ForwardSegment]:
    """Split a validated body at jumps and at labels with waiting paths.

    bounds is the half-open body range. The caller updates waiting by label
    index after each yielded segment. A jump segment includes its marker and
    setup but not the JUMP itself. A resume segment stops before the label marker.
    """
    first, last = bounds
    for index in range(first, last):
        if (jump := records[index].jump) is not None:
            yield ForwardSegment(first, jump.end, jump=jump)
            first = jump.end + 1
        elif index in waiting:
            yield ForwardSegment(first, index, resume=index)
            first = index + 1
    yield ForwardSegment(first, last)


@dataclass(frozen=True)
class NativeLoop:
    """Validated boundaries in the predecoded record sequence, not word addresses.

    name identifies an acquisition axis or body-local loop. start/body_start/
    body_end/end index its four marker records. counter is the canonical hardware
    register. native is True for a QICK-owned counter/update shell, False for
    a framework inner loop with an optional zero-count entry guard.
    """

    name: str
    start: int
    body_start: int
    body_end: int
    end: int
    counter: str
    native: bool


@dataclass(frozen=True)
class StructuredBranch:
    """Marked dispatch and ordered cases, indexed in the decoded record sequence.

    name is diagnostic identity; compare is its canonical selector register.
    start/end bound the branch; cases contains each paired case marker interval.
    """

    name: str
    compare: str
    start: int
    end: int
    cases: tuple[tuple[int, int], ...]


@dataclass
class _BranchScope:
    name: str
    compare: str
    start: int
    loops: tuple[str, ...]
    cases: list[list[int]] = field(default_factory=list)
    active: bool = False


def _loop_definitions(inst: MetaInst) -> None:
    """Require resolved word definitions before consuming any loop metadata."""
    try:
        n = inst.info["n"]
        if not isinstance(n, (int, str)) or isinstance(n, int) and n < 0:
            raise ValueError("invalid loop count")
        sweeps = inst.info["sweeps"]
        if not isinstance(sweeps, dict):
            raise ValueError("missing sweep definitions")
        registers, waves = sweeps["registers"], sweeps["waves"]
        if not isinstance(registers, dict) or not isinstance(waves, dict):
            raise ValueError("invalid sweep definitions")
        definitions = list(registers.values())
        for address, fields in waves.items():
            if (
                not isinstance(address, int)
                or address < 0
                or not isinstance(fields, dict)
            ):
                raise ValueError("invalid waveform sweep address")
            definitions.extend(fields.values())
        for definition in definitions:
            if not isinstance(definition, dict) or not all(
                isinstance(definition[key], int) for key in ("start", "step")
            ):
                raise ValueError("invalid sweep start/step words")
        _table_definitions(inst)
    except KeyError as exc:
        raise ValueError(f"missing loop definition key {exc}") from exc


def _register_table_definitions(inst: MetaInst) -> None:
    """Validate native register addresses, signed words and rotated DMEM offsets."""
    registers = inst.info.get("register_tables", {})
    if not isinstance(registers, dict):
        raise ValueError("invalid register table definitions")
    for name, definition in registers.items():
        if not isinstance(name, str) or not isinstance(definition, dict):
            raise ValueError("invalid register table address/definition")
        values, offset = definition["values"], definition["offset"]
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("invalid register table DMEM offset")
        if (
            not isinstance(values, (list, tuple))
            or len(values) != inst.info["n"]
            or not all(
                isinstance(word, int) and -(2**31) <= word < 2**31 for word in values
            )
        ):
            raise ValueError("invalid register table integer words")


def _table_definitions(inst: MetaInst) -> None:
    if inst.info.get("native") is not True:
        return
    _register_table_definitions(inst)
    tables = inst.info["tables"]
    if not isinstance(tables, dict):
        raise ValueError("invalid table definitions")
    for address, fields in tables.items():
        if not isinstance(address, int) or address < 0 or not isinstance(fields, dict):
            raise ValueError("invalid table WMEM address")
        if not fields or not all(
            isinstance(field, int)
            and 0 <= field < 6
            and isinstance(offset, int)
            and offset >= 0
            for field, offset in fields.items()
        ):
            raise ValueError("invalid table fields/DMEM offsets")


@dataclass
class _Scopes:
    sources: list[str] = field(default_factory=list)
    loops: list[str] = field(default_factory=list)
    parts: dict[str, list[int]] = field(default_factory=dict)
    counters: dict[str, str] = field(default_factory=dict)
    native: dict[str, bool] = field(default_factory=dict)
    local_loops: set[str] = field(default_factory=set)
    body: bool = False
    boundaries: list[str] = field(default_factory=list)
    disabled: int = 0
    branches: list[_BranchScope] = field(default_factory=list)
    complete_branches: list[StructuredBranch] = field(default_factory=list)

    def consume(self, inst: MetaInst, index: int) -> None:
        if inst.type == SOURCE_START:
            self.sources.append(inst.name)
        elif inst.type == SOURCE_END:
            if not self.sources or self.sources.pop() != inst.name:
                raise ValueError("unbalanced source marker")
        elif inst.type == "TIMING_BOUNDARY":
            if inst.name not in ("body_start", "body_end"):
                raise ValueError("unknown timing boundary")
            self.boundaries.append(inst.name)
            self.body = inst.name == "body_start"
        elif inst.type == "DISABLE_OPT_START":
            self.disabled += 1
        elif inst.type == "DISABLE_OPT_END":
            self.disabled -= 1
            if self.disabled < 0:
                raise ValueError("unbalanced disable-opt marker")
        elif inst.type.startswith("BRANCH_"):
            self.branch(inst, index)
        elif inst.type.startswith("LOOP_"):
            self.native_loop(inst, index)
        elif inst.type not in {CONTROL_JUMP, CONTROL_LABEL}:
            raise ValueError(f"unknown or unsupported marker {inst.type}")

    def check_control(self, record: DecodedInstruction, marker: str) -> None:
        if not self.body or self.local_loops.intersection(self.loops) or self.branches:
            raise record.failure(f"{marker} must be at the top level of the body")

    def branch(self, inst: MetaInst, index: int) -> None:
        if inst.type == "BRANCH_START":
            compare = inst.info.get("compare_reg")
            if not isinstance(compare, str):
                raise ValueError("missing branch selector register")
            self.branches.append(
                _BranchScope(inst.name, compare, index, tuple(self.loops))
            )
            return
        if not self.branches:
            raise ValueError("unbalanced branch marker")
        branch = self.branches[-1]
        if tuple(self.loops) != branch.loops:
            raise ValueError("branch case crosses loop boundary")
        if inst.type == "BRANCH_CASE_START":
            if branch.active or inst.name != str(len(branch.cases)):
                raise ValueError("unbalanced or unordered branch case")
            branch.cases.append([index])
            branch.active = True
        elif inst.type == "BRANCH_CASE_END":
            if not branch.active or inst.name != str(len(branch.cases) - 1):
                raise ValueError("unbalanced branch case end")
            branch.cases[-1].append(index)
            branch.active = False
        elif inst.type == "BRANCH_END":
            if branch.active or branch.name != inst.name or not branch.cases:
                raise ValueError("unbalanced or empty branch")
            self.complete_branches.append(
                StructuredBranch(
                    branch.name,
                    branch.compare,
                    branch.start,
                    index,
                    tuple((case[0], case[1]) for case in branch.cases),
                )
            )
            self.branches.pop()
        else:
            raise ValueError("unknown branch marker")

    def native_loop(self, inst: MetaInst, index: int) -> None:
        if inst.type == "LOOP_START":
            if inst.name in self.parts:
                raise ValueError("duplicate loop name")
            counter = inst.info.get("counter_reg")
            if not isinstance(counter, str):
                raise ValueError("missing loop counter register")
            _loop_definitions(inst)
            self.loops.append(inst.name)
            self.parts[inst.name] = [index]
            self.counters[inst.name] = counter
            self.native[inst.name] = inst.info.get("native") is True
            if self.body:
                self.local_loops.add(inst.name)
            return
        expected = {"LOOP_BODY_START": 1, "LOOP_BODY_END": 2, "LOOP_END": 3}
        if (
            not self.loops
            or self.loops[-1] != inst.name
            or len(self.parts[inst.name]) != expected.get(inst.type)
        ):
            raise ValueError("unbalanced or unknown loop marker")
        self.parts[inst.name].append(index)
        if inst.type == "LOOP_END":
            self.loops.pop()

    def finish(self, prog: MyProgramV2) -> dict[str, NativeLoop]:
        if (
            self.sources
            or self.branches
            or self.loops
            or self.disabled
            or self.boundaries != ["body_start", "body_end"]
        ):
            raise ValueError(
                "MetaInst at preopt end: missing/unbalanced source, loop or body boundary markers"
            )
        if tuple(name for name in self.parts if name not in self.local_loops) != tuple(
            name for name in prog.loop_dict if name not in self.local_loops
        ):
            raise ValueError(
                "MetaInst LOOP_START at preopt: native loops do not match compiled acquisition layout"
            )
        return {
            name: NativeLoop(
                name,
                parts[0],
                parts[1],
                parts[2],
                parts[3],
                self.counters[name],
                self.native[name],
            )
            for name, parts in self.parts.items()
        }


def _branch_jump(
    target: LabelRef, *, big: bool, condition: CondCode | None = None
) -> list[BaseInst]:
    address = Register("s15")
    result: list[BaseInst] = []
    if big:
        result.append(RegWriteInst(dst=address, src=SrcKeyword.LABEL, label=target))
    result.append(
        JumpInst(addr=address, if_cond=condition)
        if big
        else JumpInst(label=target, if_cond=condition)
    )
    return result


def _branch_dispatch_choices(
    compare: str, targets: list[LabelRef], header: tuple[DecodedInstruction, ...]
) -> list[list[BaseInst]]:
    address = Register("s15")
    table_labels = [
        r.instruction.name for r in header if isinstance(r.instruction, LabelInst)
    ]
    choices: list[list[BaseInst]] = []
    for big in (False, True):
        chain: list[BaseInst] = []
        for index, target in enumerate(targets):
            transfer = _branch_jump(
                target, big=big, condition="Z" if index < len(targets) - 1 else None
            )
            if index < len(targets) - 1:
                test = TestInst(
                    AluExpr(Register(compare), AluOp.SUB, Immediate(index)),
                    uf=True,
                )
                transfer.insert(1 if big else 0, test)
            chain.extend(transfer)
        choices.append(chain)
        if len(table_labels) == len(targets):
            table: list[BaseInst] = [
                RegWriteInst(
                    dst=address, src=SrcKeyword.LABEL, label=LabelRef(table_labels[0])
                )
            ]
            table.extend(
                RegWriteInst(
                    dst=address,
                    src=SrcKeyword.OP,
                    op=AluExpr(address, AluOp.ADD, Register(compare)),
                )
                for _ in range(2 if big else 1)
            )
            table.append(JumpInst(addr=address))
            for target in targets:
                table.extend(_branch_jump(target, big=big))
            choices.append(table)
    return choices


def _validate_branch(
    records: tuple[DecodedInstruction, ...], branch: StructuredBranch
) -> None:
    """Check skipped dispatch effects against the marker-owned case boundaries.

    Native conditional chains and Modular address tables encode the same case
    selection. Match their complete effects, including case exits, rather than
    executing raw jumps or using them to discover structure.
    """

    def effects(first: int, last: int) -> list[DecodedInstruction]:
        return [r for r in records[first:last] if isinstance(r.instruction, BaseInst)]

    def fail(record: DecodedInstruction) -> NoReturn:
        raise ValueError(
            f"{type(record.instruction).__name__} at preopt {record.position}, loops={record.loops}, cases={record.cases}, source={record.source!r}: branch dispatch disagrees with markers"
        )

    def entry(index: int) -> LabelRef:
        record = records[min(index, len(records) - 1)]
        inst = record.instruction
        if not isinstance(inst, LabelInst):
            fail(record)
        return LabelRef(inst.name)

    targets = [entry(a + 1) for a, _ in branch.cases]
    end = entry(branch.end + 1)

    def matches(
        actual: list[DecodedInstruction], choices: list[list[BaseInst]]
    ) -> None:
        prefix = 0
        for choice in choices:
            count = 0
            for record, expected in zip(actual, choice, strict=False):
                if record.instruction.to_dict() != expected.to_dict():
                    break
                count += 1
            if count == len(actual) == len(choice):
                return
            prefix = max(prefix, count)
        fail(actual[min(prefix, len(actual) - 1)] if actual else records[branch.start])

    choices = _branch_dispatch_choices(
        branch.compare, targets, records[branch.start + 1 : branch.cases[0][0]]
    )
    matches(effects(branch.start + 1, branch.cases[0][0]), choices)
    for index, (_, last) in enumerate(branch.cases):
        stop = (
            branch.cases[index + 1][0] if index + 1 < len(branch.cases) else branch.end
        )
        exits = [_branch_jump(end, big=False), _branch_jump(end, big=True)]
        if index == len(branch.cases) - 1:
            exits.append([])
        matches(effects(last + 1, stop), exits)


def _control_labels(
    records: list[DecodedInstruction],
) -> dict[str, tuple[int, LabelRef]]:
    labels = {}
    for i, record in enumerate(records):
        inst = record.instruction
        if not isinstance(inst, MetaInst) or inst.type != CONTROL_LABEL:
            continue
        label = inst.info.get("label")
        next_inst = records[i + 1].instruction if i + 1 < len(records) else None
        if (
            inst.info.keys() != {"label"}
            or not isinstance(label, str)
            or not label
            or not isinstance(next_inst, LabelInst)
        ):
            raise record.failure("CONTROL_LABEL must identify a machine label")
        if label in labels:
            raise record.failure(f"duplicate CONTROL_LABEL {label!r}")
        labels[label] = (i, LabelRef(next_inst.name))
    return labels


def _jump_condition(record: DecodedInstruction, marker: MetaInst) -> CondCode | None:
    if marker.info.keys() != {"target", "conditional", "test"}:
        raise record.failure(
            "CONTROL_JUMP requires target, conditional and test fields"
        )
    conditional, test = marker.info["conditional"], marker.info["test"]
    if conditional is False and test is None:
        return None
    if conditional is not True or not isinstance(test, str):
        raise record.failure(
            "CONTROL_JUMP condition disagrees with conditional/test fields"
        )
    if test in ("F", "NF"):
        raise record.failure(f"CONTROL_JUMP external flag test {test!r} is unsupported")
    try:
        return parse_cond_code(test)
    except ValueError as exc:
        raise record.failure(f"CONTROL_JUMP condition is invalid: {exc}") from exc


def _require_jump_test(record: DecodedInstruction) -> None:
    inst = record.instruction
    if not isinstance(inst, TestInst) or not inst.uf:
        raise record.failure("CONTROL_JUMP must be followed by a flag-setting TEST")
    measurement = inst.reg_read & PORT_RESULT_REGS
    if measurement:
        raise record.failure(f"undefined register {min(measurement)}")


def _decode_jump(
    records: list[DecodedInstruction],
    start: int,
    marker: MetaInst,
    labels: dict[str, tuple[int, LabelRef]],
) -> ForwardJump:
    record = records[start]
    condition = _jump_condition(record, marker)
    name = marker.info["target"]
    if not isinstance(name, str) or name not in labels:
        raise record.failure("CONTROL_JUMP has no matching CONTROL_LABEL")
    target, label = labels[name]
    setup = start + 1
    first = records[setup].instruction
    indirect = isinstance(first, RegWriteInst)
    end = setup + int(indirect) + int(condition is not None)
    if target <= end:
        raise record.failure("CONTROL_JUMP must be followed by a forward JUMP")
    if condition is not None:
        _require_jump_test(records[end - 1])
    jump = records[end].instruction
    if not isinstance(jump, JumpInst):
        raise record.failure(
            "CONTROL_JUMP must be immediately followed by its jump sequence"
        )
    if jump.if_cond != condition:
        raise record.failure("CONTROL_JUMP condition disagrees with instruction")
    expected = JumpInst(
        label=None if indirect else label,
        addr=Register("s15") if indirect else None,
        if_cond=condition,
    )
    if jump != expected or (
        indirect
        and first
        != RegWriteInst(dst=Register("s15"), src=SrcKeyword.LABEL, label=label)
    ):
        raise record.failure("CONTROL_JUMP sequence disagrees with target label")
    return ForwardJump(end, target, condition)


def _decode_control(records: list[DecodedInstruction]) -> None:
    labels = _control_labels(records)
    position = 0
    for i, record in enumerate(records):
        inst = record.instruction
        if record.position != position:
            raise record.failure(
                "CONTROL_* snapshot record has an inconsistent position"
            )
        if isinstance(inst, BaseInst):
            position += inst.addr_inc
        elif isinstance(inst, MetaInst) and inst.type == CONTROL_JUMP:
            records[i] = replace(record, jump=_decode_jump(records, i, inst, labels))


def _validate_body_jumps(
    records: tuple[DecodedInstruction, ...],
    loops: dict[str, NativeLoop],
    branches: list[StructuredBranch],
) -> None:
    marked = {r.jump.end for r in records if r.jump is not None}
    for loop in loops.values():
        marked.update(range(loop.start + 1, loop.body_start))
        marked.update(range(loop.body_end + 1, loop.end))
    for branch in branches:
        first = branch.start
        for a, b in branch.cases:
            marked.update(range(first + 1, a))
            first = b
        marked.update(range(first + 1, branch.end))
    for i, record in enumerate(records):
        if record.body and isinstance(record.instruction, JumpInst) and i not in marked:
            raise record.failure(
                f"unsupported instruction effect: {record.instruction}"
            )


def predecode(
    prog: MyProgramV2,
) -> tuple[
    tuple[DecodedInstruction, ...], dict[str, NativeLoop], tuple[StructuredBranch, ...]
]:
    """Read each snapshot instruction once and validate all markers before execution.

    prog must have successfully compiled. Returned copies can be reused across
    lanes and loop iterations. Unknown, missing or unbalanced metadata raises
    ValueError with its class, pre-optimization position and active scopes.
    """
    scopes = _Scopes()
    records = []
    has_control = False
    for record in prog.compiled_stream.records:
        inst = record.instruction
        try:
            if isinstance(inst, MetaInst):
                scopes.consume(inst, len(records))
        except ValueError as exc:
            raise ValueError(
                f"MetaInst {inst.type if isinstance(inst, MetaInst) else ''} at preopt {record.p_addr}, loops={scopes.loops}, source={scopes.sources}: {exc}"
            ) from exc
        decoded = DecodedInstruction(
            inst,
            record.p_addr,
            scopes.sources[-1] if scopes.sources else "",
            scopes.body,
            tuple(scopes.loops),
            tuple(
                f"{b.name}:case-{len(b.cases) - 1}" for b in scopes.branches if b.active
            ),
        )
        if isinstance(inst, MetaInst) and inst.type in {
            CONTROL_JUMP,
            CONTROL_LABEL,
        }:
            scopes.check_control(decoded, inst.type)
            has_control = True
        records.append(decoded)
    loops = scopes.finish(prog)
    if has_control:
        _decode_control(records)
    decoded = tuple(records)
    for branch in scopes.complete_branches:
        _validate_branch(decoded, branch)
    _validate_body_jumps(decoded, loops, scopes.complete_branches)
    return decoded, loops, tuple(scopes.complete_branches)
