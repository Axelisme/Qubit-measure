"""Predecode detached instructions and validate semantic marker scopes once."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..base import MyProgramV2
from ..ir.instructions import (
    BaseInst,
    Instruction,
    JumpInst,
    LabelInst,
    MetaInst,
    NopInst,
    RegWriteInst,
    TestInst,
)
from ..ir.labels import LabelRef
from ..ir.operands import AluExpr, AluOp, Immediate, Register, SrcKeyword


@dataclass(frozen=True)
class DecodedInstruction:
    """One retained IR instruction and its pre-optimization diagnostic context.

    instruction is the sole detached snapshot copy. position is its logical word
    offset. source is the innermost source name or empty. body excludes the QICK
    shot shell. loops lists enclosing loop names from outermost inward. cases
    lists enclosing branch names and zero-based case indices, outermost first.
    """

    instruction: Instruction
    position: int
    source: str
    body: bool
    loops: tuple[str, ...]
    cases: tuple[str, ...]


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


def _table_definitions(inst: MetaInst) -> None:
    if inst.info.get("native") is not True:
        return
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
        if inst.type == "SOURCE_START":
            self.sources.append(inst.name)
        elif inst.type == "SOURCE_END":
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
        else:
            raise ValueError(f"unknown or unsupported marker {inst.type}")

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


def _validate_branch(
    records: tuple[DecodedInstruction, ...], branch: StructuredBranch
) -> None:
    """Check skipped dispatch effects against the marker-owned case boundaries.

    Native conditional chains and Modular address tables encode the same case
    selection. Match their complete effects, including case exits, rather than
    executing raw jumps or using them to discover structure.
    """

    def effects(first: int, last: int) -> list[DecodedInstruction]:
        return [
            r
            for r in records[first:last]
            if isinstance(r.instruction, BaseInst)
            and not isinstance(r.instruction, NopInst)
        ]

    def fail(record: DecodedInstruction) -> None:
        raise ValueError(
            f"{type(record.instruction).__name__} at preopt {record.position}, loops={record.loops}, cases={record.cases}, source={record.source!r}: branch dispatch disagrees with markers"
        )

    def entry(index: int) -> LabelRef:
        if index >= len(records) or not isinstance(
            records[index].instruction, LabelInst
        ):
            fail(records[min(index, len(records) - 1)])
        inst = records[index].instruction
        if not isinstance(inst, LabelInst):
            raise ValueError("missing branch entry label")
        return LabelRef(inst.name)

    targets = [entry(a + 1) for a, _ in branch.cases]
    end = entry(branch.end + 1)
    address = Register("s15")

    def jump(
        target: LabelRef, big: bool, condition: str | None = None
    ) -> list[BaseInst]:
        if big:
            return [
                RegWriteInst(dst=address, src=SrcKeyword.LABEL, label=target),
                JumpInst(addr=address, if_cond=condition),
            ]
        return [JumpInst(label=target, if_cond=condition)]

    def matches(
        actual: list[DecodedInstruction], choices: list[list[BaseInst]]
    ) -> None:
        prefix = 0
        for choice in choices:
            count = 0
            for record, expected in zip(actual, choice):
                if record.instruction.to_dict() != expected.to_dict():
                    break
                count += 1
            if count == len(actual) == len(choice):
                return
            prefix = max(prefix, count)
        fail(actual[min(prefix, len(actual) - 1)] if actual else records[branch.start])

    header = records[branch.start + 1 : branch.cases[0][0]]
    table_labels = [
        r.instruction.name for r in header if isinstance(r.instruction, LabelInst)
    ]
    choices: list[list[BaseInst]] = []
    for big in (False, True):
        chain: list[BaseInst] = []
        for index, target in enumerate(targets):
            transfer = jump(target, big, "Z" if index < len(targets) - 1 else None)
            if index < len(targets) - 1:
                test = TestInst(
                    AluExpr(Register(branch.compare), AluOp.SUB, Immediate(index)),
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
                    op=AluExpr(address, AluOp.ADD, Register(branch.compare)),
                )
                for _ in range(2 if big else 1)
            )
            table.append(JumpInst(addr=address))
            for target in targets:
                table.extend(jump(target, big))
            choices.append(table)
    matches(effects(branch.start + 1, branch.cases[0][0]), choices)
    for index, (_, last) in enumerate(branch.cases):
        stop = (
            branch.cases[index + 1][0] if index + 1 < len(branch.cases) else branch.end
        )
        exits = [jump(end, False), jump(end, True)]
        if index == len(branch.cases) - 1:
            exits.append([])
        matches(effects(last + 1, stop), exits)


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
    for record in prog.compiled_stream.records:
        inst = record.instruction
        try:
            if isinstance(inst, MetaInst):
                scopes.consume(inst, len(records))
        except ValueError as exc:
            raise ValueError(
                f"MetaInst {inst.type if isinstance(inst, MetaInst) else ''} at preopt {record.p_addr}, loops={scopes.loops}, source={scopes.sources}: {exc}"
            ) from exc
        records.append(
            DecodedInstruction(
                inst,
                record.p_addr,
                scopes.sources[-1] if scopes.sources else "",
                scopes.body,
                tuple(scopes.loops),
                tuple(
                    f"{b.name}:case-{len(b.cases) - 1}"
                    for b in scopes.branches
                    if b.active
                ),
            )
        )
    loops = scopes.finish(prog)
    decoded = tuple(records)
    for branch in scopes.complete_branches:
        _validate_branch(decoded, branch)
    return decoded, loops, tuple(scopes.complete_branches)
