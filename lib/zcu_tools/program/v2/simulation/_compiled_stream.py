"""Predecode detached instructions and validate semantic marker scopes once."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..base import MyProgramV2
from ..ir.instructions import Instruction, MetaInst


@dataclass(frozen=True)
class DecodedInstruction:
    """One retained IR instruction and its pre-optimization diagnostic context.

    instruction is the sole detached snapshot copy. position is its logical word
    offset. source is the innermost source name or empty. body excludes the QICK
    shot shell. loops lists enclosing native loop names from outermost inward.
    """

    instruction: Instruction
    position: int
    source: str
    body: bool
    loops: tuple[str, ...]


@dataclass(frozen=True)
class NativeLoop:
    """Validated boundaries in the predecoded record sequence, not word addresses.

    name identifies the acquisition axis. start/body_start/body_end/end index its
    four marker records. counter is the resolved canonical hardware register.
    """

    name: str
    start: int
    body_start: int
    body_end: int
    end: int
    counter: str


@dataclass
class _Scopes:
    sources: list[str] = field(default_factory=list)
    loops: list[str] = field(default_factory=list)
    parts: dict[str, list[int]] = field(default_factory=dict)
    counters: dict[str, str] = field(default_factory=dict)
    body: bool = False
    boundaries: list[str] = field(default_factory=list)
    disabled: int = 0

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
        elif inst.type.startswith("LOOP_"):
            self.native_loop(inst, index)
        else:
            raise ValueError(f"unknown or unsupported marker {inst.type}")

    def native_loop(self, inst: MetaInst, index: int) -> None:
        if inst.info.get("native") is not True:
            raise ValueError("non-native structured loop is not yet supported")
        if inst.type == "LOOP_START":
            if inst.name in self.parts:
                raise ValueError("duplicate loop name")
            counter = inst.info.get("counter_reg")
            if not isinstance(counter, str):
                raise ValueError("missing loop counter register")
            self.loops.append(inst.name)
            self.parts[inst.name] = [index]
            self.counters[inst.name] = counter
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
            or self.loops
            or self.disabled
            or self.boundaries != ["body_start", "body_end"]
        ):
            raise ValueError(
                "MetaInst at preopt end: missing/unbalanced source, loop or body boundary markers"
            )
        if tuple(self.parts) != tuple(prog.loop_dict):
            raise ValueError(
                "MetaInst LOOP_START at preopt: native loops do not match compiled acquisition layout"
            )
        return {
            name: NativeLoop(
                name, parts[0], parts[1], parts[2], parts[3], self.counters[name]
            )
            for name, parts in self.parts.items()
        }


def predecode(
    prog: MyProgramV2,
) -> tuple[tuple[DecodedInstruction, ...], dict[str, NativeLoop]]:
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
            )
        )
    return tuple(records), scopes.finish(prog)
