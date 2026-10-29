"""Detached logical IR and linked positions before optimization."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field

from .instructions import BaseInst, Instruction
from .linker import IRLinker


@dataclass(frozen=True, init=False)
class CompiledRecord:
    """One logical instruction or marker at its pre-optimization word address.

    p_addr is a zero-based program-memory word offset, not an optimized address
    or a stream index. Labels and metadata occupy no words and can share an
    address, including the address immediately after the final instruction.
    Construct with an Instruction and its linked p_addr; the payload is copied.
    """

    p_addr: int
    _instruction: Instruction = field(repr=False)

    def __init__(self, instruction: Instruction, p_addr: int) -> None:
        object.__setattr__(self, "p_addr", p_addr)
        object.__setattr__(self, "_instruction", deepcopy(instruction))

    @property
    def instruction(self) -> Instruction:
        """Return a detached typed IR copy, including operands and metadata.

        Every access deep-copies the payload. Mutating the returned operand or
        metadata cannot change this snapshot, another read, the program or the
        optimizer. Consumers should read each record once per request build and
        retain that copy for pre-decoding/execution, not read it per lane/loop.
        """
        return deepcopy(self._instruction)


@dataclass(frozen=True)
class CompiledStream:
    """Ordered logical IR records from one compile, before any optimizer pass.

    records is an immutable tuple in original instruction/label/marker order.
    Each record preserves its pre-optimization linked position. This artifact
    contains no simulation state and is separate from optimized program tables.
    """

    records: tuple[CompiledRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))

    @classmethod
    def from_instructions(cls, instructions: Sequence[Instruction]) -> CompiledStream:
        """Copy logical instructions and derive positions through IRLinker.

        Accept a flat ordered IR sequence, including labels and metadata.
        Linker errors (such as duplicate labels) propagate. The input and its
        nested operands/metadata are never retained as mutable aliases.
        """
        logical = list(instructions)
        program, _, markers, _ = IRLinker().link(logical)
        instruction_addresses = iter(item["P_ADDR"] for item in program)
        marker_addresses = iter(item["p_addr"] for item in markers)
        records: list[CompiledRecord] = []
        for instruction in logical:
            address = next(
                instruction_addresses
                if isinstance(instruction, BaseInst)
                else marker_addresses
            )
            if not isinstance(address, int):
                raise TypeError("IRLinker returned a noninteger program address")
            records.append(CompiledRecord(instruction, address))
        return cls(tuple(records))
