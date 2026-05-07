from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from typing_extensions import Iterator

from .instructions import (
    Instruction,
    JumpInst,
    LabelInst,
    MetaInst,
    RegWriteInst,
    TestInst,
)
from .labels import Label


class IRNode:
    """Base class for all IR nodes."""

    fix_addr_size: bool = False

    def children(self) -> Iterator[IRNode]:
        """Yield all immediate child nodes."""
        return iter([])

    def _into_str(self, indent: int = 0) -> str:
        """Helper for __str__ that takes an indent level."""
        return f"{'    ' * indent}{self.__class__.__name__}()"

    def __str__(self) -> str:
        return self._into_str()


@dataclass
class BasicBlockNode(IRNode):
    """A basic block: a straight-line sequence with an optional terminal jump.

    labels: LabelInst(s) that mark the entry point of this block.
    insts:  Linear instructions (no labels, no jumps except TestInst).
    branch: Optional terminal JumpInst that ends this block.
    fix_addr_size: When True, the instruction count is frozen (set by jump-table
                  lowering). Post-LIR passes must NOP-pad instead of removing.
    """

    labels: list[LabelInst] = field(default_factory=list)
    insts: list[Instruction] = field(default_factory=list)
    branch: Optional[JumpInst] = None
    fix_addr_size: bool = False

    def __post_init__(self) -> None:
        for inst in self.insts:
            if isinstance(inst, LabelInst):
                raise ValueError(
                    f"BasicBlockNode.insts must not contain LabelInst; "
                    f"use BasicBlockNode.labels instead. Got: {inst}"
                )
            if isinstance(inst, JumpInst):
                raise ValueError(
                    f"BasicBlockNode.insts must not contain JumpInst; "
                    f"use BasicBlockNode.branch instead. Got: {inst}"
                )

    def children(self) -> Iterator[IRNode]:
        return iter([])

    def _into_str(self, indent: int = 0) -> str:
        prefix = "    " * indent
        lines = []
        for lbl in self.labels:
            lines.append(f"{prefix}{lbl}:")
        for inst in self.insts:
            lines.append(f"{prefix}  {inst}")
        if self.branch is not None:
            lines.append(f"{prefix}  -> {self.branch}")
        return "\n".join(lines) + "\n"


@dataclass
class BlockNode(IRNode):
    """A sequence of IR nodes (structural container).

    Children must be BasicBlockNode | IRLoop | IRBranch | BlockNode.
    """

    insts: list[IRNode] = field(default_factory=list)
    fix_addr_size: bool = False

    def append(self, item: IRNode) -> None:
        self.insts.append(item)

    def children(self) -> Iterator[IRNode]:
        yield from self.insts

    def _into_str(self, indent: int = 0) -> str:
        prefix = "    " * indent
        return f"{prefix}{self.__class__.__name__}()\n" + "\n".join(
            i._into_str(indent + 1) for i in self.insts
        )


@dataclass
class RootNode(BlockNode):
    """The root of the IR tree."""


def _needs_big_jump(pmem_size: Optional[int]) -> bool:
    return pmem_size is not None and pmem_size > 2**11


@dataclass
class IRLoop(IRNode):
    """A loop node."""

    name: str = ""
    counter_reg: str = ""
    n: Union[int, str] = 0
    range_hint: Optional[tuple[int, int]] = None
    body: BlockNode = field(default_factory=BlockNode)
    fix_addr_size: bool = False

    def children(self) -> Iterator[IRNode]:
        yield self.body

    def lower(self, pmem_size: Optional[int] = None) -> list[BasicBlockNode]:
        """Lower this loop into a list of BasicBlockNode (no optimisation).

        Shape:
            meta_start block
            [guard block]   -- only for runtime-driven n
            init block      -- REG_WR counter imm #0
            start_label block
            <body blocks>
            back_edge block -- counter++ then cond JUMP start
            end_label block
        """
        start = Label.make_new(f"{self.name}_start")
        end = Label.make_new(f"{self.name}_end")

        result: list[BasicBlockNode] = []

        result.append(
            BasicBlockNode(
                insts=[
                    MetaInst(
                        type="LOOP_START",
                        name=self.name,
                        info=dict(
                            counter_reg=self.counter_reg,
                            n=self.n,
                            range_hint=self.range_hint,
                        ),
                    )
                ]
            )
        )

        # Guard: skip when runtime n == 0.
        if isinstance(self.n, str):
            if _needs_big_jump(pmem_size):
                guard = BasicBlockNode(
                    insts=[RegWriteInst(dst="s15", src="label", label=end)],
                    branch=JumpInst(addr="s15", if_cond="Z", op=f"{self.n} - #0"),
                )
            else:
                guard = BasicBlockNode(
                    branch=JumpInst(label=end, if_cond="Z", op=f"{self.n} - #0"),
                )
            result.append(guard)

        # Counter init.
        result.append(
            BasicBlockNode(
                insts=[RegWriteInst(dst=self.counter_reg, src="imm", lit="#0")]
            )
        )

        # Start label + LOOP_BODY_START meta.
        result.append(
            BasicBlockNode(
                labels=[LabelInst(name=start, can_remove=True)],
                insts=[MetaInst(type="LOOP_BODY_START", name=self.name)],
            )
        )

        # Body blocks.
        result.extend(_lower_block_node(self.body, pmem_size))

        # Back-edge: counter++ then cond jump back to start.
        op_str = (
            f"{self.counter_reg} - #{self.n}"
            if isinstance(self.n, int)
            else f"{self.counter_reg} - {self.n}"
        )
        back_insts: list[Instruction] = [
            MetaInst(type="LOOP_BODY_END", name=self.name),
            RegWriteInst(
                dst=self.counter_reg,
                src="op",
                op=f"{self.counter_reg} + #1",
            ),
        ]
        if _needs_big_jump(pmem_size):
            back_insts.append(RegWriteInst(dst="s15", src="label", label=start))
            back_edge = BasicBlockNode(
                insts=back_insts,
                branch=JumpInst(addr="s15", if_cond="NS", op=op_str),
            )
        else:
            back_edge = BasicBlockNode(
                insts=back_insts,
                branch=JumpInst(label=start, if_cond="NS", op=op_str),
            )
        result.append(back_edge)

        # End label + LOOP_END meta.
        result.append(
            BasicBlockNode(
                labels=[LabelInst(name=end, can_remove=True)],
                insts=[MetaInst(type="LOOP_END", name=self.name)],
            )
        )

        return result

    def _into_str(self, indent: int = 0) -> str:
        prefix = "    " * indent
        return (
            f"{prefix}IRLoop(name={self.name}, n={self.n}, counter={self.counter_reg}, range_hint={self.range_hint})\n"
            + "\n".join(i._into_str(indent + 1) for i in self.body.insts)
        )


@dataclass
class IRBranch(IRNode):
    """A branch node containing multiple cases (each a BlockNode)."""

    name: str = ""
    compare_reg: str = ""
    cases: list[BlockNode] = field(default_factory=list)
    fix_addr_size: bool = False

    def children(self) -> Iterator[IRNode]:
        yield from self.cases

    def lower(self, pmem_size: Optional[int] = None) -> list[BasicBlockNode]:
        """Lower this branch into BasicBlockNode list using binary dispatch."""
        n = len(self.cases)
        result: list[BasicBlockNode] = []

        result.append(
            BasicBlockNode(
                insts=[
                    MetaInst(
                        type="BRANCH_START",
                        name=self.name,
                        info=dict(compare_reg=self.compare_reg),
                    )
                ]
            )
        )

        def emit_dispatch(lo: int, hi: int) -> None:
            if hi - lo == 1:
                result.extend(_lower_block_node(self.cases[lo], pmem_size))
                return

            mid = (lo + hi) // 2
            left_label = Label.make_new(f"{self.name}_branch_l_{lo}_{mid}")
            end_label = Label.make_new(f"{self.name}_branch_e_{lo}_{hi}")

            result.append(
                BasicBlockNode(
                    insts=[TestInst(op=f"{self.compare_reg} - #{mid}")],
                    branch=JumpInst(label=left_label, if_cond="S"),
                )
            )
            emit_dispatch(mid, hi)
            result.append(BasicBlockNode(branch=JumpInst(label=end_label)))
            result.append(BasicBlockNode(labels=[LabelInst(name=left_label, can_remove=True)]))
            emit_dispatch(lo, mid)
            result.append(BasicBlockNode(labels=[LabelInst(name=end_label, can_remove=True)]))

        emit_dispatch(0, n)

        result.append(
            BasicBlockNode(insts=[MetaInst(type="BRANCH_END", name=self.name)])
        )
        return result

    def _into_str(self, indent: int = 0) -> str:
        prefix = "    " * indent
        return (
            f"{prefix}IRBranch(name={self.name}, compare_reg={self.compare_reg})\n"
            + "\n".join(i._into_str(indent + 1) for i in self.cases)
        )


# ---------------------------------------------------------------------------
# Helper: recursively lower a BlockNode into list[BasicBlockNode].
# Used by IRLoop.lower(), IRBranch.lower(), loop.py, and IRLinker.
# ---------------------------------------------------------------------------


def _lower_block_node(
    block: BlockNode, pmem_size: Optional[int] = None
) -> list[BasicBlockNode]:
    """Recursively flatten a BlockNode into a list of BasicBlockNode.

    - BasicBlockNode  → yield as-is
    - IRLoop          → call .lower()
    - IRBranch        → call .lower()
    - BlockNode       → recurse
    - anything else   → raise TypeError
    """
    result: list[BasicBlockNode] = []
    for child in block.insts:
        if isinstance(child, BasicBlockNode):
            result.append(child)
        elif isinstance(child, IRLoop):
            result.extend(child.lower(pmem_size))
        elif isinstance(child, IRBranch):
            result.extend(child.lower(pmem_size))
        elif isinstance(child, BlockNode):
            result.extend(_lower_block_node(child, pmem_size))
        else:
            raise TypeError(
                f"_lower_block_node: unexpected node type {type(child).__name__}. "
                f"Only BasicBlockNode, IRLoop, IRBranch, and BlockNode are allowed."
            )
    return result
