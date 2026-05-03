from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Iterator, Optional, Union

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

    def children(self) -> Iterator[IRNode]:
        """Yield all immediate child nodes."""
        return iter([])

    def emit(self, inst_list: list[Instruction]) -> None:
        """Flatten this node into a list of Instruction objects."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement emit()"
        )


@dataclass
class InstNode(IRNode):
    """A wrapper for a single linear instruction."""

    inst: Instruction

    def emit(self, inst_list: list[Instruction]) -> None:
        inst_list.append(self.inst)


@dataclass
class BlockNode(IRNode):
    """A sequence of IR nodes."""

    insts: list[IRNode] = field(default_factory=list)

    def append(self, item: IRNode) -> None:
        self.insts.append(item)

    def children(self) -> Iterator[IRNode]:
        yield from self.insts

    def emit(self, inst_list: list[Instruction]) -> None:
        for item in self.insts:
            item.emit(inst_list)


@dataclass
class RootNode(BlockNode):
    """The root of the IR tree."""


@dataclass
class IRLoop(IRNode):
    """A loop node."""

    name: str = ""
    counter_reg: str = ""
    n: Union[int, str] = 0
    range_hint: Optional[tuple[int, int]] = None

    # Structural Labels (Attributes)
    start_label: Optional["Label"] = None
    end_label: Optional["Label"] = None

    body: BlockNode = field(default_factory=BlockNode)

    def children(self) -> Iterator[IRNode]:
        yield self.body

    def emit(self, inst_list: list[Instruction]) -> None:

        start = self.start_label or Label.make_new(f"{self.name}_start")
        end = self.end_label or Label.make_new(f"{self.name}_end")

        # META: LOOP_START
        inst_list.append(
            MetaInst(
                type="LOOP_START",
                name=self.name,
                info=dict(
                    counter_reg=self.counter_reg,
                    n=self.n,
                    range_hint=self.range_hint,
                ),
            )
        )

        # Initialize counter
        inst_list.append(RegWriteInst(dst=self.counter_reg, src="imm", lit="#0"))

        # Loop start label
        inst_list.append(LabelInst(name=start))

        # Stop check
        op_str = (
            f"{self.counter_reg} - #{self.n}"
            if isinstance(self.n, int)
            else f"{self.counter_reg} - {self.n}"
        )
        # TODO: need to consider case of pmem_size > 2**11, it will need s15 instead of immediate
        inst_list.append(TestInst(op=op_str, uf="0"))
        inst_list.append(JumpInst(label=end, if_cond="NS"))

        # META: LOOP_BODY_START
        inst_list.append(MetaInst(type="LOOP_BODY_START", name=self.name))

        self.body.emit(inst_list)

        # Increment counter
        inst_list.append(
            RegWriteInst(dst=self.counter_reg, src="op", op=f"{self.counter_reg} + #1")
        )

        # META: LOOP_BODY_END
        inst_list.append(MetaInst(type="LOOP_BODY_END", name=self.name))

        # Jump back
        inst_list.append(JumpInst(label=start))

        # Loop end label
        inst_list.append(LabelInst(name=end))

        # META: LOOP_END
        inst_list.append(MetaInst(type="LOOP_END", name=self.name))


@dataclass
class IRBranchCase(BlockNode):
    """A branch case with a stable logical identity."""

    name: str = ""

    def emit(self, inst_list: list[Instruction]) -> None:
        inst_list.append(MetaInst(type="BRANCH_CASE_START", name=self.name))
        super().emit(inst_list)
        inst_list.append(MetaInst(type="BRANCH_CASE_END", name=self.name))


@dataclass
class IRBranch(IRNode):
    """A branch node containing multiple cases."""

    name: str = ""
    compare_reg: str = ""
    cases: list[IRBranchCase] = field(default_factory=list)

    def children(self) -> Iterator[IRNode]:
        yield from self.cases

    def emit(self, inst_list: list[Instruction]) -> None:
        n = len(self.cases)

        def emit_dispatch(lo: int, hi: int) -> None:
            if hi - lo == 1:
                self.cases[lo].emit(inst_list)
                return

            mid = (lo + hi) // 2
            left_label = Label.make_new(f"{self.name}_branch_l_{lo}_{mid}")
            end_label = Label.make_new(f"{self.name}_branch_e_{lo}_{hi}")

            # compare_reg - mid < 0  (i.e. compare_reg < mid) → jump to left half
            inst_list.append(TestInst(op=f"{self.compare_reg} - #{mid}", uf="0"))
            inst_list.append(JumpInst(label=left_label, if_cond="S"))
            emit_dispatch(mid, hi)
            inst_list.append(JumpInst(label=end_label))
            inst_list.append(LabelInst(name=left_label))
            emit_dispatch(lo, mid)
            inst_list.append(LabelInst(name=end_label))

        inst_list.append(MetaInst(type="BRANCH_START", name=self.name))
        emit_dispatch(0, n)
        inst_list.append(MetaInst(type="BRANCH_END", name=self.name))
