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

    def emit(
        self, inst_list: list[Instruction], *, pmem_size: Optional[int] = None
    ) -> None:
        """Flatten this node into a list of Instruction objects."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement emit()"
        )


@dataclass
class InstNode(IRNode):
    """A wrapper for a single linear instruction."""

    inst: Instruction

    def emit(
        self, inst_list: list[Instruction], *, pmem_size: Optional[int] = None
    ) -> None:
        inst_list.append(self.inst)


@dataclass
class BlockNode(IRNode):
    """A sequence of IR nodes."""

    insts: list[IRNode] = field(default_factory=list)

    def append(self, item: IRNode) -> None:
        self.insts.append(item)

    def children(self) -> Iterator[IRNode]:
        yield from self.insts

    def emit(
        self, inst_list: list[Instruction], *, pmem_size: Optional[int] = None
    ) -> None:
        for item in self.insts:
            item.emit(inst_list, pmem_size=pmem_size)


@dataclass
class RootNode(BlockNode):
    """The root of the IR tree."""


def _needs_big_jump(pmem_size: Optional[int]) -> bool:
    return pmem_size is not None and pmem_size > 2**11


def _emit_label_jump(
    inst_list: list[Instruction],
    *,
    target: "Label",
    pmem_size: Optional[int],
    if_cond: Optional[str] = None,
    op: Optional[str] = None,
) -> None:
    if _needs_big_jump(pmem_size):
        inst_list.append(RegWriteInst(dst="s15", src="label", label=target))
        inst_list.append(JumpInst(addr="s15", if_cond=if_cond, op=op))
        return
    inst_list.append(JumpInst(label=target, if_cond=if_cond, op=op))


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

    def emit(
        self, inst_list: list[Instruction], *, pmem_size: Optional[int] = None
    ) -> None:

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

        # Guard: skip the loop when runtime n == 0. Constant n is assumed
        # positive by upstream passes.
        if isinstance(self.n, str):
            _emit_label_jump(
                inst_list,
                target=end,
                pmem_size=pmem_size,
                if_cond="Z",
                op=f"{self.n} - #0",
            )

        # Initialize counter
        inst_list.append(RegWriteInst(dst=self.counter_reg, src="imm", lit="#0"))

        # Loop start label sits at the top of the body so the back-edge jumps
        # straight into the next body iteration.
        inst_list.append(LabelInst(name=start))

        # META: LOOP_BODY_START
        inst_list.append(MetaInst(type="LOOP_BODY_START", name=self.name))

        self.body.emit(inst_list, pmem_size=pmem_size)

        # META: LOOP_BODY_END
        inst_list.append(MetaInst(type="LOOP_BODY_END", name=self.name))

        # counter += 1
        inst_list.append(
            RegWriteInst(
                dst=self.counter_reg,
                src="op",
                op=f"{self.counter_reg} + #1",
            )
        )

        # Back-edge: continue while counter < n (signed). Combined cond-jump
        # replaces the prior stop-check + unconditional jump-back pair.
        op_str = (
            f"{self.counter_reg} - #{self.n}"
            if isinstance(self.n, int)
            else f"{self.counter_reg} - {self.n}"
        )
        _emit_label_jump(
            inst_list,
            target=start,
            pmem_size=pmem_size,
            if_cond="NS",
            op=op_str,
        )

        # Loop end label
        inst_list.append(LabelInst(name=end))

        # META: LOOP_END
        inst_list.append(MetaInst(type="LOOP_END", name=self.name))


@dataclass
class IRJumpTableLoop(IRNode):
    """Register-driven loop unrolled to k body copies with last-round dispatch.

    See `passes.loop_dispatch` for the emit() implementation and the full
    asm shape. This node lives in `node.py` so analysis helpers can
    recognize it without importing from the passes package.
    """

    n_reg: str = ""
    counter_reg: str = ""
    k: int = 0
    body_words: int = 0
    entry_labels: list["Label"] = field(default_factory=list)
    exit_label: Optional["Label"] = None
    bodies: list[BlockNode] = field(default_factory=list)
    name: str = ""

    def children(self) -> Iterator[IRNode]:
        yield from self.bodies

    def emit(
        self, inst_list: list[Instruction], *, pmem_size: Optional[int] = None
    ) -> None:
        # Implementation lives in passes.loop_dispatch to avoid pulling
        # codegen helpers into the core node module. Imported lazily.
        from .passes.loop_dispatch import emit_jump_table_loop

        emit_jump_table_loop(self, inst_list, pmem_size=pmem_size)


@dataclass
class IRBranchCase(BlockNode):
    """A branch case with a stable logical identity."""

    name: str = ""

    def emit(
        self, inst_list: list[Instruction], *, pmem_size: Optional[int] = None
    ) -> None:
        inst_list.append(MetaInst(type="BRANCH_CASE_START", name=self.name))
        super().emit(inst_list, pmem_size=pmem_size)
        inst_list.append(MetaInst(type="BRANCH_CASE_END", name=self.name))


@dataclass
class IRBranch(IRNode):
    """A branch node containing multiple cases."""

    name: str = ""
    compare_reg: str = ""
    cases: list[IRBranchCase] = field(default_factory=list)

    def children(self) -> Iterator[IRNode]:
        yield from self.cases

    def emit(
        self, inst_list: list[Instruction], *, pmem_size: Optional[int] = None
    ) -> None:
        n = len(self.cases)

        def emit_dispatch(lo: int, hi: int) -> None:
            if hi - lo == 1:
                self.cases[lo].emit(inst_list, pmem_size=pmem_size)
                return

            mid = (lo + hi) // 2
            left_label = Label.make_new(f"{self.name}_branch_l_{lo}_{mid}")
            end_label = Label.make_new(f"{self.name}_branch_e_{lo}_{hi}")

            # compare_reg - mid < 0  (i.e. compare_reg < mid) → jump to left half
            inst_list.append(TestInst(op=f"{self.compare_reg} - #{mid}"))
            inst_list.append(JumpInst(label=left_label, if_cond="S"))
            emit_dispatch(mid, hi)
            inst_list.append(JumpInst(label=end_label))
            inst_list.append(LabelInst(name=left_label))
            emit_dispatch(lo, mid)
            inst_list.append(LabelInst(name=end_label))

        inst_list.append(
            MetaInst(
                type="BRANCH_START",
                name=self.name,
                info=dict(compare_reg=self.compare_reg),
            )
        )
        emit_dispatch(0, n)
        inst_list.append(MetaInst(type="BRANCH_END", name=self.name))
