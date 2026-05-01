from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .instructions import Instruction, LabelInst, MetaInst
from .node import BlockNode, IRBranch, IRBranchCase, IRLoop, IRNode, RootNode


@dataclass
class BuildContext:
    root: RootNode
    block_stack: list[BlockNode] = field(default_factory=list)
    struct_stack: list[IRNode] = field(default_factory=list)
    case_stack: list[tuple[IRBranch, str]] = field(default_factory=list)

    # Cache for a label that appears immediately after LOOP_START or BRANCH_START
    pending_label: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.block_stack:
            self.block_stack.append(self.root)

    @property
    def current_block(self) -> BlockNode:
        return self.block_stack[-1]


def _loop_sections(
    loop_node: IRLoop,
) -> tuple[BlockNode, BlockNode, BlockNode, BlockNode, BlockNode]:
    return (
        loop_node.initial,
        loop_node.update,
        loop_node.stop_check,
        loop_node.body,
        loop_node.jump_back,
    )


def _build_loop_start(inst: MetaInst, ctx: BuildContext) -> None:
    loop_node = IRLoop(name=inst.name, trip_count=inst.args["trip_count"])
    ctx.current_block.append(loop_node)
    ctx.struct_stack.append(loop_node)


def _build_loop_section(inst: MetaInst, ctx: BuildContext) -> None:
    if not ctx.struct_stack or not isinstance(ctx.struct_stack[-1], IRLoop):
        raise ValueError("LOOP_SECTION outside of IRLoop")

    loop_node = ctx.struct_stack[-1]
    section_name = inst.name
    if section_name not in {"initial", "update", "stop_check", "body", "jump_back"}:
        raise ValueError(f"Unknown LOOP_SECTION name: {section_name}")

    # Capture start_label if it's the first section and we have a pending label
    if section_name == "initial" and ctx.pending_label:
        loop_node.start_label = ctx.pending_label
        ctx.pending_label = None

    if len(ctx.block_stack) > 1 and ctx.block_stack[-1] in _loop_sections(loop_node):
        ctx.block_stack.pop()

    section_block = getattr(loop_node, section_name)
    ctx.block_stack.append(section_block)


def _build_loop_end(inst: MetaInst, ctx: BuildContext) -> None:
    if (
        not ctx.struct_stack
        or not isinstance(ctx.struct_stack[-1], IRLoop)
        or ctx.struct_stack[-1].name != inst.name
    ):
        raise ValueError(f"Mismatched LOOP_END for {inst.name}")

    loop_node = ctx.struct_stack.pop()
    if not isinstance(loop_node, IRLoop):
        raise ValueError(f"Mismatched LOOP_END for {inst.name}")

    # Capture end_label if the last thing in jump_back was a label
    if loop_node.jump_back.insts:
        last_item = loop_node.jump_back.insts[-1]
        if isinstance(last_item, LabelInst):
            loop_node.end_label = last_item.name
            loop_node.jump_back.insts.pop()

    if len(ctx.block_stack) > 1 and ctx.block_stack[-1] in _loop_sections(loop_node):
        ctx.block_stack.pop()


def _build_branch_start(inst: MetaInst, ctx: BuildContext) -> None:
    branch_node = IRBranch(name=inst.name)
    ctx.current_block.append(branch_node)
    ctx.struct_stack.append(branch_node)
    # BRANCH dispatch is where instructions go until CASE_START
    ctx.block_stack.append(branch_node.dispatch)


def _build_branch_case_start(inst: MetaInst, ctx: BuildContext) -> None:
    if not ctx.struct_stack or not isinstance(ctx.struct_stack[-1], IRBranch):
        raise ValueError("BRANCH_CASE_START outside of IRBranch")

    branch_node = ctx.struct_stack[-1]
    
    # If we were in dispatch, pop it
    if ctx.block_stack[-1] is branch_node.dispatch:
        ctx.block_stack.pop()
    
    case_block = IRBranchCase(name=inst.name)
    branch_node.cases.append(case_block)
    ctx.block_stack.append(case_block)
    ctx.case_stack.append((branch_node, inst.name))


def _build_branch_case_end(inst: MetaInst, ctx: BuildContext) -> None:
    if (
        len(ctx.block_stack) <= 1
        or not ctx.struct_stack
        or not isinstance(ctx.struct_stack[-1], IRBranch)
    ):
        raise ValueError("BRANCH_CASE_END outside of IRBranch")

    branch_node = ctx.struct_stack[-1]
    if not ctx.case_stack or ctx.case_stack[-1][0] is not branch_node:
        raise ValueError("BRANCH_CASE_END without matching case start")

    _case_branch, case_name = ctx.case_stack[-1]
    if case_name != inst.name:
        raise ValueError(
            f"Mismatched BRANCH_CASE_END for {inst.name}, expected {case_name}"
        )

    if ctx.block_stack[-1] not in branch_node.cases:
        raise ValueError("BRANCH_CASE_END without matching case block")

    ctx.case_stack.pop()
    ctx.block_stack.pop()
    
    # After a case ends, we might go back to dispatch or wait for next case
    ctx.block_stack.append(branch_node.dispatch)


def _build_branch_end(inst: MetaInst, ctx: BuildContext) -> None:
    if (
        not ctx.struct_stack
        or not isinstance(ctx.struct_stack[-1], IRBranch)
        or ctx.struct_stack[-1].name != inst.name
    ):
        raise ValueError(f"Mismatched BRANCH_END for {inst.name}")

    if ctx.case_stack and isinstance(ctx.struct_stack[-1], IRBranch) and ctx.case_stack[-1][0] is ctx.struct_stack[-1]:
        raise ValueError(f"BRANCH_END with unclosed case for {inst.name}")

    branch_node = ctx.struct_stack.pop()
    if not isinstance(branch_node, IRBranch):
         raise ValueError(f"Mismatched BRANCH_END for {inst.name}")

    if ctx.block_stack[-1] is branch_node.dispatch:
        ctx.block_stack.pop()
    else:
        raise ValueError("BRANCH_END with unclosed inner blocks")


_META_BUILDERS: dict[str, Callable[[MetaInst, BuildContext], None]] = {
    "LOOP_START": _build_loop_start,
    "LOOP_SECTION": _build_loop_section,
    "LOOP_END": _build_loop_end,
    "BRANCH_START": _build_branch_start,
    "BRANCH_CASE_START": _build_branch_case_start,
    "BRANCH_CASE_END": _build_branch_case_end,
    "BRANCH_END": _build_branch_end,
}


def build_from_instruction(inst: Instruction, ctx: BuildContext) -> None:
    if isinstance(inst, MetaInst):
        handler = _META_BUILDERS.get(inst.type)
        if handler is None:
            raise ValueError(f"Unknown MetaInst type: {inst.type}")
        handler(inst, ctx)
        return

    # Handle Label capture for structural nodes
    if isinstance(inst, LabelInst):
        # If we just saw LOOP_START, this might be start_label
        if ctx.struct_stack and isinstance(ctx.struct_stack[-1], IRLoop) and ctx.block_stack[-1] is ctx.root:
             ctx.pending_label = inst.name
             # Do NOT append it to root block
             return
        if ctx.struct_stack and isinstance(ctx.struct_stack[-1], IRBranch) and ctx.block_stack[-1] is ctx.root:
             # Similar for branch? Usually labels are inside dispatch though.
             pass

    ctx.current_block.append(inst)
