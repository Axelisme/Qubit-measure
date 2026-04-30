from __future__ import annotations

from typing import Union

from .instructions import Instruction, MetaInst
from .node import BlockNode, IRBranch, IRBranchCase, IRLoop, IRNode, RootNode


class IRBuilder:
    def build(self, prog_list: list[dict], labels: dict[str, str]) -> RootNode:
        root = RootNode(labels=labels)
        # block_stack tracks the active BlockNode we append instructions to
        block_stack: list[BlockNode] = [root]
        # struct_stack tracks the active IRLoop or IRBranch to manage sections/cases
        struct_stack: list[Union[IRLoop, IRBranch]] = []
        case_stack: list[tuple[IRBranch, str]] = []
        loop_section_names = frozenset(
            {"initial", "update", "stop_check", "body", "jump_back"}
        )

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

        for d in prog_list:
            inst = Instruction.from_dict(d)
            current_block = block_stack[-1]

            if isinstance(inst, MetaInst):
                if inst.type == "LOOP_START":
                    loop_node = IRLoop(
                        name=inst.name, trip_count=inst.args["trip_count"]
                    )
                    current_block.append(loop_node)
                    struct_stack.append(loop_node)

                elif inst.type == "LOOP_SECTION":
                    if not struct_stack or not isinstance(struct_stack[-1], IRLoop):
                        raise ValueError("LOOP_SECTION outside of IRLoop")
                    if inst.name not in loop_section_names:
                        raise ValueError(f"Unknown LOOP_SECTION name: {inst.name}")
                    loop_node = struct_stack[-1]
                    # If there's an active section on block_stack, pop it
                    if len(block_stack) > 1 and block_stack[-1] in _loop_sections(
                        loop_node
                    ):
                        block_stack.pop()

                    section_block = getattr(loop_node, inst.name)
                    block_stack.append(section_block)

                elif inst.type == "LOOP_END":
                    if (
                        not struct_stack
                        or not isinstance(struct_stack[-1], IRLoop)
                        or struct_stack[-1].name != inst.name
                    ):
                        raise ValueError(f"Mismatched LOOP_END for {inst.name}")
                    loop_node = struct_stack.pop()
                    if not isinstance(loop_node, IRLoop):
                        raise ValueError(f"Mismatched LOOP_END for {inst.name}")
                    if len(block_stack) > 1 and block_stack[-1] in _loop_sections(
                        loop_node
                    ):
                        block_stack.pop()

                elif inst.type == "BRANCH_START":
                    branch_node = IRBranch(name=inst.name)
                    current_block.append(branch_node)
                    struct_stack.append(branch_node)
                    block_stack.append(branch_node)

                elif inst.type == "BRANCH_CASE_START":
                    if not struct_stack or not isinstance(struct_stack[-1], IRBranch):
                        raise ValueError("BRANCH_CASE_START outside of IRBranch")
                    branch_node = struct_stack[-1]
                    case_block = IRBranchCase(name=inst.name)
                    branch_node.cases.append(case_block)
                    branch_node.append(case_block)
                    block_stack.append(case_block)
                    case_stack.append((branch_node, inst.name))

                elif inst.type == "BRANCH_CASE_END":
                    if (
                        len(block_stack) > 1
                        and struct_stack
                        and isinstance(struct_stack[-1], IRBranch)
                    ):
                        branch_node = struct_stack[-1]
                        if not case_stack or case_stack[-1][0] is not branch_node:
                            raise ValueError(
                                "BRANCH_CASE_END without matching case start"
                            )
                        _case_branch, case_name = case_stack[-1]
                        if case_name != inst.name:
                            raise ValueError(
                                f"Mismatched BRANCH_CASE_END for {inst.name}, expected {case_name}"
                            )
                        if block_stack[-1] in branch_node.cases:
                            case_stack.pop()
                            block_stack.pop()
                        else:
                            raise ValueError(
                                "BRANCH_CASE_END without matching case block"
                            )
                    else:
                        raise ValueError("BRANCH_CASE_END outside of IRBranch")

                elif inst.type == "BRANCH_END":
                    if (
                        not struct_stack
                        or not isinstance(struct_stack[-1], IRBranch)
                        or struct_stack[-1].name != inst.name
                    ):
                        raise ValueError(f"Mismatched BRANCH_END for {inst.name}")
                    if case_stack and case_stack[-1][0] is struct_stack[-1]:
                        raise ValueError(
                            f"BRANCH_END with unclosed case for {inst.name}"
                        )
                    branch_node = struct_stack.pop()
                    if block_stack[-1] is branch_node:
                        block_stack.pop()
                    else:
                        raise ValueError("BRANCH_END with unclosed inner blocks")
                else:
                    raise ValueError(f"Unknown MetaInst type: {inst.type}")
            else:
                current_block.append(inst)

        if len(block_stack) != 1:
            raise ValueError("Unclosed blocks in IRBuilder")
        if len(struct_stack) != 0:
            raise ValueError("Unclosed structures in IRBuilder")
        if len(case_stack) != 0:
            raise ValueError("Unclosed branch cases in IRBuilder")

        return root

    def unbuild(self, ir: IRNode) -> tuple[list[dict], dict[str, str]]:
        if not isinstance(ir, RootNode):
            raise ValueError("IR node passed to unbuild must be a RootNode")

        prog_list: list[dict] = []

        def _flatten(node: IRNode) -> None:
            if isinstance(node, IRLoop):
                _flatten(node.initial)
                _flatten(node.stop_check)
                _flatten(node.body)
                _flatten(node.update)
            elif isinstance(node, BlockNode):
                for item in node.insts:
                    if isinstance(item, Instruction):
                        # Skip MetaInst on flattening, though they shouldn't be in the tree
                        if not isinstance(item, MetaInst):
                            prog_list.append(item.to_dict())
                    elif isinstance(item, IRNode):
                        _flatten(item)
            else:
                raise ValueError(f"Unknown IRNode type: {type(node)}")

        _flatten(ir)
        return prog_list, ir.labels
