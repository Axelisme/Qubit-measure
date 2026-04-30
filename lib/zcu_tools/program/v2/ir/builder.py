from __future__ import annotations

from typing import Union

from .instructions import Instruction, MetaInst
from .node import BlockNode, IRNode, LoopNode, RootNode


class IRBuilder:
    def build(self, prog_list: list[dict], labels: dict[str, str]) -> RootNode:
        root = RootNode(labels=labels)
        stack: list[BlockNode] = [root]

        for d in prog_list:
            inst = Instruction.from_dict(d)
            current_block = stack[-1]

            if isinstance(inst, MetaInst):
                if inst.type == "LOOP_START":
                    loop_node = LoopNode(name=inst.name)
                    current_block.append(loop_node)
                    stack.append(loop_node)
                elif inst.type == "LOOP_END":
                    if not isinstance(current_block, LoopNode) or current_block.name != inst.name:
                        raise ValueError(f"Mismatched LOOP_END for {inst.name}")
                    stack.pop()
                else:
                    raise ValueError(f"Unknown MetaInst type: {inst.type}")
            else:
                current_block.append(inst)

        if len(stack) != 1:
            raise ValueError("Unclosed blocks in IRBuilder")

        return root

    def unbuild(self, ir: IRNode) -> tuple[list[dict], dict[str, str]]:
        if not isinstance(ir, RootNode):
            raise ValueError("IR node passed to unbuild must be a RootNode")

        prog_list: list[dict] = []
        
        def _flatten(node: IRNode) -> None:
            if isinstance(node, BlockNode):
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
