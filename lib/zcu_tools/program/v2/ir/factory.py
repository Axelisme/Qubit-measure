from __future__ import annotations

from typing import Optional

from .instructions import Instruction, LabelInst, MetaInst
from .node import BlockNode, InstNode, IRBranch, IRBranchCase, IRLoop, RootNode


class InstructionStream:
    """A stream of IR instructions for recursive descent parsing."""

    def __init__(self, insts: list[Instruction]):
        self.insts = insts
        self.pos = 0

    def peek(self) -> Optional[Instruction]:
        if self.pos < len(self.insts):
            return self.insts[self.pos]
        return None

    def consume(self) -> Instruction:
        if self.pos >= len(self.insts):
            raise ValueError("Unexpected end of instruction stream")
        inst = self.insts[self.pos]
        self.pos += 1
        return inst

    def consume_meta(self, expected_type: str) -> MetaInst:
        inst = self.consume()
        if not isinstance(inst, MetaInst) or inst.type != expected_type:
            raise ValueError(f"Expected META {expected_type}, got {inst}")
        return inst


def parse_root(stream: InstructionStream) -> RootNode:
    root = RootNode()
    parse_block(stream, block=root)
    return root


def parse_block(
    stream: InstructionStream,
    block: BlockNode,
    end_markers: frozenset[str] | set[str] = frozenset(),
) -> None:
    """Parses instructions into the given block until an end_marker is encountered."""
    while True:
        inst = stream.peek()
        if inst is None:
            break

        if isinstance(inst, MetaInst) and inst.type in end_markers:
            break

        if isinstance(inst, MetaInst):
            if inst.type == "LOOP_START":
                block.append(parse_loop(stream))
            elif inst.type == "BRANCH_START":
                block.append(parse_branch(stream))
            else:
                raise ValueError(f"Unexpected MetaInst encountered: {inst}")
        else:
            block.append(InstNode(stream.consume()))


def parse_loop(stream: InstructionStream) -> IRLoop:
    start_meta = stream.consume_meta("LOOP_START")

    # 1. Skip over physical loop control logic until LOOP_BODY_START, but capture the start label.
    start_label = None
    while True:
        inst = stream.peek()
        if inst is None:
            raise ValueError("Unexpected end of stream while parsing LOOP_START")
        if isinstance(inst, MetaInst) and inst.type == "LOOP_BODY_START":
            break

        inst = stream.consume()
        if isinstance(inst, LabelInst):
            start_label = inst.name

    stream.consume_meta("LOOP_BODY_START")

    loop_node = IRLoop(
        name=start_meta.name,
        counter_reg=start_meta.args["counter_reg"],
        n=start_meta.args["n"],
        range_hint=start_meta.args.get("range_hint"),
        start_label=start_label,
    )

    # 2. Parse the body.
    parse_block(stream, loop_node.body, end_markers={"LOOP_BODY_END"})

    stream.consume_meta("LOOP_BODY_END")

    # 3. Skip over physical jump back logic until LOOP_END, but capture the end label.
    end_label = None
    while True:
        inst = stream.peek()
        if inst is None:
            raise ValueError("Unexpected end of stream while parsing LOOP_BODY_END")
        if isinstance(inst, MetaInst) and inst.type == "LOOP_END":
            break

        inst = stream.consume()
        if isinstance(inst, LabelInst):
            end_label = inst.name

    end_meta = stream.consume_meta("LOOP_END")
    if end_meta.name != loop_node.name:
        raise ValueError(
            f"Mismatched LOOP_END for {loop_node.name}, got {end_meta.name}"
        )

    loop_node.end_label = end_label
    return loop_node


def parse_branch(stream: InstructionStream) -> IRBranch:
    start_meta = stream.consume_meta("BRANCH_START")
    branch_node = IRBranch(name=start_meta.name)

    # 1. Parse dispatch block until we hit cases or the end
    parse_block(
        stream,
        branch_node.dispatch,
        end_markers={"BRANCH_CASE_START", "BRANCH_END"},
    )

    # 2. Parse cases and any intervening dispatch instructions
    while True:
        inst = stream.peek()
        if inst is None:
            raise ValueError("Unexpected end of stream while parsing branch cases")

        if isinstance(inst, MetaInst) and inst.type == "BRANCH_END":
            break

        if isinstance(inst, MetaInst) and inst.type == "BRANCH_CASE_START":
            branch_node.cases.append(parse_branch_case(stream))
        elif isinstance(inst, MetaInst):
            raise ValueError(f"Unexpected MetaInst between branch cases: {inst}")
        else:
            branch_node.dispatch.append(InstNode(stream.consume()))

    end_meta = stream.consume_meta("BRANCH_END")
    if end_meta.name != branch_node.name:
        raise ValueError(
            f"Mismatched BRANCH_END for {branch_node.name}, got {end_meta.name}"
        )

    return branch_node


def parse_branch_case(stream: InstructionStream) -> IRBranchCase:
    start_meta = stream.consume_meta("BRANCH_CASE_START")
    case_node = IRBranchCase(name=start_meta.name)

    parse_block(stream, case_node, end_markers={"BRANCH_CASE_END"})

    end_meta = stream.consume_meta("BRANCH_CASE_END")
    if end_meta.name != case_node.name:
        raise ValueError(
            f"Mismatched BRANCH_CASE_END expected {case_node.name}, got {end_meta.name}"
        )

    return case_node
