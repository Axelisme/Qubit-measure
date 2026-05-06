from __future__ import annotations

from typing import Optional

from .instructions import Instruction, JumpInst, LabelInst, MetaInst
from .node import BasicBlockNode, BlockNode, IRBranch, IRLoop, RootNode


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


def _flush_basic_block(
    pending_labels: list[LabelInst],
    pending_insts: list[Instruction],
    pending_branch: Optional[JumpInst],
    block: BlockNode,
) -> tuple[list[LabelInst], list[Instruction], Optional[JumpInst]]:
    """Flush accumulated instructions into a BasicBlockNode if non-empty."""
    if pending_labels or pending_insts or pending_branch is not None:
        block.append(
            BasicBlockNode(
                labels=pending_labels,
                insts=pending_insts,
                branch=pending_branch,
            )
        )
    return [], [], None


def parse_block(
    stream: InstructionStream,
    block: BlockNode,
    end_markers: frozenset[str] | set[str] = frozenset(),
) -> None:
    """Parse instructions into the given block until an end_marker is hit.

    Linear instructions are accumulated into BasicBlockNode(s). A new
    BasicBlockNode starts whenever a LabelInst is encountered (the label
    becomes the entry of the new block). A BasicBlockNode is terminated
    by a JumpInst (stored as `branch`).

    When a structural MetaInst (LOOP_START / BRANCH_START) is encountered,
    any pending accumulation is flushed first, then the structural node is
    appended directly to `block`.
    """
    pending_labels: list[LabelInst] = []
    pending_insts: list[Instruction] = []
    pending_branch: Optional[JumpInst] = None

    while True:
        inst = stream.peek()
        if inst is None:
            break

        if isinstance(inst, MetaInst) and inst.type in end_markers:
            break

        if isinstance(inst, MetaInst):
            if inst.type in ("LOOP_START", "BRANCH_START"):
                # Flush current basic block before the structural node.
                pending_labels, pending_insts, pending_branch = _flush_basic_block(
                    pending_labels, pending_insts, pending_branch, block
                )
                if inst.type == "LOOP_START":
                    block.append(parse_loop(stream))
                else:
                    block.append(parse_branch(stream))
            else:
                raise ValueError(f"Unexpected MetaInst encountered: {inst}")

        elif isinstance(inst, LabelInst):
            # Flush pending accumulation (no branch), then start new block
            # with this label as the entry.
            pending_labels, pending_insts, pending_branch = _flush_basic_block(
                pending_labels, pending_insts, pending_branch, block
            )
            stream.consume()
            pending_labels = [inst]

        elif isinstance(inst, JumpInst):
            # Terminal instruction: ends the current basic block.
            stream.consume()
            pending_branch = inst
            pending_labels, pending_insts, pending_branch = _flush_basic_block(
                pending_labels, pending_insts, pending_branch, block
            )

        else:
            # Regular linear instruction.
            stream.consume()
            pending_insts.append(inst)

    # Flush any remaining accumulation.
    _flush_basic_block(pending_labels, pending_insts, pending_branch, block)


def parse_loop(stream: InstructionStream) -> IRLoop:
    start_meta = stream.consume_meta("LOOP_START")

    # Skip over physical loop control logic until LOOP_BODY_START.
    # We no longer capture the start_label from the instruction stream;
    # it will be regenerated from `name` at lower() time.
    # TODO: start_label / end_label fields on IRLoop will be removed later.
    while True:
        inst = stream.peek()
        if inst is None:
            raise ValueError("Unexpected end of stream while parsing LOOP_START")
        if isinstance(inst, MetaInst) and inst.type == "LOOP_BODY_START":
            break
        stream.consume()

    stream.consume_meta("LOOP_BODY_START")

    loop_node = IRLoop(
        name=start_meta.name,
        counter_reg=start_meta.info["counter_reg"],
        n=start_meta.info["n"],
        range_hint=start_meta.info.get("range_hint"),
    )

    # Parse the body into a BlockNode of BasicBlockNode / structural nodes.
    parse_block(stream, loop_node.body, end_markers={"LOOP_BODY_END"})

    stream.consume_meta("LOOP_BODY_END")

    # Skip over physical back-edge / end-label logic until LOOP_END.
    while True:
        inst = stream.peek()
        if inst is None:
            raise ValueError("Unexpected end of stream while parsing LOOP_BODY_END")
        if isinstance(inst, MetaInst) and inst.type == "LOOP_END":
            break
        stream.consume()

    end_meta = stream.consume_meta("LOOP_END")
    if end_meta.name != loop_node.name:
        raise ValueError(
            f"Mismatched LOOP_END for {loop_node.name}, got {end_meta.name}"
        )

    return loop_node


def parse_branch(stream: InstructionStream) -> IRBranch:
    start_meta = stream.consume_meta("BRANCH_START")
    compare_reg = start_meta.info["compare_reg"]

    branch_node = IRBranch(name=start_meta.name, compare_reg=compare_reg)

    # Non-META instructions between BRANCH_START and BRANCH_END are dispatch
    # control flow (TEST/JUMP/LABEL) — discard them.
    # BRANCH_CASE_START marks the beginning of each case body.
    while True:
        inst = stream.peek()
        if inst is None:
            raise ValueError("Unexpected end of stream while parsing BRANCH")
        if isinstance(inst, MetaInst) and inst.type == "BRANCH_END":
            break
        if isinstance(inst, MetaInst) and inst.type == "BRANCH_CASE_START":
            branch_node.cases.append(parse_branch_case(stream))
        else:
            stream.consume()  # dispatch control instruction

    end_meta = stream.consume_meta("BRANCH_END")
    if end_meta.name != branch_node.name:
        raise ValueError(
            f"Mismatched BRANCH_END for {branch_node.name}, got {end_meta.name}"
        )

    return branch_node


def parse_branch_case(stream: InstructionStream) -> BlockNode:
    """Parse a branch case into a plain BlockNode."""
    start_meta = stream.consume_meta("BRANCH_CASE_START")
    case_node = BlockNode()

    parse_block(stream, case_node, end_markers={"BRANCH_CASE_END"})

    end_meta = stream.consume_meta("BRANCH_CASE_END")
    if end_meta.name != start_meta.name:
        raise ValueError(
            f"Mismatched BRANCH_CASE_END expected {start_meta.name}, got {end_meta.name}"
        )

    return case_node
