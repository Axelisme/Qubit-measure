from __future__ import annotations

from typing import Optional, Union

from .instructions import (
    Instruction,
    JumpInst,
    LabelInst,
    MetaInst,
    RegWriteInst,
    TestInst,
)
from .labels import Label
from .node import BasicBlockNode, BlockNode, IRBranch, IRLoop, RootNode

BIG_JUMP_PMEM_THRESHOLD = 2**11


def _needs_big_jump(pmem_size: Optional[int]) -> bool:
    return pmem_size is not None and pmem_size > BIG_JUMP_PMEM_THRESHOLD


# ---------------------------------------------------------------------------
# IRLexer: flat Instruction list  <->  list[BasicBlockNode | MetaInst]
# ---------------------------------------------------------------------------


class IRLexer:
    """Converts between flat instruction lists and chunked block lists.

    lex()     : list[Instruction]               -> list[BasicBlockNode | MetaInst]
    flatten() : list[BasicBlockNode | MetaInst] -> list[Instruction]
    """

    # -- lex -----------------------------------------------------------------

    def lex(
        self, insts: list[Instruction]
    ) -> list[Union[BasicBlockNode, MetaInst]]:
        """Chunk a flat instruction list into BasicBlockNodes and MetaInsts.

        Splitting rules:
        - LabelInst  : flushes current accumulation, then starts a new block
                       with the label as entry.
        - JumpInst   : terminates the current block (stored as branch), flushes.
        - MetaInst   : flushes current accumulation, emitted as-is.
        - everything else: accumulated into current block's insts.
        """
        result: list[Union[BasicBlockNode, MetaInst]] = []
        pending_labels: list[LabelInst] = []
        pending_insts: list[Instruction] = []
        pending_branch: Optional[JumpInst] = None

        def flush() -> None:
            nonlocal pending_labels, pending_insts, pending_branch
            if pending_labels or pending_insts or pending_branch is not None:
                result.append(
                    BasicBlockNode(
                        labels=pending_labels,
                        insts=pending_insts,
                        branch=pending_branch,
                    )
                )
            pending_labels = []
            pending_insts = []
            pending_branch = None

        for inst in insts:
            if isinstance(inst, MetaInst):
                flush()
                result.append(inst)
            elif isinstance(inst, LabelInst):
                flush()
                pending_labels = [inst]
            elif isinstance(inst, JumpInst):
                pending_branch = inst
                flush()
            else:
                pending_insts.append(inst)

        flush()
        return result

    # -- flatten -------------------------------------------------------------

    def flatten(
        self, items: list[Union[BasicBlockNode, MetaInst]]
    ) -> list[Instruction]:
        """Inverse of lex(): emit labels + insts + branch for each BasicBlockNode;
        pass MetaInst through unchanged."""
        result: list[Instruction] = []
        for item in items:
            if isinstance(item, MetaInst):
                result.append(item)
            else:
                result.extend(item.labels)
                result.extend(item.insts)
                if item.branch is not None:
                    result.append(item.branch)
        return result


# ---------------------------------------------------------------------------
# IRParser: list[BasicBlockNode | MetaInst]  <->  RootNode
# ---------------------------------------------------------------------------


class IRParser:
    """Converts between a chunked block list and a structured IR tree.

    parse()   : list[BasicBlockNode | MetaInst] -> RootNode
    unparse() : RootNode                        -> list[BasicBlockNode | MetaInst]

    parse() recognises LOOP_START/END, BRANCH_START/END, BRANCH_CASE_START/END
    MetaInst pairs and folds them into IRLoop / IRBranch nodes.

    unparse() lowers IRLoop and IRBranch back to flat BasicBlockNode lists,
    embedding MetaInst markers for round-trip fidelity.
    """

    def __init__(self, pmem_size: Optional[int] = None) -> None:
        self.pmem_size = pmem_size

    # -----------------------------------------------------------------------
    # parse
    # -----------------------------------------------------------------------

    def parse(
        self, items: list[Union[BasicBlockNode, MetaInst]]
    ) -> RootNode:
        self._check_sese(items)
        root = RootNode()
        pos = [0]
        self._parse_block(items, pos, root, end_markers=frozenset())
        return root

    # -----------------------------------------------------------------------
    # SESE validation
    # -----------------------------------------------------------------------

    def _check_sese(
        self, items: list[Union[BasicBlockNode, MetaInst]]
    ) -> None:
        """Verify that no jump from outside a structural region targets a label
        defined inside the region's control skeleton (the skip-segments between
        LOOP_START and LOOP_BODY_START, or between any BRANCH meta markers and
        BRANCH_CASE_START).  Violations break the SESE assumption that lets
        IRParser reconstruct IRLoop / IRBranch from a flat block list.
        """
        # Pass 1: collect control labels (defined in skip-segments) and record
        # which item indices belong to skip-segments.
        control_labels: set[str] = set()
        skip_indices: set[int] = set()

        depth = 0  # nesting depth of structural regions
        skip_active = False  # inside a skip-segment right now

        # We need to scan MetaInsts that appear *inside* BasicBlockNodes
        # (e.g. LOOP_START sits in a BasicBlockNode.insts).  We also need
        # LabelInsts from BasicBlockNode.labels in skip-segments.
        # Strategy: walk item by item; use MetaInst.type transitions to track
        # skip vs. body regions; harvest labels from skip-region items.

        # Build a flat timeline of (item_index, event) where event is either
        # a MetaInst type string or None (plain BasicBlockNode).
        def meta_types_in(bb: BasicBlockNode) -> list[str]:
            return [i.type for i in bb.insts if isinstance(i, MetaInst)]

        i = 0
        while i < len(items):
            item = items[i]
            if isinstance(item, MetaInst):
                # Top-level MetaInst (should not occur in well-formed input,
                # but handle gracefully by treating as skip content if active).
                if skip_active:
                    skip_indices.add(i)
                i += 1
                continue

            # BasicBlockNode: check MetaInst markers inside its insts
            assert isinstance(item, BasicBlockNode)
            metas = meta_types_in(item)

            # Determine if this whole block is part of a skip-segment
            # before processing its internal transitions.
            was_skip = skip_active

            for mt in metas:
                if mt in ("LOOP_START", "BRANCH_START"):
                    depth += 1
                    skip_active = True
                elif mt == "LOOP_BODY_START":
                    skip_active = False  # body begins; skip ends
                elif mt in ("BRANCH_CASE_START",):
                    skip_active = False
                elif mt in ("LOOP_BODY_END",):
                    skip_active = True  # back to skip (back-edge region)
                elif mt in ("LOOP_END", "BRANCH_END"):
                    depth -= 1
                    if depth == 0:
                        skip_active = False

            # The block is in skip-territory if it was skip before *or*
            # the MetaInst transition in this block initiates skip.
            in_skip = was_skip or (skip_active and depth > 0)
            if in_skip:
                skip_indices.add(i)
                for lbl in item.labels:
                    if lbl.name is not None:
                        control_labels.add(str(lbl.name))

            i += 1

        if not control_labels:
            return  # nothing to check

        # Pass 2: collect all jump targets from non-skip items.
        for idx, item in enumerate(items):
            if idx in skip_indices:
                continue
            if not isinstance(item, BasicBlockNode):
                continue
            for jump in ([item.branch] if item.branch else []):
                if jump.label is not None and str(jump.label) in control_labels:
                    raise ValueError(
                        f"IRParser: jump to control label {str(jump.label)!r} from "
                        f"outside its structural region violates SESE assumption. "
                        f"Block index {idx}: {item}"
                    )

    def _parse_block(
        self,
        items: list[Union[BasicBlockNode, MetaInst]],
        pos: list[int],
        block: BlockNode,
        end_markers: frozenset[str],
    ) -> None:
        while pos[0] < len(items):
            item = items[pos[0]]

            if isinstance(item, MetaInst) and item.type in end_markers:
                return  # do NOT consume — caller consumes the end marker

            if isinstance(item, MetaInst):
                if item.type == "LOOP_START":
                    block.append(self._parse_loop(items, pos))
                elif item.type == "BRANCH_START":
                    block.append(self._parse_branch(items, pos))
                else:
                    raise ValueError(f"IRParser.parse: unexpected MetaInst: {item}")
            else:
                pos[0] += 1
                block.append(item)

    def _consume_meta(
        self,
        items: list[Union[BasicBlockNode, MetaInst]],
        pos: list[int],
        expected_type: str,
    ) -> MetaInst:
        if pos[0] >= len(items):
            raise ValueError(
                f"IRParser: expected META {expected_type!r} but stream ended"
            )
        item = items[pos[0]]
        if not isinstance(item, MetaInst) or item.type != expected_type:
            raise ValueError(
                f"IRParser: expected META {expected_type!r}, got {item!r}"
            )
        pos[0] += 1
        return item

    def _parse_loop(
        self,
        items: list[Union[BasicBlockNode, MetaInst]],
        pos: list[int],
    ) -> IRLoop:
        start_meta = self._consume_meta(items, pos, "LOOP_START")

        # Skip physical control blocks up to LOOP_BODY_START
        while pos[0] < len(items):
            item = items[pos[0]]
            if isinstance(item, MetaInst) and item.type == "LOOP_BODY_START":
                break
            pos[0] += 1

        self._consume_meta(items, pos, "LOOP_BODY_START")

        loop = IRLoop(
            name=start_meta.name,
            counter_reg=start_meta.info["counter_reg"],
            n=start_meta.info["n"],
            range_hint=start_meta.info.get("range_hint"),
        )
        self._parse_block(items, pos, loop.body, end_markers=frozenset({"LOOP_BODY_END"}))
        self._consume_meta(items, pos, "LOOP_BODY_END")

        # Skip physical back-edge / end-label blocks up to LOOP_END
        while pos[0] < len(items):
            item = items[pos[0]]
            if isinstance(item, MetaInst) and item.type == "LOOP_END":
                break
            pos[0] += 1

        end_meta = self._consume_meta(items, pos, "LOOP_END")
        if end_meta.name != loop.name:
            raise ValueError(
                f"IRParser: mismatched LOOP_END for {loop.name!r}, got {end_meta.name!r}"
            )
        return loop

    def _parse_branch(
        self,
        items: list[Union[BasicBlockNode, MetaInst]],
        pos: list[int],
    ) -> IRBranch:
        start_meta = self._consume_meta(items, pos, "BRANCH_START")
        branch = IRBranch(
            name=start_meta.name,
            compare_reg=start_meta.info["compare_reg"],
        )

        while pos[0] < len(items):
            item = items[pos[0]]
            if isinstance(item, MetaInst) and item.type == "BRANCH_END":
                break
            if isinstance(item, MetaInst) and item.type == "BRANCH_CASE_START":
                branch.cases.append(self._parse_branch_case(items, pos))
            else:
                pos[0] += 1  # discard dispatch control

        end_meta = self._consume_meta(items, pos, "BRANCH_END")
        if end_meta.name != branch.name:
            raise ValueError(
                f"IRParser: mismatched BRANCH_END for {branch.name!r}, got {end_meta.name!r}"
            )
        return branch

    def _parse_branch_case(
        self,
        items: list[Union[BasicBlockNode, MetaInst]],
        pos: list[int],
    ) -> BlockNode:
        start_meta = self._consume_meta(items, pos, "BRANCH_CASE_START")
        case = BlockNode()
        self._parse_block(items, pos, case, end_markers=frozenset({"BRANCH_CASE_END"}))
        end_meta = self._consume_meta(items, pos, "BRANCH_CASE_END")
        if end_meta.name != start_meta.name:
            raise ValueError(
                f"IRParser: mismatched BRANCH_CASE_END expected {start_meta.name!r}, "
                f"got {end_meta.name!r}"
            )
        return case

    # -----------------------------------------------------------------------
    # unparse
    # -----------------------------------------------------------------------

    def unparse(
        self, root: RootNode
    ) -> list[Union[BasicBlockNode, MetaInst]]:
        """Lower an IR tree back to a flat list of BasicBlockNode / MetaInst.

        IRLoop and IRBranch are lowered here (logic moved from node.py).
        """
        return self._unparse_block_node(root)

    def lower_block(self, block: BlockNode) -> list[BasicBlockNode]:
        """Flatten a BlockNode into BasicBlockNodes only (no MetaInst wrappers).

        Use this when the block contains only BasicBlockNode / IRLoop / IRBranch
        children that do not need structural MetaInst markers (e.g. loop bodies
        being cloned for unrolling).  MetaInst items emitted by nested lowering
        are silently dropped — callers that need round-trip fidelity should use
        unparse() instead.
        """
        items = self._unparse_block_node(block)
        return [item for item in items if isinstance(item, BasicBlockNode)]

    def _unparse_block_node(
        self, block: BlockNode
    ) -> list[Union[BasicBlockNode, MetaInst]]:
        result: list[Union[BasicBlockNode, MetaInst]] = []
        for child in block.insts:
            if isinstance(child, BasicBlockNode):
                result.append(child)
            elif isinstance(child, IRLoop):
                result.extend(self._lower_loop(child))
            elif isinstance(child, IRBranch):
                result.extend(self._lower_branch(child))
            elif isinstance(child, BlockNode):
                result.extend(self._unparse_block_node(child))
            else:
                raise TypeError(
                    f"IRParser.unparse: unexpected node type {type(child).__name__}"
                )
        return result

    def _lower_loop(
        self, node: IRLoop
    ) -> list[Union[BasicBlockNode, MetaInst]]:
        """Lower IRLoop into BasicBlockNodes + MetaInst markers.

        Shape:
            MetaInst(LOOP_START)  [inside a BasicBlockNode.insts]
            [guard block]         -- only when n is a register string
            init block            -- REG_WR counter imm #0
            start_label block + MetaInst(LOOP_BODY_START)
            <body blocks>
            back_edge block + MetaInst(LOOP_BODY_END)
            end_label block + MetaInst(LOOP_END)
        """
        start = Label.make_new(f"{node.name}_start")
        end = Label.make_new(f"{node.name}_end")

        result: list[Union[BasicBlockNode, MetaInst]] = []

        result.append(
            BasicBlockNode(
                insts=[
                    MetaInst(
                        type="LOOP_START",
                        name=node.name,
                        info=dict(
                            counter_reg=node.counter_reg,
                            n=node.n,
                            range_hint=node.range_hint,
                        ),
                    )
                ]
            )
        )

        # Guard block (runtime n only)
        if isinstance(node.n, str):
            if _needs_big_jump(self.pmem_size):
                result.append(
                    BasicBlockNode(
                        insts=[RegWriteInst(dst="s15", src="label", label=end)],
                        branch=JumpInst(addr="s15", if_cond="Z", op=f"{node.n} - #0"),
                    )
                )
            else:
                result.append(
                    BasicBlockNode(
                        branch=JumpInst(label=end, if_cond="Z", op=f"{node.n} - #0"),
                    )
                )

        # Counter init
        result.append(
            BasicBlockNode(
                insts=[RegWriteInst(dst=node.counter_reg, src="imm", lit="#0")]
            )
        )

        # Start label + LOOP_BODY_START
        result.append(
            BasicBlockNode(
                labels=[LabelInst(name=start, can_remove=True)],
                insts=[MetaInst(type="LOOP_BODY_START", name=node.name)],
            )
        )

        # Body
        result.extend(self._unparse_block_node(node.body))

        # Back-edge: counter++ then conditional jump to start
        op_str = (
            f"{node.counter_reg} - #{node.n}"
            if isinstance(node.n, int)
            else f"{node.counter_reg} - {node.n}"
        )
        back_insts: list[Instruction] = [
            MetaInst(type="LOOP_BODY_END", name=node.name),
            RegWriteInst(dst=node.counter_reg, src="op", op=f"{node.counter_reg} + #1"),
        ]
        if _needs_big_jump(self.pmem_size):
            back_insts.append(RegWriteInst(dst="s15", src="label", label=start))
            result.append(
                BasicBlockNode(
                    insts=back_insts,
                    branch=JumpInst(addr="s15", if_cond="NS", op=op_str),
                )
            )
        else:
            result.append(
                BasicBlockNode(
                    insts=back_insts,
                    branch=JumpInst(label=start, if_cond="NS", op=op_str),
                )
            )

        # End label + LOOP_END
        result.append(
            BasicBlockNode(
                labels=[LabelInst(name=end, can_remove=True)],
                insts=[MetaInst(type="LOOP_END", name=node.name)],
            )
        )

        return result

    def _lower_branch(
        self, node: IRBranch
    ) -> list[Union[BasicBlockNode, MetaInst]]:
        """Lower IRBranch into BasicBlockNodes using binary dispatch."""
        n = len(node.cases)
        result: list[Union[BasicBlockNode, MetaInst]] = []

        result.append(
            BasicBlockNode(
                insts=[
                    MetaInst(
                        type="BRANCH_START",
                        name=node.name,
                        info=dict(compare_reg=node.compare_reg),
                    )
                ]
            )
        )

        def emit_dispatch(lo: int, hi: int) -> None:
            if hi - lo == 1:
                result.extend(self._unparse_block_node(node.cases[lo]))
                return

            mid = (lo + hi) // 2
            left_label = Label.make_new(f"{node.name}_branch_l_{lo}_{mid}")
            end_label = Label.make_new(f"{node.name}_branch_e_{lo}_{hi}")

            result.append(
                BasicBlockNode(
                    insts=[TestInst(op=f"{node.compare_reg} - #{mid}")],
                    branch=JumpInst(label=left_label, if_cond="S"),
                )
            )
            emit_dispatch(mid, hi)
            result.append(BasicBlockNode(branch=JumpInst(label=end_label)))
            result.append(
                BasicBlockNode(labels=[LabelInst(name=left_label, can_remove=True)])
            )
            emit_dispatch(lo, mid)
            result.append(
                BasicBlockNode(labels=[LabelInst(name=end_label, can_remove=True)])
            )

        emit_dispatch(0, n)

        result.append(
            BasicBlockNode(insts=[MetaInst(type="BRANCH_END", name=node.name)])
        )
        return result

