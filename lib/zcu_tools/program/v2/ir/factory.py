from __future__ import annotations

from typing import cast

from typing_extensions import Optional, Union

from .dispatch import (
    build_dispatch_table_island,
    emit_dispatch_address_setup,
    needs_big_jump,
)
from .instructions import (
    BaseInst,
    Instruction,
    JumpInst,
    LabelInst,
    MetaInst,
    RegWriteInst,
)
from .labels import LabelRef, make_label
from .node import BasicBlockNode, BlockNode, IRBranch, IRDispatch, IRLoop, IRNode
from .operands import AluExpr, AluOp, Immediate, Register, SrcKeyword


class IRLexer:
    """Converts between flat instruction lists and chunked block lists.

    lex()     : list[Instruction]               -> list[BasicBlockNode | MetaInst]
    flatten() : list[BasicBlockNode | MetaInst] -> list[Instruction]
    """

    def lex(self, insts: list[Instruction]) -> list[Union[BasicBlockNode, MetaInst]]:
        result: list[Union[BasicBlockNode, MetaInst]] = []
        pending_labels: list[LabelInst] = []
        pending_insts: list[BaseInst] = []
        pending_branch: Optional[JumpInst] = None
        in_disable_opt = False

        def flush() -> None:
            nonlocal pending_labels, pending_insts, pending_branch
            if pending_labels or pending_insts or pending_branch is not None:
                bb = BasicBlockNode(
                    labels=pending_labels,
                    insts=pending_insts,
                    branch=pending_branch,
                )
                bb.disable_opt = in_disable_opt
                result.append(bb)
            pending_labels = []
            pending_insts = []
            pending_branch = None

        for inst in insts:
            if isinstance(inst, MetaInst):
                if inst.type == "DISABLE_OPT_START":
                    flush()
                    in_disable_opt = True
                elif inst.type == "DISABLE_OPT_END":
                    flush()
                    in_disable_opt = False
                else:
                    flush()
                    result.append(inst)
            elif isinstance(inst, LabelInst):
                flush()
                pending_labels = [inst]
            elif isinstance(inst, JumpInst):
                pending_branch = inst
                flush()
            else:
                assert isinstance(inst, BaseInst)
                pending_insts.append(inst)

        flush()
        return result

    def flatten(
        self, items: list[Union[BasicBlockNode, MetaInst]]
    ) -> list[Instruction]:
        result: list[Instruction] = []
        for item in items:
            if isinstance(item, MetaInst):
                result.append(item)
            else:
                if item.disable_opt:
                    result.append(MetaInst(type="DISABLE_OPT_START", name=""))
                result.extend(item.labels)
                result.extend(item.insts)
                if item.branch is not None:
                    result.append(item.branch)
                if item.disable_opt:
                    result.append(MetaInst(type="DISABLE_OPT_END", name=""))
        return result


class IRParser:
    """Converts between a chunked block list and a structured IR tree."""

    def __init__(
        self,
        pmem_size: Optional[int] = None,
    ) -> None:
        self.pmem_size = pmem_size
        self.allocated: set[str] = set()

    def parse(self, items: list[Union[BasicBlockNode, MetaInst]]) -> BlockNode:
        self._check_sese(items)
        root = BlockNode()
        pos = [0]
        self._parse_block(items, pos, root, end_markers=frozenset())
        return root

    def _check_sese(self, items: list[Union[BasicBlockNode, MetaInst]]) -> None:
        control_labels: set[str] = set()
        skip_indices: set[int] = set()

        depth = 0
        skip_active = False

        i = 0
        while i < len(items):
            item = items[i]
            if isinstance(item, MetaInst):
                mt = item.type
                if mt in ("LOOP_START", "BRANCH_START"):
                    depth += 1
                    skip_active = True
                elif mt in ("LOOP_BODY_START", "BRANCH_CASE_START"):
                    skip_active = False
                elif mt == "LOOP_BODY_END":
                    skip_active = True
                elif mt == "BRANCH_CASE_END":
                    if depth > 0:
                        skip_active = True
                elif mt in ("LOOP_END", "BRANCH_END"):
                    depth -= 1
                    if depth < 0:
                        raise ValueError(
                            f"IRParser: unexpected META {mt!r} without matching start"
                        )
                    skip_active = depth > 0
            else:
                assert isinstance(item, BasicBlockNode)
                if skip_active and depth > 0:
                    skip_indices.add(i)
                    for lbl in item.labels:
                        control_labels.add(str(lbl.name))
            i += 1

        if depth != 0:
            raise ValueError("IRParser: unbalanced structural META markers")

        if not control_labels:
            return

        for idx, item in enumerate(items):
            if idx in skip_indices:
                continue
            if not isinstance(item, BasicBlockNode):
                continue
            for jump in [item.branch] if item.branch else []:
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
                return

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
            raise ValueError(f"IRParser: expected META {expected_type!r}, got {item!r}")
        pos[0] += 1
        return item

    def _parse_loop(
        self,
        items: list[Union[BasicBlockNode, MetaInst]],
        pos: list[int],
    ) -> IRLoop:
        start_meta = self._consume_meta(items, pos, "LOOP_START")

        while pos[0] < len(items):
            item = items[pos[0]]
            if isinstance(item, MetaInst) and item.type == "LOOP_BODY_START":
                break
            pos[0] += 1

        self._consume_meta(items, pos, "LOOP_BODY_START")

        n_raw = start_meta.info["n"]
        loop = IRLoop(
            name=start_meta.name,
            counter_reg=Register(start_meta.info["counter_reg"]),
            n=Register(n_raw) if isinstance(n_raw, str) else n_raw,
            body=BlockNode(),
            range_hint=start_meta.info.get("range_hint"),
        )
        self._parse_block(
            items,
            pos,
            cast(BlockNode, loop.body),
            end_markers=frozenset({"LOOP_BODY_END"}),
        )
        self._consume_meta(items, pos, "LOOP_BODY_END")

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
            compare_reg=Register(start_meta.info["compare_reg"]),
            cases=[],
        )
        parsed_cases: list[tuple[str, BlockNode]] = []

        while pos[0] < len(items):
            item = items[pos[0]]
            if isinstance(item, MetaInst) and item.type == "BRANCH_END":
                break
            if isinstance(item, MetaInst) and item.type == "BRANCH_CASE_START":
                parsed_cases.append(self._parse_branch_case(items, pos))
            else:
                pos[0] += 1

        if not parsed_cases:
            raise ValueError(
                f"IRParser: BRANCH {branch.name!r} does not contain any cases"
            )

        if all(name.isdigit() for name, _ in parsed_cases):
            parsed_cases.sort(key=lambda pair: int(pair[0]))
        branch.cases = [case for _, case in parsed_cases]

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
    ) -> tuple[str, BlockNode]:
        start_meta = self._consume_meta(items, pos, "BRANCH_CASE_START")
        case = BlockNode()
        self._parse_block(
            items,
            pos,
            case,
            end_markers=frozenset({"BRANCH_CASE_END", "BRANCH_END"}),
        )
        end_meta = self._consume_meta(items, pos, "BRANCH_CASE_END")
        if end_meta.name != start_meta.name:
            raise ValueError(
                f"IRParser: mismatched BRANCH_CASE_END expected {start_meta.name!r}, "
                f"got {end_meta.name!r}"
            )
        return start_meta.name, case

    def unparse(self, root: BlockNode) -> list[Union[BasicBlockNode, MetaInst]]:
        return self._unparse_block_node(root)

    def lower_block(self, block: BlockNode) -> list[BasicBlockNode]:
        """Recursively lower a BlockNode to flat BasicBlockNodes without MetaInst markers."""
        return self._lower_node_flat(block)

    def _lower_node_flat(self, node: IRNode) -> list[BasicBlockNode]:
        """Recursively lower an IRNode to flat BasicBlockNodes (no MetaInst)."""
        if isinstance(node, BasicBlockNode):
            return [node]
        if isinstance(node, BlockNode):
            result: list[BasicBlockNode] = []
            for child in node.insts:
                result.extend(self._lower_node_flat(child))
            return result
        if isinstance(node, IRLoop):
            body_chunks = self._lower_node_flat(cast(BlockNode, node.body))
            return self._lower_loop(node, body_chunks)
        if isinstance(node, IRBranch):
            case_chunks_list = [
                self._lower_node_flat(cast(BlockNode, case)) for case in node.cases
            ]
            return self._lower_branch(node, case_chunks_list)
        if isinstance(node, IRDispatch):
            return self._lower_dispatch(node)
        raise TypeError(
            f"IRParser._lower_node_flat: unexpected node type {type(node).__name__}"
        )

    def _unparse_node(self, node: IRNode) -> list[Union[BasicBlockNode, MetaInst]]:
        """Recursively lower a single IRNode to a flat chunk list (with MetaInst markers).

        Used by the legacy unparse() path so that IRParser.parse() can reconstruct
        the IR tree for the U-shape iteration pipeline.
        """
        if isinstance(node, BasicBlockNode):
            return [node]
        if isinstance(node, BlockNode):
            result: list[Union[BasicBlockNode, MetaInst]] = []
            for child in node.insts:
                result.extend(self._unparse_node(child))
            return result
        if isinstance(node, IRLoop):
            # Build the full lowered loop (guard/init/start_label/body/back_edge/end_label)
            # and wrap it in MetaInst markers so parse() can reconstruct the IRLoop.
            # parse() scans: skips LOOP_START→LOOP_BODY_START, reads body to LOOP_BODY_END.
            body_chunks = self._lower_node_flat(cast(BlockNode, node.body))
            flat = self._lower_loop(node, body_chunks)

            # Identify the boundary between pre-body (guard/init/start_label) and
            # body+post (body copies + back_edge + end_label) by finding which flat
            # blocks originated from body_chunks.
            body_chunk_ids = {id(b) for b in body_chunks}
            split = next(
                (i for i, b in enumerate(flat) if id(b) in body_chunk_ids),
                len(flat),
            )
            pre_loop = flat[:split]
            body_and_post = flat[split:]

            # Find end of body section (last body block).
            last_body = max(
                (i for i, b in enumerate(body_and_post) if id(b) in body_chunk_ids),
                default=-1,
            )
            post_loop = body_and_post[last_body + 1 :]

            # Use the original _unparse_node output for body so nested structures
            # get their MetaInst markers too.
            body_items = self._unparse_node(cast(BlockNode, node.body))

            return [
                MetaInst(
                    type="LOOP_START",
                    name=node.name,
                    info=dict(
                        counter_reg=node.counter_reg.name,
                        n=node.n.name if isinstance(node.n, Register) else node.n,
                        range_hint=node.range_hint,
                    ),
                ),
                *pre_loop,
                MetaInst(type="LOOP_BODY_START", name=node.name),
                *body_items,
                MetaInst(type="LOOP_BODY_END", name=node.name),
                *post_loop,
                MetaInst(type="LOOP_END", name=node.name),
            ]
        if isinstance(node, IRBranch):
            n = len(node.cases)
            case_entry_labels = [
                make_label(f"{node.name}_case_entry_{i}", self.allocated)
                for i in range(n)
            ]
            end_label = make_label(f"{node.name}_end", self.allocated)
            dispatch_node = IRDispatch(
                name=node.name,
                value_reg=node.compare_reg,
                target_labels=case_entry_labels,
            )
            result_branch: list[Union[BasicBlockNode, MetaInst]] = [
                MetaInst(
                    type="BRANCH_START",
                    name=node.name,
                    info=dict(compare_reg=node.compare_reg.name),
                ),
                MetaInst(type="DISPATCH_START", name=node.name),
                *self._lower_dispatch(dispatch_node),
                MetaInst(type="DISPATCH_END", name=node.name),
            ]
            for idx, case in enumerate(node.cases):
                is_last = idx == n - 1
                result_branch.append(MetaInst(type="BRANCH_CASE_START", name=str(idx)))
                case_items = self._unparse_node(case)
                first_block_attached = False
                for item in case_items:
                    if not first_block_attached and isinstance(item, BasicBlockNode):
                        item.labels.insert(
                            0,
                            LabelInst(name=case_entry_labels[idx], can_remove=True),
                        )
                        first_block_attached = True
                    result_branch.append(item)
                if not first_block_attached:
                    result_branch.append(
                        BasicBlockNode(
                            labels=[
                                LabelInst(name=case_entry_labels[idx], can_remove=True)
                            ]
                        )
                    )
                if not is_last:
                    if needs_big_jump(self.pmem_size):
                        result_branch.append(
                            BasicBlockNode(
                                insts=[
                                    RegWriteInst(
                                        dst=Register("s15"),
                                        src=SrcKeyword.LABEL,
                                        label=LabelRef(end_label),
                                    )
                                ],
                                branch=JumpInst(addr=Register("s15")),
                            )
                        )
                    else:
                        result_branch.append(
                            BasicBlockNode(branch=JumpInst(label=LabelRef(end_label)))
                        )
                result_branch.append(MetaInst(type="BRANCH_CASE_END", name=str(idx)))
            result_branch.append(MetaInst(type="BRANCH_END", name=node.name))
            result_branch.append(
                BasicBlockNode(labels=[LabelInst(name=end_label, can_remove=True)])
            )
            return result_branch
        if isinstance(node, IRDispatch):
            flat_dispatch = self._lower_dispatch(node)
            return [
                MetaInst(type="DISPATCH_START", name=node.name),
                *flat_dispatch,
                MetaInst(type="DISPATCH_END", name=node.name),
            ]
        raise TypeError(
            f"IRParser._unparse_node: unexpected node type {type(node).__name__}"
        )

    def _unparse_block_node(
        self, block: BlockNode
    ) -> list[Union[BasicBlockNode, MetaInst]]:
        result: list[Union[BasicBlockNode, MetaInst]] = []
        for child in block.insts:
            result.extend(self._unparse_node(child))
        return result

    def _lower_loop(
        self, node: IRLoop, body_chunks: list[BasicBlockNode]
    ) -> list[BasicBlockNode]:
        """Lower an IRLoop to flat BasicBlockNodes.

        Shape (do-while):
            [guard_bb?]          <- present only for register-driven n
            init_bb              <- REG_WR counter imm #0
            start_label_bb       <- entry label for back-edge
            body_chunks...       <- caller supplies already-lowered body
            back_edge_bb         <- JUMP start -if(S) -op(counter - n)
            end_label_bb         <- end label (n==0 escape)
        """
        lexer = IRLexer()
        start = make_label(f"{node.name}_start", self.allocated)
        end = make_label(f"{node.name}_end", self.allocated)

        counter = node.counter_reg
        if isinstance(node.n, int):
            n_val: Union[Immediate, Register] = Immediate(node.n)
        else:
            n_val = node.n

        op_str = AluExpr(counter, AluOp.SUB, n_val)

        pre: list[Instruction] = []
        if isinstance(node.n, Register):
            if needs_big_jump(self.pmem_size):
                pre += [
                    RegWriteInst(
                        dst=Register("s15"), src=SrcKeyword.LABEL, label=LabelRef(end)
                    ),
                    JumpInst(
                        addr=Register("s15"),
                        if_cond="Z",
                        op=AluExpr(node.n, AluOp.SUB, Immediate(0)),
                    ),
                ]
            else:
                pre.append(
                    JumpInst(
                        label=LabelRef(end),
                        if_cond="Z",
                        op=AluExpr(node.n, AluOp.SUB, Immediate(0)),
                    )
                )
        pre += [
            RegWriteInst(dst=counter, src=SrcKeyword.IMM, lit=Immediate(0)),
            LabelInst(name=start, can_remove=True),
        ]

        post: list[Instruction] = []
        if needs_big_jump(self.pmem_size):
            post += [
                RegWriteInst(
                    dst=Register("s15"), src=SrcKeyword.LABEL, label=LabelRef(start)
                ),
                JumpInst(addr=Register("s15"), if_cond="S", op=op_str),
            ]
        else:
            post.append(JumpInst(label=LabelRef(start), if_cond="S", op=op_str))
        post.append(LabelInst(name=end, can_remove=True))

        pre_blocks = lexer.lex(pre)
        post_blocks = lexer.lex(post)
        return (
            [b for b in pre_blocks if isinstance(b, BasicBlockNode)]
            + body_chunks
            + [b for b in post_blocks if isinstance(b, BasicBlockNode)]
        )

    def _lower_branch(
        self, node: IRBranch, case_chunks_list: list[list[BasicBlockNode]]
    ) -> list[BasicBlockNode]:
        """Lower an IRBranch to flat BasicBlockNodes.

        The dispatch table is emitted first (guard + indirect jump + table island),
        then each case body with its entry label prepended.  caller supplies
        case_chunks_list[i] as the already-lowered flat chunks for cases[i].

        Every case except the last gets an unconditional jump to the branch-end
        label so that cases do not fall through into the next case body.
        """
        n = len(node.cases)
        case_entry_labels = [
            make_label(f"{node.name}_case_entry_{i}", self.allocated) for i in range(n)
        ]
        end_label = make_label(f"{node.name}_end", self.allocated)

        dispatch_node = IRDispatch(
            name=node.name,
            value_reg=node.compare_reg,
            target_labels=case_entry_labels,
        )

        result: list[BasicBlockNode] = list(self._lower_dispatch(dispatch_node))

        for idx, case_chunks in enumerate(case_chunks_list):
            is_last = idx == n - 1
            if case_chunks:
                case_chunks[0].labels.insert(
                    0, LabelInst(name=case_entry_labels[idx], can_remove=True)
                )
                result.extend(case_chunks)
            else:
                result.append(
                    BasicBlockNode(
                        labels=[LabelInst(name=case_entry_labels[idx], can_remove=True)]
                    )
                )
            if not is_last:
                if needs_big_jump(self.pmem_size):
                    result.append(
                        BasicBlockNode(
                            insts=[
                                RegWriteInst(
                                    dst=Register("s15"),
                                    src=SrcKeyword.LABEL,
                                    label=LabelRef(end_label),
                                )
                            ],
                            branch=JumpInst(addr=Register("s15")),
                        )
                    )
                else:
                    result.append(
                        BasicBlockNode(branch=JumpInst(label=LabelRef(end_label)))
                    )

        result.append(
            BasicBlockNode(labels=[LabelInst(name=end_label, can_remove=True)])
        )
        return result

    def _lower_dispatch(self, node: IRDispatch) -> list[BasicBlockNode]:
        """Lower an IRDispatch leaf node to flat BasicBlockNodes (guard + table stubs).

        Shape emitted (n = len(target_labels)):

            BasicBlockNode(branch=JUMP target_labels[-1] -if(S) -op(value_reg - #n))
                -- out-of-range guard: if value_reg >= n, jump to the last case
            BasicBlockNode(insts=[setup...], branch=JUMP s15)
                -- address computation + indirect jump
            [dispatch table island (disable_opt blocks)]

        Note: The guard jumps to ``target_labels[-1]`` (the last case).  Callers
        that want a specific fallback must place it at index n-1.  This is always
        the case for IRBranch lowering (the else-branch is the last case).
        Case bodies are NOT emitted here; the caller (_lower_branch) appends them.
        """
        n = len(node.target_labels)
        table_labels = [
            make_label(f"{node.name}_dispatch_{i}", self.allocated) for i in range(n)
        ]

        result: list[BasicBlockNode] = []

        # Out-of-range guard: if value_reg >= n → jump to last case.
        last_label = node.target_labels[-1]
        op_guard = AluExpr(node.value_reg, AluOp.SUB, Immediate(n))
        if needs_big_jump(self.pmem_size):
            result.append(
                BasicBlockNode(
                    insts=[
                        RegWriteInst(
                            dst=Register("s15"),
                            src=SrcKeyword.LABEL,
                            label=LabelRef(last_label),
                        )
                    ],
                    branch=JumpInst(addr=Register("s15"), if_cond="S", op=op_guard),
                )
            )
        else:
            result.append(
                BasicBlockNode(
                    branch=JumpInst(
                        label=LabelRef(last_label), if_cond="S", op=op_guard
                    )
                )
            )

        # Address computation + indirect jump.
        result.append(
            BasicBlockNode(
                insts=emit_dispatch_address_setup(
                    index_reg=node.value_reg.name,
                    table_base=table_labels[0],
                    pmem_size=self.pmem_size,
                ),
                branch=JumpInst(addr=Register("s15")),
            )
        )

        result.extend(
            build_dispatch_table_island(
                table_labels=table_labels,
                target_labels=node.target_labels,
                pmem_size=self.pmem_size,
            )
        )

        return result
