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
from .labels import Label
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

    def __init__(self, pmem_size: Optional[int] = None) -> None:
        self.pmem_size = pmem_size

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
        items = self._unparse_block_node(block)
        return [item for item in items if isinstance(item, BasicBlockNode)]

    def _unparse_node(self, node: IRNode) -> list[Union[BasicBlockNode, MetaInst]]:
        """Recursively lower a single IRNode to a flat chunk list.

        Iterates until no IRNode remains in the output so that lowering steps
        that produce intermediate IRNodes (e.g. _lower_branch → IRDispatch) are
        fully resolved in one call.
        """
        pending: list[Union[BasicBlockNode, MetaInst, IRNode]] = [node]
        result: list[Union[BasicBlockNode, MetaInst]] = []

        while pending:
            item = pending.pop(0)
            if isinstance(item, BasicBlockNode):
                result.append(item)
            elif isinstance(item, MetaInst):
                result.append(item)
            elif isinstance(item, IRLoop):
                pending[0:0] = self._lower_loop(item)
            elif isinstance(item, IRBranch):
                pending[0:0] = self._lower_branch(item)
            elif isinstance(item, IRDispatch):
                pending[0:0] = self._lower_dispatch(item)
            elif isinstance(item, BlockNode):
                pending[0:0] = list(item.insts)
            else:
                raise TypeError(
                    f"IRParser.unparse: unexpected node type {type(item).__name__}"
                )
        return result

    def _unparse_block_node(
        self, block: BlockNode
    ) -> list[Union[BasicBlockNode, MetaInst]]:
        result: list[Union[BasicBlockNode, MetaInst]] = []
        for child in block.insts:
            result.extend(self._unparse_node(child))
        return result

    def _lower_loop(self, node: IRLoop) -> list[Union[BasicBlockNode, MetaInst]]:
        lexer = IRLexer()
        start = Label.make_new(f"{node.name}_start")
        end = Label.make_new(f"{node.name}_end")

        counter = node.counter_reg
        if isinstance(node.n, int):
            n_val = Immediate(node.n)
        else:
            n_val = node.n

        op_str = AluExpr(counter, AluOp.SUB, n_val)

        pre: list[Instruction] = [
            MetaInst(
                type="LOOP_START",
                name=node.name,
                info=dict(
                    counter_reg=node.counter_reg.name,
                    n=node.n.name if isinstance(node.n, Register) else node.n,
                    range_hint=node.range_hint,
                ),
            ),
        ]
        if isinstance(node.n, Register):
            if needs_big_jump(self.pmem_size):
                pre += [
                    RegWriteInst(dst=Register("s15"), src=SrcKeyword.LABEL, label=end),
                    JumpInst(
                        addr=Register("s15"),
                        if_cond="Z",
                        op=AluExpr(node.n, AluOp.SUB, Immediate(0)),
                    ),
                ]
            else:
                pre.append(
                    JumpInst(
                        label=end,
                        if_cond="Z",
                        op=AluExpr(node.n, AluOp.SUB, Immediate(0)),
                    )
                )
        pre += [
            RegWriteInst(dst=counter, src=SrcKeyword.IMM, lit=Immediate(0)),
            LabelInst(name=start, can_remove=True),
            MetaInst(type="LOOP_BODY_START", name=node.name),
        ]

        post: list[Instruction] = [
            MetaInst(type="LOOP_BODY_END", name=node.name),
        ]
        if needs_big_jump(self.pmem_size):
            post += [
                RegWriteInst(dst=Register("s15"), src=SrcKeyword.LABEL, label=start),
                JumpInst(addr=Register("s15"), if_cond="S", op=op_str),
            ]
        else:
            post.append(JumpInst(label=start, if_cond="S", op=op_str))
        post += [
            LabelInst(name=end, can_remove=True),
            MetaInst(type="LOOP_END", name=node.name),
        ]

        return lexer.lex(pre) + self._unparse_node(node.body) + lexer.lex(post)

    def _lower_branch(self, node: IRBranch) -> list[Union[BasicBlockNode, MetaInst]]:
        """Lower an IRBranch to a flat chunk list via an IRDispatch node.

        All IRBranch sizes (including n==2) go through _lower_dispatch so that
        SimplifyDispatchPass can later apply the n==2 cond_jump optimisation at
        the IR-tree layer rather than duplicating the logic here.
        """
        n = len(node.cases)
        case_entry_labels = [
            Label.make_new(f"{node.name}_case_entry_{i}") for i in range(n)
        ]
        dispatch_node = IRDispatch(
            name=node.name,
            value_reg=node.compare_reg,
            target_labels=case_entry_labels,
        )

        result: list[Union[BasicBlockNode, MetaInst]] = []
        result.append(
            MetaInst(
                type="BRANCH_START",
                name=node.name,
                info=dict(compare_reg=node.compare_reg.name),
            )
        )
        result.extend(self._lower_dispatch(dispatch_node))

        for idx, case in enumerate(node.cases):
            case_name = str(idx)
            result.append(MetaInst(type="BRANCH_CASE_START", name=case_name))
            case_items = self._unparse_node(case)
            first_block_attached = False
            for item in case_items:
                if not first_block_attached and isinstance(item, BasicBlockNode):
                    item.labels.insert(
                        0, LabelInst(name=case_entry_labels[idx], can_remove=True)
                    )
                    first_block_attached = True
                result.append(item)
            if not first_block_attached:
                result.append(
                    BasicBlockNode(
                        labels=[LabelInst(name=case_entry_labels[idx], can_remove=True)]
                    )
                )
            result.append(MetaInst(type="BRANCH_CASE_END", name=case_name))

        result.append(MetaInst(type="BRANCH_END", name=node.name))
        return result

    def _lower_dispatch(
        self, node: IRDispatch
    ) -> list[Union[BasicBlockNode, MetaInst]]:
        """Lower an IRDispatch leaf node to a flat chunk list.

        Shape emitted (n = len(target_labels)):

            DISPATCH_START
            BasicBlockNode(branch=JUMP target_labels[-1] -if(S) -op(value_reg - #n))
                -- out-of-range guard: if value_reg >= n, jump to the last case
            BasicBlockNode(insts=[setup...], branch=JUMP s15)
                -- address computation + indirect jump
            [dispatch table island (disable_opt blocks)]
            DISPATCH_END

        Note: The guard jumps to ``target_labels[-1]`` (the last case).  Callers
        that want a specific fallback must place it at index n-1.  This is always
        the case for IRBranch lowering (the else-branch is the last case).
        """
        n = len(node.target_labels)
        table_labels = [Label.make_new(f"{node.name}_dispatch_{i}") for i in range(n)]

        result: list[Union[BasicBlockNode, MetaInst]] = []
        result.append(MetaInst(type="DISPATCH_START", name=node.name))

        # Out-of-range guard: if value_reg >= n → jump to last case.
        result.append(
            BasicBlockNode(
                branch=JumpInst(
                    label=node.target_labels[-1],
                    if_cond="S",
                    op=AluExpr(node.value_reg, AluOp.SUB, Immediate(n)),
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

        result.append(MetaInst(type="DISPATCH_END", name=node.name))
        return result
