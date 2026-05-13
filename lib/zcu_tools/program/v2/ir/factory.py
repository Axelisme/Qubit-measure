from __future__ import annotations

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
from .node import BasicBlockNode, BlockNode, IRBranch, IRLoop, RootNode
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
        in_fix_addr = False

        def flush() -> None:
            nonlocal pending_labels, pending_insts, pending_branch
            if pending_labels or pending_insts or pending_branch is not None:
                bb = BasicBlockNode(
                    labels=pending_labels,
                    insts=pending_insts,
                    branch=pending_branch,
                )
                bb.fix_addr_size = in_fix_addr
                result.append(bb)
            pending_labels = []
            pending_insts = []
            pending_branch = None

        for inst in insts:
            if isinstance(inst, MetaInst):
                if inst.type == "FIX_ADDR_START":
                    flush()
                    in_fix_addr = True
                elif inst.type == "FIX_ADDR_END":
                    flush()
                    in_fix_addr = False
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
                if item.fix_addr_size:
                    result.append(MetaInst(type="FIX_ADDR_START", name=""))
                result.extend(item.labels)
                result.extend(item.insts)
                if item.branch is not None:
                    result.append(item.branch)
                if item.fix_addr_size:
                    result.append(MetaInst(type="FIX_ADDR_END", name=""))
        return result


class IRParser:
    """Converts between a chunked block list and a structured IR tree."""

    def __init__(self, pmem_size: Optional[int] = None) -> None:
        self.pmem_size = pmem_size

    def parse(self, items: list[Union[BasicBlockNode, MetaInst]]) -> RootNode:
        self._check_sese(items)
        root = RootNode()
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
            items, pos, loop.body, end_markers=frozenset({"LOOP_BODY_END"})
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

        if all(case_name.isdigit() for case_name, _case in parsed_cases):
            parsed_cases.sort(key=lambda pair: int(pair[0]))
        branch.cases = [case for _case_name, case in parsed_cases]

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

    def unparse(self, root: RootNode) -> list[Union[BasicBlockNode, MetaInst]]:
        return self._unparse_block_node(root)

    def lower_block(self, block: BlockNode) -> list[BasicBlockNode]:
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

        return lexer.lex(pre) + self._unparse_block_node(node.body) + lexer.lex(post)

    def _lower_branch(self, node: IRBranch) -> list[Union[BasicBlockNode, MetaInst]]:
        if len(node.cases) == 2:
            return self._lower_branch_binary(node)
        return self._lower_branch_dispatch(node)

    def _lower_branch_binary(
        self, node: IRBranch
    ) -> list[Union[BasicBlockNode, MetaInst]]:
        """Lower a 2-case IRBranch using a single conditional jump.

        Shape (small and big PMEM alike — no dispatch table needed):

            BRANCH_START
            BasicBlockNode(branch=JUMP else_label -if(NZ) -op(compare_reg - #0))
            BRANCH_CASE_START("0")
              [case 0 body]
              BasicBlockNode(branch=JUMP end_label)   # eliminated by BranchEliminationPass
            BRANCH_CASE_END("0")
            BRANCH_CASE_START("1")
              BasicBlockNode(labels=[else_label], ...)  # case 1 body
            BRANCH_CASE_END("1")
            BasicBlockNode(labels=[end_label])
            BRANCH_END

        compare_reg == 0  → Z flag set, NZ not taken → fallthrough to case 0
        compare_reg != 0  → NZ taken → jump to case 1 (else)
        """
        result: list[Union[BasicBlockNode, MetaInst]] = []

        else_label = Label.make_new(f"{node.name}_case_entry_1")
        end_label = Label.make_new(f"{node.name}_end")
        case0_entry_label = Label.make_new(f"{node.name}_case_entry_0")

        result.append(
            MetaInst(
                type="BRANCH_START",
                name=node.name,
                info=dict(compare_reg=node.compare_reg.name),
            )
        )

        # Dispatch: single cond jump — no dispatch table, no s15 manipulation.
        result.append(
            BasicBlockNode(
                branch=JumpInst(
                    label=else_label,
                    if_cond="NZ",
                    op=AluExpr(node.compare_reg, AluOp.SUB, Immediate(0)),
                )
            )
        )

        def _emit_case(
            idx: int,
            entry_label: Label,
            case_items: list[Union[BasicBlockNode, MetaInst]],
        ) -> None:
            result.append(MetaInst(type="BRANCH_CASE_START", name=str(idx)))
            first_block_attached = False
            for item in case_items:
                if not first_block_attached and isinstance(item, BasicBlockNode):
                    item.labels.insert(0, LabelInst(name=entry_label, can_remove=True))
                    first_block_attached = True
                result.append(item)
            if not first_block_attached:
                result.append(
                    BasicBlockNode(
                        labels=[LabelInst(name=entry_label, can_remove=True)]
                    )
                )
            result.append(MetaInst(type="BRANCH_CASE_END", name=str(idx)))

        # Case 0: emit body, then add an unconditional jump to end_label so
        # control does not fall into case 1.  BranchEliminationPass will remove
        # this jump if end_label immediately follows.
        case0_items = self._unparse_block_node(node.cases[0])

        # Find the last BasicBlockNode produced by case 0 before _emit_case
        # appends the BRANCH_CASE_END marker.  If it has no branch we attach
        # the end jump directly; otherwise we append a new stub block.
        last_case0_bb: Optional[BasicBlockNode] = None
        for item in reversed(case0_items):
            if isinstance(item, BasicBlockNode):
                last_case0_bb = item
                break

        if (
            last_case0_bb is not None
            and last_case0_bb.branch is None
            and not last_case0_bb.fix_addr_size
        ):
            last_case0_bb.branch = JumpInst(label=end_label)
            _emit_case(0, case0_entry_label, case0_items)
        else:
            _emit_case(0, case0_entry_label, case0_items)
            # Insert end-jump stub before the BRANCH_CASE_END marker.
            case_end_idx = next(
                i
                for i in range(len(result) - 1, -1, -1)
                if isinstance(result[i], MetaInst)
                and result[i].type == "BRANCH_CASE_END"  # type: ignore[union-attr]
            )
            result.insert(
                case_end_idx, BasicBlockNode(branch=JumpInst(label=end_label))
            )

        # Case 1: else branch.
        case1_items = self._unparse_block_node(node.cases[1])
        _emit_case(1, else_label, case1_items)

        # End landing pad is placed AFTER BRANCH_END so _check_sese does not
        # classify it as a dispatch-region control label.
        result.append(MetaInst(type="BRANCH_END", name=node.name))
        result.append(
            BasicBlockNode(labels=[LabelInst(name=end_label, can_remove=True)])
        )
        return result

    def _lower_branch_dispatch(
        self, node: IRBranch
    ) -> list[Union[BasicBlockNode, MetaInst]]:
        n = len(node.cases)
        result: list[Union[BasicBlockNode, MetaInst]] = []
        case_entry_labels = [
            Label.make_new(f"{node.name}_case_entry_{i}") for i in range(n)
        ]
        table_labels = [Label.make_new(f"{node.name}_dispatch_{i}") for i in range(n)]

        result.append(
            MetaInst(
                type="BRANCH_START",
                name=node.name,
                info=dict(compare_reg=node.compare_reg.name),
            )
        )
        result.append(
            BasicBlockNode(
                insts=emit_dispatch_address_setup(
                    index_reg=node.compare_reg.name,
                    table_base=table_labels[0],
                    pmem_size=self.pmem_size,
                ),
                branch=JumpInst(addr=Register("s15")),
            )
        )
        result.extend(
            build_dispatch_table_island(
                table_labels=table_labels,
                target_labels=case_entry_labels,
                pmem_size=self.pmem_size,
            )
        )

        for idx, case in enumerate(node.cases):
            case_name = str(idx)
            result.append(MetaInst(type="BRANCH_CASE_START", name=case_name))
            case_items = self._unparse_block_node(case)
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
