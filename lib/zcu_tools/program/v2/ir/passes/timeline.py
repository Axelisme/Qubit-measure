from __future__ import annotations

from ..instructions import GenericInst, Instruction
from ..node import BlockNode, IRLoop, IRNode
from ..pipeline import AbsPipeLinePass, PipeLineContext


class ZeroDelayDCEPass(AbsPipeLinePass):
    """Remove lower-level zero reference-time increments."""

    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode:
        self._rewrite_node(ir)
        return ir

    def _rewrite_node(self, node: IRNode) -> None:
        if isinstance(node, IRLoop):
            self._rewrite_node(node.initial)
            self._rewrite_node(node.stop_check)
            self._rewrite_node(node.body)
            self._rewrite_node(node.update)
            self._rewrite_node(node.jump_back)
            return

        if not isinstance(node, BlockNode):
            return

        rewritten = []
        for item in node.insts:
            if isinstance(item, Instruction):
                if not _is_zero_delay_inst(item):
                    rewritten.append(item)
            elif isinstance(item, IRNode):
                self._rewrite_node(item)
                rewritten.append(item)
            else:
                rewritten.append(item)
        node.insts = rewritten


def _is_zero_delay_inst(inst: Instruction) -> bool:
    if not isinstance(inst, GenericInst):
        return False
    if inst.cmd != "TIME":
        return False
    if set(inst.args) != {"C_OP", "LIT"}:
        return False
    if inst.args.get("C_OP") != "inc_ref":
        return False

    lit = inst.args.get("LIT")
    if not isinstance(lit, str) or not lit.startswith("#"):
        return False

    try:
        return int(lit[1:]) == 0
    except ValueError:
        return False


class TimedInstructionMergePass(AbsPipeLinePass):
    """Merge adjacent reference-time increments with identical semantics."""

    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode:
        self._rewrite_node(ir)
        return ir

    def _rewrite_node(self, node: IRNode) -> None:
        if isinstance(node, IRLoop):
            self._rewrite_node(node.initial)
            self._rewrite_node(node.stop_check)
            self._rewrite_node(node.body)
            self._rewrite_node(node.update)
            self._rewrite_node(node.jump_back)
            return

        if not isinstance(node, BlockNode):
            return

        rewritten: list[Instruction | IRNode] = []
        pending_inst: GenericInst | None = None
        pending_value = 0
        pending_count = 0

        def flush_pending() -> None:
            nonlocal pending_inst, pending_value, pending_count
            if pending_inst is None:
                return
            if pending_count == 1:
                rewritten.append(pending_inst)
            else:
                rewritten.append(
                    GenericInst(
                        cmd="TIME",
                        args={"C_OP": "inc_ref", "LIT": f"#{pending_value}"},
                        line=pending_inst.line,
                        p_addr=pending_inst.p_addr,
                    )
                )
            pending_inst = None
            pending_value = 0
            pending_count = 0

        for item in node.insts:
            if isinstance(item, IRNode):
                flush_pending()
                self._rewrite_node(item)
                rewritten.append(item)
                continue

            merge_value = _positive_time_increment(item)
            if merge_value is None:
                flush_pending()
                rewritten.append(item)
                continue

            if not isinstance(item, GenericInst):
                flush_pending()
                rewritten.append(item)
                continue

            if pending_inst is None:
                pending_inst = item
                pending_value = merge_value
                pending_count = 1
            else:
                pending_value += merge_value
                pending_count += 1

        flush_pending()
        node.insts = rewritten


def _positive_time_increment(inst: Instruction) -> int | None:
    if not isinstance(inst, GenericInst):
        return None
    if inst.cmd != "TIME":
        return None
    if set(inst.args) != {"C_OP", "LIT"}:
        return None
    if inst.args.get("C_OP") != "inc_ref":
        return None

    lit = inst.args.get("LIT")
    if not isinstance(lit, str) or not lit.startswith("#"):
        return None

    try:
        value = int(lit[1:])
    except ValueError:
        return None

    if value <= 0:
        return None
    return value
