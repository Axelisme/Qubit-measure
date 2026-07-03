from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import cast

from .factory import IRLexer, IRParser
from .instructions import BaseInst, Instruction, MetaInst, RegWriteInst
from .labels import Label
from .node import BasicBlockNode, BlockNode, IRNode

logger = logging.getLogger(__name__)

# Re-exported for backward compatibility with external imports.
__all__ = [
    "AbsChunkPass",
    "AbsChunkListPass",
    "AbsIRTreePass",
    "PipeLineConfig",
    "PipeLineContext",
    "IRPipeLine",
    "make_default_pipeline",
    "DEFAULT_PIPELINE_CONFIG",
]

ChunkList = list[BasicBlockNode | MetaInst]


@dataclass
class PipeLineConfig:
    pmem_capacity: int = 4096
    disable_all_opt: bool = False

    max_opt_iterations: int = 8

    # Hard cap on the unroll factor k. For register-driven loops k is also
    # rounded down to the nearest power of 2.
    max_unroll_factor: int = 32

    # Unified cycle cost model
    cost_default: int = 1
    cost_wmem: int = 2
    cost_dmem: int = 2
    cost_jump_flush: int = 40


# Global default configuration for quick debugging and toggling optimization options.
DEFAULT_PIPELINE_CONFIG = PipeLineConfig()


@dataclass
class PipeLineContext:
    config: PipeLineConfig
    pmem_budget: int
    available_regs: set[str] = field(default_factory=set)
    allocated_names: set[str] = field(default_factory=set)

    # dmem dispatch tables. `dmem_base_offset` is the dmem index at
    # which IR-generated dispatch tables start (set by the caller from the
    # current dmem buffer length). `dmem_tables` is filled by the resolve step:
    # an ordered list of each table's entry-label list, contiguous from
    # `dmem_base_offset`. The caller resolves the labels to program addresses
    # after linking and appends them to dmem.
    dmem_base_offset: int = 0
    dmem_tables: list[list[Label]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pass interfaces
# ---------------------------------------------------------------------------


class AbsChunkPass(ABC):
    """Per-block optimization pass: correctness depends only on the block's own content.

    Subclasses see one ``BasicBlockNode`` at a time and must not rely on the
    global ordering or jump-reference structure of the surrounding chunk list.
    Examples: ``IncRegMergePass``, ``TimedMergePass``, ``ZeroDelayDCEPass``.
    """

    @abstractmethod
    def process(
        self, chunks: ChunkList, ctx: PipeLineContext
    ) -> tuple[ChunkList, bool]: ...


class AbsChunkListPass(ABC):
    """Global-structure optimization pass: correctness requires the full chunk list.

    Subclasses may inspect jump references, adjacent-block relationships, or
    reachability across the entire chunk list.  They receive (and may mutate)
    the complete ``ChunkList`` including any remaining ``MetaInst`` entries.
    Examples: ``DeadLabelEliminationPass``, ``BranchEliminationPass``,
    ``BlockMergePass``, ``UnreachableEliminationPass``.
    """

    @abstractmethod
    def process(
        self, chunks: ChunkList, ctx: PipeLineContext
    ) -> tuple[ChunkList, bool]: ...


class AbsIRTreePass(ABC):
    """Post-order IR tree pass: called on IRLoop / IRBranch / IRDispatch nodes.

    ``transform`` receives:
      - ``node``: the IRNode being considered (its children are already
        post-order converged when the pass is invoked)
      - ``ctx``: pipeline context

    Return values:
      - ``None``: the pass did not change the node; the pipeline tries the
        next pass and leaves ``node`` untouched.
      - ``IRNode``: a replacement subtree. The pipeline re-recurses into it,
        so any structural nodes the replacement contains get lowered too.

    A pass that needs information about the *un-lowered* subtree (e.g. an
    unroll heuristic that estimates body size) must compute it at the start
    of ``transform``: at that point children are converged but still
    structural, not yet flattened.
    """

    @abstractmethod
    def transform(
        self,
        node: IRNode,
        ctx: PipeLineContext,
    ) -> IRNode | None: ...


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _validate_fixed_block_words(
    block: BasicBlockNode, *, before_addr_size: int
) -> None:
    block.__post_init__()
    if not block.disable_opt:
        raise ValueError(
            "Pass violated disable_opt invariant: fixed block lost disable_opt=True."
        )
    if block.addr_size != before_addr_size:
        raise ValueError(
            "Pass violated disable_opt invariant: "
            "block program-memory word count changed."
        )


def _collect_fixed_blocks(node: IRNode) -> dict[int, tuple[BasicBlockNode, int]]:
    from .node import BlockNode, IRBranch, IRLoop

    if isinstance(node, BasicBlockNode):
        return {id(node): (node, node.addr_size)} if node.disable_opt else {}
    result: dict[int, tuple[BasicBlockNode, int]] = {}
    if isinstance(node, BlockNode):
        for child in node.insts:
            result.update(_collect_fixed_blocks(child))
    elif isinstance(node, IRLoop):
        result.update(_collect_fixed_blocks(node.body))
    elif isinstance(node, IRBranch):
        for case in node.cases:
            result.update(_collect_fixed_blocks(case))
    return result


def _validate_fixed_blocks_preserved(
    before: dict[int, tuple[BasicBlockNode, int]],
    node_or_chunks: IRNode | ChunkList,
) -> None:
    if not before:
        return
    if isinstance(node_or_chunks, list):
        after: dict[int, tuple[BasicBlockNode, int]] = {
            id(chunk): (chunk, chunk.addr_size)
            for chunk in node_or_chunks
            if isinstance(chunk, BasicBlockNode)
        }
    else:
        after = _collect_fixed_blocks(node_or_chunks)
    missing = set(before) - set(after)
    if missing:
        raise ValueError(
            "Pass violated disable_opt invariant: fixed block was removed or replaced."
        )
    for block_id, (_, before_addr_size) in before.items():
        block = after[block_id][0]
        _validate_fixed_block_words(block, before_addr_size=before_addr_size)


def _run_passes(
    passes: Sequence[AbsChunkPass | AbsChunkListPass],
    chunks: ChunkList,
    ctx: PipeLineContext,
) -> tuple[ChunkList, bool]:
    """Run a homogeneous list of chunk passes, validating disable_opt invariants."""
    changed = False
    for chunk_pass in passes:
        fixed_before = {
            id(chunk): (chunk, chunk.addr_size)
            for chunk in chunks
            if isinstance(chunk, BasicBlockNode) and chunk.disable_opt
        }
        chunks, pass_changed = chunk_pass.process(chunks, ctx)
        _validate_fixed_blocks_preserved(fixed_before, chunks)
        changed |= pass_changed
    return chunks, changed


def _run_chunklist_opt(
    passes: list[AbsChunkPass],
    chunk_list_passes: list[AbsChunkListPass],
    chunks: ChunkList,
    ctx: PipeLineContext,
) -> ChunkList:
    """Run AbsChunkPass + AbsChunkListPass as one iteration group to convergence.

    Both pass kinds operate at the ChunkList layer and are interleaved in the
    same fixed-point loop: a ChunkListPass may expose work for a ChunkPass and
    vice versa.  If convergence is not reached within ``max_opt_iterations``,
    the loop stops and logs which passes are still reporting changes — this is
    an oscillation signal, not a correctness problem (the result is still a
    valid program, just not maximally optimized).
    """
    max_iters = ctx.config.max_opt_iterations
    for _ in range(max_iters):
        chunks, changed1 = _run_passes(passes, chunks, ctx)
        chunks, changed2 = _run_passes(chunk_list_passes, chunks, ctx)
        if not changed1 and not changed2:
            return chunks
    # Did not converge — identify the still-changing passes for debugging.
    stuck: list[str] = []
    for p in (*passes, *chunk_list_passes):
        _, changed = p.process(deepcopy(chunks), deepcopy(ctx))
        if changed:
            stuck.append(type(p).__name__)
    logger.warning(
        "IR ChunkList optimization did not converge within %d iterations; "
        "passes still reporting changes: %s",
        max_iters,
        ", ".join(stuck) if stuck else "(none — spurious change signal)",
    )
    return chunks


def _optimize_tree(
    node: IRNode,
    tree_passes: list[AbsIRTreePass],
    ctx: PipeLineContext,
) -> IRNode:
    """Post-order apply AbsIRTreePass chain to an IR tree.

    The pipeline owns recursion: children are transformed to convergence
    *before* a pass sees their parent, so each ``transform`` only reasons
    about its own level.  A pass that returns a replacement node triggers a
    re-recurse into that node, so any structural nodes the replacement
    contains (e.g. an IRDispatch produced by unrolling) get visited too.

    Only AbsIRTreePass runs here.  ChunkPass / ChunkListPass run separately at
    the ChunkList layer, before parse and after unparse.
    """
    from .node import BlockNode, IRBranch, IRLoop

    # ── Leaf nodes: BasicBlockNode / IRDispatch have no IRNode children ───────
    if isinstance(node, BasicBlockNode):
        return node

    fixed_before = _collect_fixed_blocks(node)

    # ── Recurse into children first (post-order) ──────────────────────────────
    if isinstance(node, BlockNode):
        node.insts = [_optimize_tree(child, tree_passes, ctx) for child in node.insts]
    elif isinstance(node, IRLoop):
        node.body = _optimize_tree(node.body, tree_passes, ctx)
    elif isinstance(node, IRBranch):
        node.cases = [_optimize_tree(case, tree_passes, ctx) for case in node.cases]
    # IRDispatch is a leaf (case bodies are not stored inside it).

    # ── Apply the tree-pass chain to this node ────────────────────────────────
    # A pass returning a new node replaces the subtree; re-recurse into it so
    # the replacement's children and any new structural nodes are optimized.
    for tree_pass in tree_passes:
        replacement = tree_pass.transform(node, ctx)
        if replacement is not None:
            _validate_fixed_blocks_preserved(fixed_before, replacement)
            return _optimize_tree(replacement, tree_passes, ctx)
    _validate_fixed_blocks_preserved(fixed_before, node)
    return node


# Structural MetaInst types that mark IR-tree boundaries. They are consumed
# by IRParser.parse / unparse and must be stripped before the post-tree
# ChunkList optimization (DISABLE_OPT info already lives on
# BasicBlockNode.disable_opt, so no MetaInst needs to survive).
_STRUCTURAL_META_TYPES: frozenset[str] = frozenset(
    {
        "LOOP_START",
        "LOOP_BODY_START",
        "LOOP_BODY_END",
        "LOOP_END",
        "BRANCH_START",
        "BRANCH_CASE_START",
        "BRANCH_CASE_END",
        "BRANCH_END",
        "DISPATCH_START",
        "DISPATCH_END",
        "DISABLE_OPT_START",
        "DISABLE_OPT_END",
    }
)


def _strip_structural_meta(chunks: ChunkList) -> list[BasicBlockNode]:
    """Drop structural MetaInst, keeping only BasicBlockNode chunks.

    By the time this runs the chunk list has already been through IRLexer.lex
    (which records DISABLE_OPT spans onto BasicBlockNode.disable_opt) and
    IRParser.unparse, so every structural marker is redundant. Any leftover
    non-structural MetaInst is unexpected and raises.
    """
    result: list[BasicBlockNode] = []
    for chunk in chunks:
        if isinstance(chunk, BasicBlockNode):
            result.append(chunk)
        elif chunk.type in _STRUCTURAL_META_TYPES:
            continue
        else:
            # Internal consistency assertion: IRParser.unparse() must only emit
            # structural MetaInsts. Reaching here means unparse() has a bug.
            raise ValueError(
                f"_strip_structural_meta: unexpected non-structural MetaInst "
                f"{chunk.type!r} in chunk list"
            )
    return result


def _resolve_dmem_dispatch(chunks: list[BasicBlockNode], ctx: PipeLineContext) -> None:
    """Resolve every DmemAddr reference to a concrete dmem base offset.

    Runs after every clone-capable pass and after ChunkList optimization, so
    each DmemAddr is final. Walks all instructions, dedupes DmemAddr references
    by their entry-label tuple (so identical dispatch tables share one dmem
    run), assigns each unique table a contiguous dmem range starting at
    ``ctx.dmem_base_offset``, and rewrites the ``DmemAddr`` operand to an
    ``Immediate(base)``. The per-table entry-label lists are recorded in
    ``ctx.dmem_tables`` in allocation order for the caller to materialize.

    DmemAddr lives in ``RegWriteInst.op`` (an ``AluExpr`` whose ``rhs`` is the
    DmemAddr), produced by ``DmemDispatchPass``.
    """
    import dataclasses

    from .operands import AluExpr, DmemAddr, Immediate

    # table_labels tuple -> assigned dmem base offset
    assigned: dict[tuple[Label, ...], int] = {}
    next_offset = ctx.dmem_base_offset
    ref_count = 0  # total DmemAddr references rewritten (incl. dedup hits)

    def _resolve_inst(inst: BaseInst) -> BaseInst:
        nonlocal next_offset, ref_count
        if not isinstance(inst, RegWriteInst):
            return inst
        op = inst.op
        if not isinstance(op, AluExpr) or not isinstance(op.rhs, DmemAddr):
            return inst
        ref_count += 1
        key = op.rhs.table_labels
        base = assigned.get(key)
        if base is None:
            base = next_offset
            assigned[key] = base
            next_offset += len(key)
            ctx.dmem_tables.append(list(key))
            logger.debug(
                "dmem dispatch: allocated table #%d at dmem offset %d, "
                "%d entries: [%s]",
                len(ctx.dmem_tables) - 1,
                base,
                len(key),
                ", ".join(lbl.name for lbl in key),
            )
        else:
            logger.debug(
                "dmem dispatch: reused table at dmem offset %d for [%s]",
                base,
                ", ".join(lbl.name for lbl in key),
            )
        new_op = dataclasses.replace(op, rhs=Immediate(base))
        return dataclasses.replace(inst, op=new_op)

    for block in chunks:
        block.insts = [_resolve_inst(i) for i in block.insts]

    if ctx.dmem_tables:
        words = next_offset - ctx.dmem_base_offset
        logger.debug(
            "dmem dispatch: resolved %d reference(s) into %d unique table(s), "
            "dmem offsets [%d, %d) (%d words)",
            ref_count,
            len(ctx.dmem_tables),
            ctx.dmem_base_offset,
            next_offset,
            words,
        )


class IRPipeLine:
    """U-shape single-pass IR optimization pipeline.

    Flow::

        list[Instruction]
          → IRLexer.lex            → ChunkList
          → ChunkList optimization (ChunkPass + ChunkListPass, one iter group)
          → IRParser.parse         → IR tree
          → IR-tree optimization   (AbsIRTreePass, post-order, pipeline-driven)
          → IRParser.unparse       → ChunkList (with structural MetaInst)
          → strip structural MetaInst
          → ChunkList optimization (ChunkPass + ChunkListPass, one iter group)
          → IRLexer.flatten        → list[Instruction]

    IR optimization is optional: ``disable_all_opt`` returns the input
    untouched (the macro/module layer already emits runnable ASM).
    """

    def __init__(
        self,
        config: PipeLineConfig,
        chunk_passes: list[AbsChunkPass],
        chunk_list_passes: list[AbsChunkListPass],
        tree_passes: list[AbsIRTreePass] | None = None,
    ) -> None:
        self.config = config
        self.chunk_passes = chunk_passes
        self.chunk_list_passes = chunk_list_passes
        self.tree_passes: list[AbsIRTreePass] = tree_passes or []

    def __call__(
        self,
        insts: list[Instruction],
        dmem_base_offset: int = 0,
    ) -> tuple[list[Instruction], PipeLineContext]:
        """Optimize a flat instruction list.

        ``dmem_base_offset`` is the dmem index at which IR-generated dispatch
        tables may start (the caller passes the current dmem buffer length).
        After the call, ``ctx.dmem_tables`` lists the tables that were
        allocated, contiguous from that offset, for the caller to materialize.
        """
        from .hw_semantics import GENERAL_REGS
        from .operands import parse_register

        used_regs = set()
        for inst in insts:
            if isinstance(inst, BaseInst):
                used_regs.update(inst.reg_read)
                used_regs.update(inst.reg_write)
            elif isinstance(inst, MetaInst) and inst.info:
                for v in inst.info.values():
                    if isinstance(v, str):
                        reg = parse_register(v)
                        if reg is not None:
                            used_regs.update(reg.regs())

        available_regs = set(GENERAL_REGS) - used_regs

        ctx = PipeLineContext(
            config=self.config,
            pmem_budget=int(0.8 * self.config.pmem_capacity),
            available_regs=available_regs,
            dmem_base_offset=dmem_base_offset,
        )
        if self.config.disable_all_opt:
            return insts, ctx

        lexer = IRLexer()
        parser = IRParser(pmem_size=self.config.pmem_capacity)

        # --- lex → ChunkList ---
        chunks = lexer.lex(insts)

        # --- ChunkList optimization (pre-parse) ---
        chunks = _run_chunklist_opt(
            self.chunk_passes, self.chunk_list_passes, chunks, ctx
        )

        # --- parse → IR tree ---
        ir = parser.parse(chunks)
        from .node import _collect_subtree_names

        label_names, struct_names = _collect_subtree_names(ir)
        ctx.allocated_names = label_names | struct_names

        # --- IR-tree optimization (post-order, pipeline-driven recursion) ---
        ir = _optimize_tree(ir, self.tree_passes, ctx)

        # --- unparse → ChunkList, then strip structural MetaInst ---
        unparsed = parser.unparse(cast(BlockNode, ir))
        flat_chunks: ChunkList = list(_strip_structural_meta(unparsed))

        # --- ChunkList optimization (post-unparse) ---
        flat_chunks = _run_chunklist_opt(
            self.chunk_passes, self.chunk_list_passes, flat_chunks, ctx
        )

        # --- resolve dmem dispatch tables (after every clone-capable pass) ---
        bb_chunks = [c for c in flat_chunks if isinstance(c, BasicBlockNode)]
        _resolve_dmem_dispatch(bb_chunks, ctx)

        return lexer.flatten(flat_chunks), ctx


# ---------------------------------------------------------------------------
# Default pipeline factory
# ---------------------------------------------------------------------------


def make_default_pipeline(
    pmem_capacity: int, max_unroll_factor: int | None = None
) -> IRPipeLine:
    from .passes import (
        BlockMergePass,
        BranchEliminationPass,
        DeadLabelEliminationPass,
        DeadTestEliminationPass,
        DeadWriteEliminationPass,
        DmemDispatchPass,
        IncRegMergePass,
        SimplifyDispatchPass,
        TimedMergePass,
        UnpackIRBranchPass,
        UnreachableEliminationPass,
        UnrollLoopPass,
        ZeroDelayDCEPass,
    )

    config = deepcopy(DEFAULT_PIPELINE_CONFIG)
    config.pmem_capacity = pmem_capacity
    if max_unroll_factor is not None:
        config.max_unroll_factor = max_unroll_factor

    return IRPipeLine(
        config=config,
        chunk_passes=[
            IncRegMergePass(),
            TimedMergePass(),
            ZeroDelayDCEPass(),
            DeadTestEliminationPass(),
            DeadWriteEliminationPass(),
        ],
        chunk_list_passes=[
            UnreachableEliminationPass(),
            DeadLabelEliminationPass(),
            BranchEliminationPass(),
            BlockMergePass(),
        ],
        tree_passes=[
            UnrollLoopPass(),
            UnpackIRBranchPass(),
            SimplifyDispatchPass(),
            DmemDispatchPass(),
        ],
    )
