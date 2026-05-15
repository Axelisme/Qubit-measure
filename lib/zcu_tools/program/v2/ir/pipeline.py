from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Union

from .factory import IRLexer, IRParser
from .instructions import Instruction, MetaInst
from .node import BasicBlockNode, IRNode

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

ChunkList = list[Union[BasicBlockNode, MetaInst]]


@dataclass
class PipeLineConfig:
    pmem_capacity: int = 4096
    disable_all_opt: bool = False

    max_opt_iterations: int = 8

    # Hard cap on the unroll factor k. For register-driven loops k is also
    # rounded down to the nearest power of 2 (Phase 8D).
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
      - ``node``: the IRNode being considered
      - ``ctx``: pipeline context

    Return values:
      - ``IRNode`` (same identity as ``node``): pass did nothing; try next pass
      - ``IRNode`` (different object): replace subtree; pipeline will re-recurse _lower_node
    """

    @abstractmethod
    def transform(
        self,
        node: IRNode,
        ctx: PipeLineContext,
    ) -> IRNode: ...


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _validate_fixed_block_words(
    block: BasicBlockNode, *, before_addr_size: int | None
) -> None:
    block.__post_init__()
    if before_addr_size is not None and block.addr_size != before_addr_size:
        raise ValueError(
            "AbsChunkPass violated disable_opt invariant: "
            "block program-memory word count changed."
        )


def _run_chunk_passes(
    passes: list[AbsChunkPass], chunks: ChunkList, ctx: PipeLineContext
) -> tuple[ChunkList, bool]:
    changed = False
    for chunk_pass in passes:
        fixed_before = {
            id(chunk): chunk.addr_size
            for chunk in chunks
            if isinstance(chunk, BasicBlockNode) and chunk.disable_opt
        }
        chunks, pass_changed = chunk_pass.process(chunks, ctx)
        for chunk in chunks:
            if not isinstance(chunk, BasicBlockNode):
                continue
            _validate_fixed_block_words(
                chunk,
                before_addr_size=fixed_before.get(id(chunk)),
            )
        changed |= pass_changed
    return chunks, changed


def _run_chunk_list_passes(
    passes: list[AbsChunkListPass], chunks: ChunkList, ctx: PipeLineContext
) -> tuple[ChunkList, bool]:
    changed = False
    for chunk_pass in passes:
        fixed_before = {
            id(chunk): chunk.addr_size
            for chunk in chunks
            if isinstance(chunk, BasicBlockNode) and chunk.disable_opt
        }
        chunks, pass_changed = chunk_pass.process(chunks, ctx)
        for chunk in chunks:
            if not isinstance(chunk, BasicBlockNode):
                continue
            _validate_fixed_block_words(
                chunk,
                before_addr_size=fixed_before.get(id(chunk)),
            )
        changed |= pass_changed
    return chunks, changed


def _run_chunk_passes_on_block(
    passes: list[AbsChunkPass], block: BasicBlockNode, ctx: PipeLineContext
) -> BasicBlockNode:
    """Run AbsChunkPass to convergence on a single BasicBlockNode."""
    max_iters = ctx.config.max_opt_iterations
    chunks: ChunkList = [block]
    for _ in range(max_iters):
        chunks, changed = _run_chunk_passes(passes, chunks, ctx)
        if not changed:
            break
    # There is always exactly one BasicBlockNode in the list.
    return next(c for c in chunks if isinstance(c, BasicBlockNode))


def _run_all_passes_to_convergence(
    passes: list[AbsChunkPass],
    chunk_list_passes: list[AbsChunkListPass],
    chunks: list[BasicBlockNode],
    ctx: PipeLineContext,
) -> list[BasicBlockNode]:
    """Run AbsChunkPass + AbsChunkListPass to convergence on a flat BasicBlockNode list.

    AbsChunkListPass requires the full flat chunk list to be correct (dead-label
    detection, block adjacency for branch elimination, etc.).  Only call this
    function when ``chunks`` represents the complete, globally-flat program — not
    a subtree fragment produced during post-order lowering.
    """
    max_iters = ctx.config.max_opt_iterations
    full_chunks: ChunkList = list(chunks)
    for _ in range(max_iters):
        full_chunks, changed1 = _run_chunk_passes(passes, full_chunks, ctx)
        full_chunks, changed2 = _run_chunk_list_passes(
            chunk_list_passes, full_chunks, ctx
        )
        if not changed1 and not changed2:
            break
    return [c for c in full_chunks if isinstance(c, BasicBlockNode)]


def _lower_node(
    node: IRNode,
    chunk_passes: list[AbsChunkPass],
    tree_passes: list[AbsIRTreePass],
    ctx: PipeLineContext,
) -> IRNode:
    """Post-order lower a single IRNode.

    Recursively transforms the IR tree until only BlockNode and BasicBlockNode
    remain.  The caller is responsible for flattening the result into a
    list[BasicBlockNode].

    Rules:
      - BasicBlockNode (leaf): run ChunkPass to convergence, return as-is.
      - BlockNode (container): recursively lower each child; inline-flatten any
        BlockNode children into the parent's inst list.
      - IRLoop / IRBranch / IRDispatch: run AbsIRTreePass chain; if any pass
        returns a new node, re-recurse on it.  After all passes, if the node is
        still an IR control node, fall back to IRParser which returns a BlockNode;
        re-recurse on that BlockNode.

    AbsChunkListPass is intentionally excluded: those passes require the
    globally-flat program and are applied only after the full tree is lowered.
    """
    from .node import BlockNode, IRBranch, IRDispatch, IRLoop

    # ── BasicBlockNode ────────────────────────────────────────────────────────
    if isinstance(node, BasicBlockNode):
        return _run_chunk_passes_on_block(chunk_passes, node, ctx)

    # ── BlockNode ─────────────────────────────────────────────────────────────
    if isinstance(node, BlockNode):
        new_insts: list[IRNode] = []
        for child in node.insts:
            result = _lower_node(child, chunk_passes, tree_passes, ctx)
            if isinstance(result, BlockNode):
                new_insts.extend(result.insts)
            else:
                new_insts.append(result)
        node.insts = new_insts
        return node

    # ── IRLoop / IRBranch / IRDispatch ────────────────────────────────────────
    # Run AbsIRTreePass chain; any pass that returns a new node triggers a
    # full re-recurse so children get lowered in the new subtree.
    current: IRNode = node
    for tree_pass in tree_passes:
        result_tp = tree_pass.transform(current, ctx)
        if result_tp is not current:
            return _lower_node(result_tp, chunk_passes, tree_passes, ctx)

    # Fallback: lower children first, then call IRParser default lower.
    # The result is a flat list[BasicBlockNode]; wrap in BlockNode and re-recurse
    # so that the resulting BasicBlockNodes get ChunkPass treatment.
    from .factory import IRParser

    parser = IRParser(pmem_size=ctx.config.pmem_capacity)
    if isinstance(current, IRLoop):
        body_node = _lower_node(current.body, chunk_passes, tree_passes, ctx)
        body_chunks: list[BasicBlockNode] = []
        if isinstance(body_node, BasicBlockNode):
            body_chunks = [body_node]
        elif isinstance(body_node, BlockNode):
            body_chunks = [c for c in body_node.insts if isinstance(c, BasicBlockNode)]
        flat = parser._lower_loop(current, body_chunks)
    elif isinstance(current, IRBranch):
        case_chunks_list: list[list[BasicBlockNode]] = []
        for case in current.cases:
            case_node = _lower_node(case, chunk_passes, tree_passes, ctx)
            if isinstance(case_node, BasicBlockNode):
                case_chunks_list.append([case_node])
            elif isinstance(case_node, BlockNode):
                case_chunks_list.append(
                    [c for c in case_node.insts if isinstance(c, BasicBlockNode)]
                )
            else:
                case_chunks_list.append([])
        flat = parser._lower_branch(current, case_chunks_list)
    elif isinstance(current, IRDispatch):
        flat = parser._lower_dispatch(current)
    else:
        raise TypeError(f"_lower_node: no lower for node type {type(current).__name__}")

    flat_nodes: list[IRNode] = list(flat)
    return _lower_node(BlockNode(insts=flat_nodes), chunk_passes, tree_passes, ctx)


class IRPipeLine:
    """Post-order IR lower pipeline.

    Flow:
      1. Lex flat instructions → ChunkList.
      2. Pre-lower AbsChunkPass + AbsChunkListPass (single round, no iteration).
      3. IRParser.parse() → IR tree (once).
      4. _lower_node(root): post-order lower — BasicBlockNode gets ChunkPass,
         IRLoop/IRBranch/IRDispatch gets IRTreePass chain then IRParser fallback,
         BlockNode children are inline-flattened.  Recurse until only BlockNode
         and BasicBlockNode remain.
         AbsChunkListPass is NOT run inside _lower_node.
      5. Flatten result tree → list[BasicBlockNode].
      6. Post-lower AbsChunkPass + AbsChunkListPass to convergence.
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
    ) -> tuple[list[Instruction], PipeLineContext]:
        ctx = PipeLineContext(
            config=self.config,
            pmem_budget=int(0.8 * self.config.pmem_capacity),
        )
        if self.config.disable_all_opt:
            return insts, ctx

        lexer = IRLexer()
        parser = IRParser(pmem_size=self.config.pmem_capacity)

        # --- Step 1: pre-lower ChunkPass (single round, no iteration) ---
        chunks = lexer.lex(insts)
        chunks, _ = _run_chunk_passes(self.chunk_passes, chunks, ctx)
        chunks, _ = _run_chunk_list_passes(self.chunk_list_passes, chunks, ctx)

        # --- Step 2: parse once → IR tree ---
        ir = parser.parse(chunks)

        # --- Step 3: post-order lower ---
        lowered = _lower_node(ir, self.chunk_passes, self.tree_passes, ctx)

        # --- Step 4: flatten result tree → list[BasicBlockNode] ---
        from .node import BlockNode

        def _flatten(node: IRNode) -> list[BasicBlockNode]:
            if isinstance(node, BasicBlockNode):
                return [node]
            if isinstance(node, BlockNode):
                result: list[BasicBlockNode] = []
                for child in node.insts:
                    result.extend(_flatten(child))
                return result
            raise TypeError(
                f"IRPipeLine: unexpected node type after lowering: {type(node).__name__}"
            )

        flat_chunks = _flatten(lowered)

        # --- Step 5: post-lower AbsChunkPass + AbsChunkListPass to convergence ---
        final_chunks = _run_all_passes_to_convergence(
            self.chunk_passes, self.chunk_list_passes, flat_chunks, ctx
        )

        curr_insts = lexer.flatten(list(final_chunks))
        return curr_insts, ctx


# ---------------------------------------------------------------------------
# Default pipeline factory
# ---------------------------------------------------------------------------


def make_default_pipeline(
    pmem_capacity: int, max_unroll_factor: Optional[int] = None
) -> IRPipeLine:
    from .passes import (
        BlockMergePass,
        BranchEliminationPass,
        DeadLabelEliminationPass,
        DeadTestEliminationPass,
        DeadWriteEliminationPass,
        IncRegMergePass,
        LoopConditionMergePass,
        SimplifyDispatchPass,
        TimedMergePass,
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
            LoopConditionMergePass(),
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
            SimplifyDispatchPass(),
        ],
    )
