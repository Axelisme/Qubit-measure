from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Union

from .factory import IRLexer, IRParser
from .instructions import Instruction, MetaInst
from .node import BasicBlockNode, IRNode

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
    """Post-order IR tree pass: runs after child nodes are lowered, before self is lowered.

    ``transform`` receives:
      - ``node``: the IRNode being considered (IRLoop / IRBranch / IRDispatch)
      - ``child_chunks``: each child's already-lowered and ChunkPass-optimized flat chunks
        - IRLoop:     child_chunks[0] = body chunks
        - IRBranch:   child_chunks[i] = cases[i] chunks
        - IRDispatch: child_chunks = [] (leaf node)
      - ``ctx``: pipeline context

    Return values:
      - ``list[BasicBlockNode]``: skip lower entirely; use this as the final result
      - ``IRNode`` (same identity as ``node``): pass did nothing; proceed to AbsNodeLower
      - ``IRNode`` (different object): replace subtree; pipeline will re-recurse _lower_node
    """

    @abstractmethod
    def transform(
        self,
        node: IRNode,
        child_chunks: list[list[BasicBlockNode]],
        ctx: PipeLineContext,
    ) -> IRNode | list[BasicBlockNode]: ...


class AbsNodeLower(ABC):
    """Chain-of-responsibility per-node lower.

    Tries to lower a single IRNode to flat BasicBlockNodes.  Return ``None`` to
    pass to the next lower in the chain; return a list to claim the node.
    The final fallback is ``IRParser._lower_*``.
    """

    @abstractmethod
    def lower(
        self,
        node: IRNode,
        child_chunks: list[list[BasicBlockNode]],
        ctx: PipeLineContext,
    ) -> list[BasicBlockNode] | None: ...


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


def _run_chunk_passes_to_convergence(
    passes: list[AbsChunkPass], chunks: list[BasicBlockNode], ctx: PipeLineContext
) -> list[BasicBlockNode]:
    """Run AbsChunkPass to convergence on a flat BasicBlockNode list."""
    max_iters = ctx.config.max_opt_iterations
    full_chunks: ChunkList = list(chunks)
    for _ in range(max_iters):
        full_chunks, changed = _run_chunk_passes(passes, full_chunks, ctx)
        if not changed:
            break
    return [c for c in full_chunks if isinstance(c, BasicBlockNode)]


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
    node_lowers: list[AbsNodeLower],
    ctx: PipeLineContext,
) -> list[BasicBlockNode]:
    """Post-order lower a single IRNode to flat BasicBlockNodes.

    For BasicBlockNode / BlockNode: straightforward recursion.
    For IRLoop / IRBranch / IRDispatch:
      1. Recursively lower children.
      2. Run AbsChunkPass on each child's chunks.
      3. Run AbsIRTreePass chain (may short-circuit lowering).
      4. Run AbsNodeLower chain (fallback to IRParser._lower_*).
      5. Run AbsChunkPass on the resulting flat chunks.

    AbsChunkListPass is intentionally excluded here: those passes require the
    globally-flat program to reason correctly about label references and block
    adjacency.  They are applied only at the pipeline level after all subtrees
    are fully lowered.
    """
    from .factory import IRParser
    from .node import BlockNode, IRBranch, IRDispatch, IRLoop

    if isinstance(node, BasicBlockNode):
        return [node]

    if isinstance(node, BlockNode):
        result: list[BasicBlockNode] = []
        for child in node.insts:
            result.extend(
                _lower_node(child, chunk_passes, tree_passes, node_lowers, ctx)
            )
        return result

    # IRLoop / IRBranch / IRDispatch ─────────────────────────────────────────
    # Step 1: recursively lower children.
    child_chunks: list[list[BasicBlockNode]] = []
    for child in node.children():
        child_chunks.append(
            _lower_node(child, chunk_passes, tree_passes, node_lowers, ctx)
        )

    # Step 2: run AbsChunkPass on each child's chunks.
    child_chunks = [
        _run_chunk_passes_to_convergence(chunk_passes, cc, ctx) for cc in child_chunks
    ]

    # Step 3: AbsIRTreePass chain.
    current: IRNode = node
    flat: list[BasicBlockNode] | None = None
    for tree_pass in tree_passes:
        result_tp = tree_pass.transform(current, child_chunks, ctx)
        if isinstance(result_tp, list):
            flat = result_tp
            break
        if result_tp is not current:
            # New subtree returned — re-recurse fully.
            return _lower_node(result_tp, chunk_passes, tree_passes, node_lowers, ctx)
        # result_tp is current: pass did nothing, try next.

    if flat is None:
        # Step 4: AbsNodeLower chain.
        for node_lower in node_lowers:
            result_nl = node_lower.lower(current, child_chunks, ctx)
            if result_nl is not None:
                flat = result_nl
                break

    if flat is None:
        # Fallback: IRParser default lower.
        parser = IRParser(pmem_size=ctx.config.pmem_capacity)
        if isinstance(current, IRLoop):
            merged_body = [b for cc in child_chunks for b in cc]
            flat = parser._lower_loop(current, merged_body)
        elif isinstance(current, IRBranch):
            flat = parser._lower_branch(current, child_chunks)
        elif isinstance(current, IRDispatch):
            flat = parser._lower_dispatch(current)
        else:
            raise TypeError(
                f"_lower_node: no lower for node type {type(current).__name__}"
            )

    # Step 5: run AbsChunkPass on the flat result.
    flat = _run_chunk_passes_to_convergence(chunk_passes, flat, ctx)
    return flat


class IRPipeLine:
    """Post-order IR lower pipeline.

    Flow:
      1. Lex flat instructions → ChunkList.
      2. Pre-lower AbsChunkPass + AbsChunkListPass (single round, no iteration).
      3. IRParser.parse() → IR tree (once).
      4. _lower_node(root): post-order lower with per-subtree AbsChunkPass and
         AbsIRTreePass / AbsNodeLower hooks.
         AbsChunkListPass is NOT run inside _lower_node — those passes require
         the globally-flat program and are deferred to Step 5.
      5. Post-lower AbsChunkPass + AbsChunkListPass to convergence.
    """

    def __init__(
        self,
        config: PipeLineConfig,
        chunk_passes: list[AbsChunkPass],
        chunk_list_passes: list[AbsChunkListPass],
        tree_passes: list[AbsIRTreePass] | None = None,
        node_lowers: list[AbsNodeLower] | None = None,
    ) -> None:
        self.config = config
        self.chunk_passes = chunk_passes
        self.chunk_list_passes = chunk_list_passes
        self.tree_passes: list[AbsIRTreePass] = tree_passes or []
        self.node_lowers: list[AbsNodeLower] = node_lowers or []

    def __call__(
        self, insts: list[Instruction]
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

        # --- Step 3: post-order lower with per-layer AbsChunkPasses ---
        flat_chunks = _lower_node(
            ir, self.chunk_passes, self.tree_passes, self.node_lowers, ctx
        )

        # --- Step 4: post-lower AbsChunkPass + AbsChunkListPass to convergence ---
        final_chunks = _run_all_passes_to_convergence(
            self.chunk_passes, self.chunk_list_passes, flat_chunks, ctx
        )

        curr_insts = lexer.flatten(list(final_chunks))
        return curr_insts, ctx


# ---------------------------------------------------------------------------
# Default pipeline factory
# ---------------------------------------------------------------------------


def make_default_pipeline(pmem_capacity: int) -> IRPipeLine:
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
