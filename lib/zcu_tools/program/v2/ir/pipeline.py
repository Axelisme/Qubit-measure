from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Union

from .factory import IRLexer, IRParser
from .instructions import Instruction, MetaInst
from .node import BasicBlockNode, BlockNode

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
    """Optimization pass on chunk layer: list[BasicBlockNode | MetaInst]."""

    @abstractmethod
    def process(
        self, chunks: ChunkList, ctx: PipeLineContext
    ) -> tuple[ChunkList, bool]: ...


class AbsIRPass(ABC):
    """Structural IR pass: transforms a whole BlockNode tree."""

    @abstractmethod
    def process(
        self, ir: BlockNode, ctx: PipeLineContext
    ) -> tuple[BlockNode, bool]: ...


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


def _run_ir_passes(
    passes: list[AbsIRPass], ir: BlockNode, ctx: PipeLineContext
) -> tuple[BlockNode, bool]:
    changed = False
    for ir_pass in passes:
        ir, pass_changed = ir_pass.process(ir, ctx)
        changed |= pass_changed
    return ir, changed


def _strip_structural_meta(chunks: ChunkList) -> ChunkList:
    """Remove all MetaInst except DISABLE_OPT_START/END markers.

    Called after the U-shaped IR optimization stage to flatten structural
    boundaries (LOOP_*, BRANCH_*, DISPATCH_*) so post-IR chunk passes can
    optimize across former IRNode boundaries.
    """
    keep = {"DISABLE_OPT_START", "DISABLE_OPT_END"}
    return [
        item for item in chunks if not isinstance(item, MetaInst) or item.type in keep
    ]


class IRPipeLine:
    """Two-stage optimization pipeline.

    Stage 1 — U-shaped IR optimization (repeated up to max_opt_iterations):
      Chunk passes → IR tree passes → Chunk passes
      Repeats until convergence.

    Stage 2 — Post-IR chunk optimization:
      Strip structural MetaInst (LOOP_*/BRANCH_*/DISPATCH_*) to expose
      cross-boundary optimization opportunities, then run chunk passes again
      (up to max_opt_iterations times) until convergence.
    """

    def __init__(
        self,
        config: PipeLineConfig,
        chunk_passes: list[AbsChunkPass],
        ir_passes: list[AbsIRPass],
    ) -> None:
        self.config = config
        self.chunk_passes = chunk_passes
        self.ir_passes = ir_passes

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

        # --- Stage 1: U-shaped IR optimization ---
        chunks = lexer.lex(insts)
        max_iters = max(1, self.config.max_opt_iterations)
        for _ in range(max_iters):
            iter_changed = False

            chunks, changed = _run_chunk_passes(self.chunk_passes, chunks, ctx)
            iter_changed |= changed

            ir = parser.parse(chunks)
            ir, changed = _run_ir_passes(self.ir_passes, ir, ctx)
            iter_changed |= changed

            chunks = parser.unparse(ir)
            chunks, changed = _run_chunk_passes(self.chunk_passes, chunks, ctx)
            iter_changed |= changed

            if not iter_changed:
                break

        # --- Stage 2: post-IR chunk optimization ---
        chunks = _strip_structural_meta(chunks)
        for _ in range(max_iters):
            chunks, changed = _run_chunk_passes(self.chunk_passes, chunks, ctx)
            if not changed:
                break

        curr_insts = lexer.flatten(chunks)
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
            UnreachableEliminationPass(),
            ZeroDelayDCEPass(),
            LoopConditionMergePass(),
            DeadTestEliminationPass(),
            DeadWriteEliminationPass(),
            DeadLabelEliminationPass(),
            BranchEliminationPass(),
            BlockMergePass(),
        ],
        ir_passes=[
            UnrollLoopPass(),
            SimplifyDispatchPass(),
        ],
    )
