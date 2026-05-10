from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Union

from .factory import IRLexer, IRParser
from .instructions import Instruction, MetaInst
from .node import BasicBlockNode, RootNode

ChunkList = list[Union[BasicBlockNode, MetaInst]]


@dataclass
class PipeLineConfig:
    disable_all_opt: bool = False
    pmem_capacity: int = 4096

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


class AbsFlatPass(ABC):
    """Optimization pass on flat instruction list."""

    @abstractmethod
    def process(
        self, insts: list[Instruction], ctx: PipeLineContext
    ) -> tuple[list[Instruction], bool]: ...


class AbsChunkPass(ABC):
    """Optimization pass on chunk layer: list[BasicBlockNode | MetaInst]."""

    @abstractmethod
    def process(
        self, chunks: ChunkList, ctx: PipeLineContext
    ) -> tuple[ChunkList, bool]: ...


class AbsIRPass(ABC):
    """Structural IR pass: transforms a whole RootNode tree."""

    @abstractmethod
    def process(self, ir: RootNode, ctx: PipeLineContext) -> tuple[RootNode, bool]: ...


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _validate_fixed_block_words(
    block: BasicBlockNode, *, before_addr_size: int | None
) -> None:
    block.__post_init__()
    if before_addr_size is not None and block.addr_size != before_addr_size:
        raise ValueError(
            "AbsChunkPass violated fix_addr_size invariant: "
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
            if isinstance(chunk, BasicBlockNode) and chunk.fix_addr_size
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
    passes: list[AbsIRPass], ir: RootNode, ctx: PipeLineContext
) -> tuple[RootNode, bool]:
    changed = False
    for ir_pass in passes:
        ir, pass_changed = ir_pass.process(ir, ctx)
        changed |= pass_changed
    return ir, changed


def _run_flat_passes(
    passes: list[AbsFlatPass], insts: list[Instruction], ctx: PipeLineContext
) -> tuple[list[Instruction], bool]:
    changed = False
    for flat_pass in passes:
        insts, pass_changed = flat_pass.process(insts, ctx)
        changed |= pass_changed
    return insts, changed


class IRPipeLine:
    """U-shaped multi-layer optimization pipeline.

    Per iteration:
      Flat-up -> Chunk-up -> Tree -> Chunk-down -> Flat-down

    If any pass reports changed=True, run another full U iteration until
    convergence or `config.max_opt_iterations`.
    """

    def __init__(
        self,
        config: PipeLineConfig,
        flat_passes: list[AbsFlatPass],
        chunk_passes: list[AbsChunkPass],
        ir_passes: list[AbsIRPass],
    ) -> None:
        self.config = config
        self.flat_passes = flat_passes
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
        curr_insts = insts

        max_iters = max(1, self.config.max_opt_iterations)
        for _ in range(max_iters):
            iter_changed = False

            # Flat Optimization
            curr_insts, changed = _run_flat_passes(self.flat_passes, curr_insts, ctx)
            iter_changed |= changed

            # Chunk Optimization
            chunks = lexer.lex(curr_insts)
            chunks, changed = _run_chunk_passes(self.chunk_passes, chunks, ctx)
            iter_changed |= changed

            # IR Tree Optimization
            ir = parser.parse(chunks)
            ir, changed = _run_ir_passes(self.ir_passes, ir, ctx)
            iter_changed |= changed

            # Chunk Optimization
            chunks = parser.unparse(ir)
            chunks, changed = _run_chunk_passes(self.chunk_passes, chunks, ctx)
            iter_changed |= changed

            # Flat Optimization
            curr_insts = lexer.flatten(chunks)
            curr_insts, changed = _run_flat_passes(self.flat_passes, curr_insts, ctx)
            iter_changed |= changed

            if not iter_changed:
                break

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
        TimedMergePass,
        UnreachableEliminationPass,
        UnrollLoopPass,
        ZeroDelayDCEPass,
    )

    config = deepcopy(DEFAULT_PIPELINE_CONFIG)
    config.pmem_capacity = pmem_capacity

    return IRPipeLine(
        config=config,
        flat_passes=[UnreachableEliminationPass()],
        chunk_passes=[
            IncRegMergePass(),
            TimedMergePass(),
            ZeroDelayDCEPass(),
            LoopConditionMergePass(),
            DeadTestEliminationPass(),
            DeadWriteEliminationPass(),
        ],
        ir_passes=[
            UnrollLoopPass(),
            BranchEliminationPass(),
            BlockMergePass(),
            DeadLabelEliminationPass(),
        ],
    )
