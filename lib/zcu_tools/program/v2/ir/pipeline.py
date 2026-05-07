from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field

from .node import BasicBlockNode, RootNode
from .traversal import walk_basic_blocks


@dataclass
class PipeLineConfig:
    disable_all_opt: bool = False
    enable_unroll_loop: bool = True
    enable_dead_write: bool = True
    enable_dead_label: bool = True
    enable_zero_delay_dce: bool = True
    enable_timed_instruction_merge: bool = True
    pmem_capacity: int | None = None
    pmem_budget: int | None = None

    # Hard cap on the unroll factor k. For register-driven loops k is also
    # rounded down to the nearest power of 2 (Phase 8D).
    max_unroll_factor: int = 8

    # Maximum number of REG_WR words allowed in the dispatch shift-add
    # sequence. If the body_words multiply cannot fit, register-driven
    # unroll falls back to no-unroll.
    max_dispatch_words: int = 8

    # Unified cycle cost model
    cost_default: int = 1
    cost_wmem: int = 4
    cost_dmem: int = 4
    cost_jump_flush: int = 1000


# Global default configuration for quick debugging and toggling optimization options.
DEFAULT_PIPELINE_CONFIG = PipeLineConfig()


@dataclass
class PipeLineContext:
    config: PipeLineConfig = field(default_factory=PipeLineConfig)
    pmem_size: int | None = None


# ---------------------------------------------------------------------------
# Pass interfaces
# ---------------------------------------------------------------------------


class AbsIRPass(ABC):
    """Structural IR pass: transforms a whole RootNode tree."""

    @abstractmethod
    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode: ...


AbsPipeLinePass = AbsIRPass  # backwards-compatible alias


class AbsLinearPass(ABC):
    """Straight-line optimization pass that operates on a single BasicBlockNode.

    Implementations receive the whole block and must modify ``block.insts``
    in-place.  The ``block.labels`` and ``block.branch`` fields must NOT be
    touched — those belong to the structural level.

    When ``block.fix_inst_num`` is True the total length of ``block.insts``
    must be preserved.  Passes that normally delete instructions must replace
    removed instructions with ``NopInst`` instead.  Passes that merge N
    instructions into 1 must pad with N-1 ``NopInst``s.
    """

    @abstractmethod
    def process_block(self, block: BasicBlockNode) -> None: ...


# ---------------------------------------------------------------------------
# IRPipeLine: three-stage structural pipeline
# ---------------------------------------------------------------------------


def _run_linear_passes(passes: list[AbsLinearPass], ir: RootNode) -> None:
    for block in walk_basic_blocks(ir):
        for lp in passes:
            lp.process_block(block)


class IRPipeLine:
    """Three-stage IR optimization pipeline.

    Stage 1 — Pre-LIR  : ``linear_passes`` applied to every BasicBlockNode
                          before any structural pass.
    Stage 2 — HIR       : sequence of ``AbsIRPass`` instances that may
                          restructure the tree (e.g. loop unrolling).
    Stage 3 — Post-LIR : same ``linear_passes`` re-applied after all
                          structural passes.

    Each AbsLinearPass handles fix_inst_num internally, so the same pass list
    is safe to run both before and after structural changes.
    """

    def __init__(
        self,
        config: PipeLineConfig,
        linear_passes: list[AbsLinearPass],
        ir_passes: list[AbsIRPass],
    ) -> None:
        self.config = config
        self.linear_passes = linear_passes
        self.ir_passes = ir_passes

    def __call__(self, ir: RootNode) -> tuple[RootNode, PipeLineContext]:
        ctx = PipeLineContext(config=self.config, pmem_size=self.config.pmem_capacity)
        if self.config.disable_all_opt:
            return ir, ctx

        # Stage 1: Pre-LIR
        _run_linear_passes(self.linear_passes, ir)

        # Stage 2: HIR structural passes
        for _pass in self.ir_passes:
            ir = _pass.process(ir, ctx)

        # Stage 3: Post-LIR
        _run_linear_passes(self.linear_passes, ir)

        return ir, ctx


# ---------------------------------------------------------------------------
# Default pipeline factory
# ---------------------------------------------------------------------------


def make_default_pipeline(pmem_capacity: int) -> IRPipeLine:
    from .passes import (
        BlockMergePass,
        BranchEliminationPass,
        DeadLabelEliminationPass,
        DeadWriteEliminationLinear,
        TimedMergeLinear,
        UnrollSmallLoopPass,
        ZeroDelayDCELinear,
    )

    config = deepcopy(DEFAULT_PIPELINE_CONFIG)
    if config.pmem_capacity is None:
        config.pmem_capacity = pmem_capacity
    if config.pmem_budget is None:
        config.pmem_budget = int(0.8 * pmem_capacity)

    return IRPipeLine(
        config=config,
        linear_passes=[
            ZeroDelayDCELinear(),
            TimedMergeLinear(),
            DeadWriteEliminationLinear(),
        ],
        ir_passes=[
            UnrollSmallLoopPass(),
            DeadLabelEliminationPass(),
            BranchEliminationPass(),
            BlockMergePass(),
        ],
    )
