from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field

from .node import BasicBlockNode, IRNode, RootNode
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


# Keep AbsPipeLinePass as an alias so existing code that imports it still works.
AbsPipeLinePass = AbsIRPass


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
# LinearPipeline: applies a sequence of AbsLinearPass to every BasicBlockNode
# ---------------------------------------------------------------------------

class LinearPipeline:
    """A pipeline of AbsLinearPass instances applied to every BasicBlockNode.

    Iterates every BasicBlockNode reachable from an IRNode and calls each
    pass's ``process_block`` in order.  Each pass is responsible for
    respecting ``fix_inst_num`` on the block it receives.
    """

    def __init__(self, *passes: AbsLinearPass) -> None:
        self.passes: list[AbsLinearPass] = list(passes)

    def process_block(self, block: BasicBlockNode) -> None:
        for lp in self.passes:
            lp.process_block(block)

    def process(self, ir: IRNode) -> None:
        for block in walk_basic_blocks(ir):
            self.process_block(block)


# ---------------------------------------------------------------------------
# IRPipeLine: three-stage structural pipeline
# ---------------------------------------------------------------------------

class IRPipeLine:
    """Three-stage IR optimization pipeline.

    Stage 1 — Pre-LIR  : ``pre_linear`` LinearPipeline applied to every
                          BasicBlockNode before any structural pass.
    Stage 2 — HIR       : sequence of ``AbsIRPass`` instances that may
                          restructure the tree (e.g. loop unrolling).
    Stage 3 — Post-LIR : ``post_linear`` LinearPipeline applied to every
                          BasicBlockNode after all structural passes.

    Each structural IR pass in Stage 2 may optionally accept a
    ``LinearPipeline`` to re-run after per-pass structural changes (e.g.
    BlockMergePass re-runs the post-linear pipeline after merging blocks).
    """

    def __init__(
        self,
        config: PipeLineConfig,
        pre_linear: LinearPipeline,
        ir_passes: list[AbsIRPass],
        post_linear: LinearPipeline,
    ) -> None:
        self.config = config
        self.pre_linear = pre_linear
        self.ir_passes = ir_passes
        self.post_linear = post_linear

    def __call__(self, ir: RootNode) -> tuple[RootNode, PipeLineContext]:
        ctx = PipeLineContext(config=self.config, pmem_size=self.config.pmem_capacity)
        if self.config.disable_all_opt:
            return ir, ctx

        # Stage 1: Pre-LIR
        self.pre_linear.process(ir)

        # Stage 2: HIR structural passes
        for _pass in self.ir_passes:
            ir = _pass.process(ir, ctx)

        # Stage 3: Post-LIR
        self.post_linear.process(ir)

        return ir, ctx


# ---------------------------------------------------------------------------
# Legacy adapter: wraps a LinearPipeline as an AbsIRPass (for backwards compat
# and for use inside existing AbsIRPass implementations that need to re-run
# linear passes, e.g. BlockMergePass).
# ---------------------------------------------------------------------------

class LinearPipelineAdapter(AbsIRPass):
    """Adapts a LinearPipeline into an AbsIRPass for use in legacy contexts."""

    def __init__(self, linear_pipeline: LinearPipeline) -> None:
        self.linear_pipeline = linear_pipeline

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:  # noqa: ARG002
        self.linear_pipeline.process(ir)
        return ir


# ---------------------------------------------------------------------------
# Legacy shim: LinearPassAdapter kept for callers that import it from here
# ---------------------------------------------------------------------------

class LinearPassAdapter(AbsIRPass):
    """DEPRECATED: use LinearPipeline instead.

    Wraps one or more AbsLinearPass instances into an AbsIRPass.
    Kept for import compatibility; will be removed in a future refactor.
    """

    def __init__(self, *passes: AbsLinearPass) -> None:
        self._pipeline = LinearPipeline(*passes)

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:  # noqa: ARG002
        self._pipeline.process(ir)
        return ir


# ---------------------------------------------------------------------------
# PipeLine: legacy flat pipeline (kept for import compatibility)
# ---------------------------------------------------------------------------

class PipeLine:
    """DEPRECATED: use IRPipeLine instead.

    Flat list of AbsIRPass instances; kept so existing code that constructs
    a PipeLine directly still works.
    """

    def __init__(self, config: PipeLineConfig, passes: list[AbsIRPass]):
        self.config = config
        self.passes = passes

    def __call__(self, ir: RootNode) -> tuple[RootNode, PipeLineContext]:
        ctx = PipeLineContext(config=self.config, pmem_size=self.config.pmem_capacity)
        if self.config.disable_all_opt:
            return ir, ctx
        for _pass in self.passes:
            ir = _pass.process(ir, ctx)
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
        DeadWriteEliminationLegacyPass,
        TimedMergeLinear,
        TimedMergeLegacyPass,
        UnrollSmallLoopPass,
        ZeroDelayDCELinear,
        ZeroDelayDCELegacyPass,
    )

    config = deepcopy(DEFAULT_PIPELINE_CONFIG)
    if config.pmem_capacity is None:
        config.pmem_capacity = pmem_capacity
    if config.pmem_budget is None:
        config.pmem_budget = int(0.8 * pmem_capacity)

    pre_linear = LinearPipeline(
        ZeroDelayDCELinear(),
        TimedMergeLinear(),
        DeadWriteEliminationLinear(),
    )

    post_linear = LinearPipeline(
        DeadWriteEliminationLinear(),
    )

    ir_passes: list[AbsIRPass] = [
        # Legacy IRTransformer passes for InstNode/BlockNode content not yet
        # migrated to BasicBlockNode (fully-expanded loops still use InstNode).
        ZeroDelayDCELegacyPass(),
        TimedMergeLegacyPass(),
        DeadWriteEliminationLegacyPass(),
        # HIR: structural loop unrolling.
        UnrollSmallLoopPass(),
        # Post-unroll legacy cleanup (catches newly-expanded InstNode sequences).
        DeadWriteEliminationLegacyPass(),
        # Post-LIR CFG cleanup.
        DeadLabelEliminationPass(),
        BranchEliminationPass(),
        # BlockMerge fuses adjacent BasicBlockNodes, then re-runs post_linear
        # across merged boundaries.
        BlockMergePass(post_linear),
    ]

    return IRPipeLine(
        config=config,
        pre_linear=pre_linear,
        ir_passes=ir_passes,
        post_linear=post_linear,
    )
